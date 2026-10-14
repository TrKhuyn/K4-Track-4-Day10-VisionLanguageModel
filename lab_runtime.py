"""Colab implementation of GUIDE steps 1-9. Instructor files are read-only."""
import gc
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen3VLForConditionalGeneration,
                          get_cosine_schedule_with_warmup, set_seed)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.import_utils import is_torchao_available
from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, _match, parse_output
from lab_reward import iou_reward

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROMPT = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "normalized on a 0-1000 scale: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []."
)
LLM_TARGET = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGET = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def yolo_to_box(label, yolo):
    xc, yc, w, h = yolo
    coords = [int(round(max(0, min(1000, x)))) for x in
              ((xc-w/2)*1000, (yc-h/2)*1000, (xc+w/2)*1000, (yc+h/2)*1000)]
    # Very thin boxes can collapse after integer rounding (or edge clipping).
    # Keep every annotation and use the minimum representable width/height.
    # At the upper boundary the lower edge must move to 999 first.
    for lower, upper in ((0, 2), (1, 3)):
        if coords[upper] <= coords[lower]:
            coords[lower] = min(coords[lower], 999)
            coords[upper] = coords[lower] + 1
    return {"label": label, "bbox_2d": coords}


def objects_to_answer(objects):
    boxes = [yolo_to_box(obj["label"], obj["yolo"]) for obj in objects]
    boxes.sort(key=lambda box: (box["bbox_2d"][1], box["bbox_2d"][0]))
    answer = json.dumps(boxes, separators=(",", ":"))
    assert parse_output(answer) is not None, "Degenerate ground-truth box after rounding."
    return answer


def label_diagnostics(splits):
    repairs = []
    images = boxes = 0
    for split, rows in splits.items():
        for row in rows:
            images += 1
            for obj in row["objects"]:
                boxes += 1
                xc, yc, w, h = obj["yolo"]
                rounded = [int(round(max(0, min(1000, x)))) for x in
                    ((xc-w/2)*1000, (yc-h/2)*1000, (xc+w/2)*1000, (yc+h/2)*1000)]
                fixed = yolo_to_box(obj["label"], obj["yolo"])["bbox_2d"]
                if rounded != fixed:
                    repairs.append(dict(split=split, image_id=row["image_id"], label=obj["label"],
                                        before=rounded, after=fixed))
    return dict(n_images=images, n_boxes=boxes, repaired_boxes=len(repairs), repairs=repairs)


def user_message(image, prompt=PROMPT):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                           {"type": "text", "text": prompt}]}


def encode_example(image, processor, answer):
    user = user_message(image)
    options = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt = processor.apply_chat_template([user], add_generation_prompt=True, **options)
    full = processor.apply_chat_template(
        [user, {"role": "assistant", "content": answer}], add_generation_prompt=False, **options)
    length = prompt["input_ids"].shape[1]
    assert torch.equal(full["input_ids"][0, :length], prompt["input_ids"][0])
    # Qwen appends a newline after im_end; supervise JSON plus im_end only.
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    endings = (full["input_ids"][0, length:] == end_id).nonzero().flatten()
    assert len(endings) == 1
    stop = length + int(endings[0]) + 1
    result = {key: full[key][0, :stop] for key in
              ("input_ids", "attention_mask", "mm_token_type_ids") if key in full}
    result["labels"] = result["input_ids"].clone()
    result["labels"][:length] = -100
    assert int(result["labels"][-1]) == end_id
    for key in ("pixel_values", "image_grid_thw"):
        result[key] = full[key]
    return result


def collate(batch, pad_token_id):
    result = {}
    for key, pad in (("input_ids", pad_token_id), ("attention_mask", 0),
                     ("mm_token_type_ids", 0), ("labels", -100)):
        if key in batch[0]:
            assert all(key in row for row in batch)
            result[key] = torch.nn.utils.rnn.pad_sequence(
                [row[key] for row in batch], batch_first=True, padding_value=pad)
    for key in ("pixel_values", "image_grid_thw"):
        result[key] = torch.cat([row[key] for row in batch], dim=0)
    return result


def is_projector(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def apply_setup(base, cfg):
    for param in base.parameters():
        param.requires_grad_(False)
    base.enable_input_require_grads()
    targets = LLM_TARGET if cfg["lora"]["targets"] == "llm" else f"(?:{LLM_TARGET}|{VIT_TARGET})"
    matched = [name for name, _ in base.named_modules() if re.fullmatch(targets, name)]
    expected = base.config.text_config.num_hidden_layers * 7
    if cfg["lora"]["targets"] == "llm+vit":
        expected += base.config.vision_config.depth * 4
    assert len(matched) == expected, f"Matched {len(matched)}/{expected} LoRA modules."
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        target_modules=targets))
    if cfg["train_projector"]:
        count = 0
        for name, param in model.named_parameters():
            if is_projector(name):
                assert param.is_floating_point(), f"Quantized projector: {name}"
                param.requires_grad_(True)
                count += param.numel()
        assert count > 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name or (cfg["train_projector"] and is_projector(name))
            param.data = param.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    print("LoRA modules:", len(matched))
    model.print_trainable_parameters()
    return model


def restore_projector(model, directory, required):
    path = Path(directory) / "projector.pt"
    if not required:
        return
    assert path.is_file(), f"Missing trained projector: {path}"
    values = torch.load(path, map_location="cpu", weights_only=True)
    params = {name: p for name, p in model.named_parameters() if is_projector(name)}
    assert values.keys() == params.keys(), "Projector parameter names do not match."
    with torch.no_grad():
        for name, value in values.items():
            param = params[name]
            assert param.shape == value.shape
            param.data = param.data.float()
            param.copy_(value.to(param.device))


def ordered_batch(samples, seed, micro_step, size):
    # Equivalent shuffled epochs, independent of resume position.
    rows = []
    for offset in range(size):
        epoch, index = divmod(micro_step * size + offset, len(samples))
        indices = list(range(len(samples)))
        random.Random(seed + epoch).shuffle(indices)
        rows.append(samples[indices[index]])
    return rows


def optimizer_update(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not torch.isfinite(norm).item():
        # Never let an overflowing norm zero finite gradients and still mutate AdamW.
        scaler.update(new_scale=scaler.get_scale() / 2)
        optimizer.zero_grad(set_to_none=True)
        return False
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return True


class RoadDataset(Dataset):
    def __init__(self, samples, processor, data_dir):
        self.samples, self.processor, self.data_dir = samples, processor, data_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(self.data_dir / sample["image_path"]) as image:
            return encode_example(image.convert("RGB"), self.processor,
                                  objects_to_answer(sample["objects"]))


class GRPORoadDataset(Dataset):
    def __init__(self, samples, data_dir):
        self.samples, self.data_dir = samples, data_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(self.data_dir / sample["image_path"]) as source:
            image = source.convert("RGB")
        # PIL image in the conversation; TRL 1.9.2 extracts it for generation/forward.
        return {"prompt": [user_message(image)], "gt_boxes": _gt_boxes(sample),
                "image_id": sample["image_id"]}


def float32_logits_hook(module, inputs, output):
    # Keep lm_head weights/forward in fp16; log-probability and entropy math
    # over Qwen's 151936-token vocabulary must use float32 outputs.
    if getattr(output, "logits", None) is not None:
        output.logits = output.logits.float()
        if not torch.isfinite(output.logits).all().item():
            raise FloatingPointError("Nonfinite GRPO policy logits; stopped before a policy update.")
    return output


def grpo_trainer_class():
    from trl import GRPOTrainer
    class SequentialGRPOTrainer(GRPOTrainer):
        """Keep four completions per group, generate one image at a time on T4."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._logits_fp32_handle = self.model.register_forward_hook(float32_logits_hook)

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
            grid = torch.as_tensor(multimodal_fields["image_grid_thw"])
            assert grid.shape[0] == len(prompt_ids), "This lab has exactly one image per prompt."
            pixels = torch.as_tensor(multimodal_fields["pixel_values"])
            counts = grid.prod(dim=-1).tolist()
            completions, start = [], 0
            for i, count in enumerate(counts):
                fields = {}
                for key, value in multimodal_fields.items():
                    if key == "pixel_values":
                        fields[key] = pixels[start:start+count]
                    elif key == "image_grid_thw":
                        fields[key] = grid[i:i+1]
                    else:
                        # Per-token multimodal types have one entry per prompt.
                        assert len(value) == len(prompt_ids), key
                        fields[key] = value[i:i+1]
                with self.accelerator.autocast():
                    result, logprobs = super()._generate_single_turn(
                        prompt_ids[i:i+1], images[i:i+1] if images else None, fields)
                assert logprobs is None, "This path uses Transformers, without vLLM."
                completions.extend(result)
                start += count
            return completions, None
    return SequentialGRPOTrainer


def grpo_arguments(output_dir, cfg, cpu=False):
    from trl import GRPOConfig
    return GRPOConfig(output_dir=str(output_dir),
        num_generations=cfg["num_generations"], max_completion_length=cfg["max_completion_length"],
        max_steps=cfg["max_steps"], learning_rate=float(cfg["learning_rate"]), beta=float(cfg["beta"]),
        per_device_train_batch_size=1, gradient_accumulation_steps=4, generation_batch_size=4,
        fp16=not cpu, bf16=False, use_cpu=cpu, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False, use_transformers_continuous_batching=False, cast_lm_head_to_fp32=False,
        optim="adamw_torch", report_to="none", seed=42, logging_steps=1,
        save_steps=20, save_total_limit=2, remove_unused_columns=False,
        dataloader_num_workers=0, log_completions=False, temperature=1.0)


def finalize_grpo_checkpoint(directory, checkpoint, max_steps):
    """Finish export after a disconnect at the last already-completed GRPO step."""
    directory, checkpoint = Path(directory), Path(checkpoint)
    assert checkpoint.parent == directory
    marker = json.loads((checkpoint/"complete.json").read_text(encoding="utf-8"))
    info = json.loads((checkpoint/"run.json").read_text(encoding="utf-8"))
    assert marker["step"] == info["completed_steps"] == max_steps
    weights = checkpoint/"adapter_model.safetensors"
    if not weights.is_file():
        weights = checkpoint/"adapter_model.bin"
    assert weights.is_file() and (checkpoint/"adapter_config.json").is_file()
    final = directory/"final"
    (final/"adapter").mkdir(parents=True, exist_ok=True)
    for path in (weights, checkpoint/"adapter_config.json"):
        shutil.copy2(path, final/"adapter"/path.name)
    if info["config"]["train_projector"]:
        assert (checkpoint/"projector.pt").is_file()
        shutil.copy2(checkpoint/"projector.pt", final/"projector.pt")
    write_json(final/"run.json", info)
    write_json(directory/"finished.json", {"checkpoint": "final"})


class Lab:
    def __init__(self, root, drive_root, hf_token=None, batch_size=2, grad_accum=4):
        self.root = Path(root).resolve()
        self.drive_root = Path(drive_root)
        self.data_dir = self.root / "data"
        self.hf_token = hf_token
        assert torch.cuda.is_available(), "Select a Colab GPU runtime."
        assert "T4" in torch.cuda.get_device_name(0), "The lab profile targets a Colab T4."
        is_torchao_available.cache_clear()
        assert not is_torchao_available(), "Remove unused torchao in the setup cell."
        self.versions = {pkg: version(pkg) for pkg in
                         ("transformers", "peft", "trl", "accelerate", "bitsandbytes")}
        expected = dict(transformers="5.19.0", peft="0.21.2", trl="1.9.2",
                        accelerate="1.15.0", bitsandbytes="0.50.2")
        assert self.versions == expected, self.versions
        self.versions["torch"] = torch.__version__
        self.configs = {name: yaml.safe_load((self.root / f"configs/{name.lower()}.yaml")
                                           .read_text(encoding="utf-8")) for name in ("E1", "E2", "E3", "E5")}
        assert batch_size * grad_accum == 8
        for cfg in self.configs.values():
            if "batch_size" in cfg:
                cfg.update(batch_size=batch_size, grad_accum=grad_accum)
        self.splits = {name: read_jsonl(self.data_dir / f"splits/{name}.jsonl")
                       for name in ("train", "val", "test")}
        self.meta = json.loads((self.data_dir / "splits/meta.json").read_text(encoding="utf-8"))
        assert self.meta["dataset_version"].startswith("d10-")
        expected_counts = {"train": (1422, (1074, 1750, 691)),
                           "val": (276, (66, 470, 103)), "test": (311, (121, 299, 163))}
        from collections import Counter
        sessions = {}
        for split, entries in self.splits.items():
            counts = Counter(obj["label"] for row in entries for obj in row["objects"])
            assert (len(entries), tuple(counts[c] for c in CLASSES)) == expected_counts[split]
            assert len({row["image_id"] for row in entries}) == len(entries)
            assert all((row["width"], row["height"]) == (640, 360) for row in entries)
            sessions[split] = {row["session"] for row in entries}
            for row in entries:
                objects_to_answer(row["objects"])
        assert not any(sessions[a] & sessions[b] for a, b in
                       (("train", "val"), ("train", "test"), ("val", "test")))
        self.protected = self.protected_hashes()
        self.model_path, self.revision = self.prepare_model_files()
        source_files = [self.root/"lab_runtime.py", self.root/"lab_reward.py"]
        if (self.root/"run_lab.py").is_file():
            source_files.append(self.root/"run_lab.py")
        implementation = hashlib.sha256(b"".join(p.read_bytes() for p in source_files)).hexdigest()
        identity = {"dataset_version": self.meta["dataset_version"], "revision": self.revision,
                    "implementation": implementation, "configs": self.configs,
                    "versions": self.versions,
                    "protected": {k: v for k, v in self.protected.items() if k != "REPORT.md"},
                    "prompt": PROMPT}
        self.identity = identity
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        self.output = Path(drive_root) / self.meta["dataset_version"] / key
        self.output.mkdir(parents=True, exist_ok=True)
        self.local = self.root / "work" / key
        self.local.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "identity.json", identity)
        diagnostics = label_diagnostics(self.splits)
        write_json(self.output/"label_checks.json", diagnostics)
        print("Grounding label checks:", diagnostics["n_boxes"], "boxes; repaired:", diagnostics["repaired_boxes"])
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        # Validate the exact GRPOConfig API before any long experiment.
        grpo_arguments(self.local/"grpo_config_check", self.configs["E5"]["grpo"])
        print("Artifacts / checkpoints:", self.output)
        print("Training profile:", batch_size, "x", grad_accum)

    def protected_hashes(self):
        files = sorted(p for directory in ("scripts", "tests", "configs")
                       for p in (self.root / directory).rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
        files += [self.root / name for name in ("GUIDE.md", "requirements-colab.txt", "REPORT.md")]
        return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

    def prepare_model_files(self):
        from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
        revision_path = self.drive_root/self.meta["dataset_version"]/"model_revision.json"
        if revision_path.exists():
            pinned = json.loads(revision_path.read_text(encoding="utf-8"))
            assert pinned["model_id"] == MODEL_ID
            revision = pinned["revision"]
        else:
            cached = try_to_load_from_cache(MODEL_ID, "config.json")
            revision = Path(cached).parent.name if isinstance(cached, str) else None
        def retry(operation):
            for attempt in range(6):
                try:
                    return operation()
                except Exception as error:
                    chain, transient = error, False
                    while chain is not None:
                        response = getattr(chain, "response", None)
                        status = getattr(response, "status_code", None)
                        transient |= status in (429, 500, 502, 503, 504)
                        transient |= isinstance(chain, (TimeoutError, ConnectionError))
                        transient |= type(chain).__name__ in ("ConnectError", "ReadError", "TimeoutException",
                            "ConnectTimeout", "ReadTimeout", "WriteTimeout", "RemoteProtocolError")
                        transient |= bool(re.search(r"(?:429|503|502|504).*?(?:requests|error|HTTP)|(?:HTTP|status).*?(?:429|503|502|504)", str(chain), re.I))
                        chain = chain.__cause__ or chain.__context__
                    if not transient or attempt == 5:
                        raise
                    seconds = min(60 * 2**attempt, 300)
                    print(f"Hugging Face temporarily unavailable; retry in {seconds}s.")
                    for _ in range(seconds // 30):
                        time.sleep(30)
            raise RuntimeError("Download retry exhausted")
        if revision is None:
            revision = retry(lambda: HfApi().model_info(MODEL_ID, token=self.hf_token).sha)
        assert re.fullmatch(r"[0-9a-f]{40}", revision), f"Invalid model revision: {revision}"
        write_json(revision_path, dict(model_id=MODEL_ID, revision=revision))
        def fetch(filename):
            cached = try_to_load_from_cache(MODEL_ID, filename, revision=revision)
            if isinstance(cached, str) and Path(cached).is_file() and Path(cached).stat().st_size > 0:
                return Path(cached)
            return Path(retry(lambda: hf_hub_download(MODEL_ID, filename,
                                                      revision=revision, token=self.hf_token)))
        files = ("config.json", "generation_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                 "vocab.json", "merges.txt", "chat_template.json", "model.safetensors.index.json")
        for filename in files:
            path = fetch(filename)
        model_path = path.parent
        index = json.loads(path.read_text(encoding="utf-8"))
        for shard in sorted(set(index["weight_map"].values())):
            assert Path(shard).name == shard and shard.endswith(".safetensors")
            assert fetch(shard).stat().st_size > 0
        return model_path, revision

    def load_base(self):
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "model.visual", "lm_head"])
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, local_files_only=True, quantization_config=quantization,
            dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
        assert not any("4bit" in type(m).__name__.lower() for m in base.model.visual.modules())
        assert "4bit" not in type(base.lm_head).__name__.lower()
        for p in list(base.model.visual.parameters()) + list(base.lm_head.parameters()):
            assert p.is_floating_point()
            if p.dtype != torch.float16:
                p.data = p.data.to(torch.float16)
        return base

    def image(self, row):
        with Image.open(self.data_dir / row["image_path"]) as image:
            return image.convert("RGB")

    def check_labels(self):
        row = self.splits["train"][0]
        image = self.image(row)
        expected = objects_to_answer(row["objects"])
        encoded = encode_example(image, self.processor, expected)
        token = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        count = int((encoded["input_ids"] == token).sum())
        assert count == 220
        text = self.processor.tokenizer.decode(encoded["labels"][encoded["labels"] != -100])
        assert text == expected + "<|im_end|>"
        print("Visual tokens:", count, "\nSupervised answer:", text)
        write_json(self.output / "mask_check.json", dict(vision_tokens=count, supervised_answer=text))

    def preflight(self):
        fallback = self.output/"preflight_fallback.json"
        if fallback.exists():
            self.__init__(self.root, self.drive_root, self.hf_token, batch_size=1, grad_accum=8)
            self.preflight()
            return
        try:
            self.preflight_attempt()
        except torch.cuda.OutOfMemoryError as error:
            if self.configs["E3"]["batch_size"] == 1:
                raise
            import traceback
            traceback.clear_frames(error.__traceback__)
            gc.collect()
            torch.cuda.empty_cache()
            write_json(fallback, {"batch_size": 1, "grad_accum": 8})
            print("Preflight OOM: all official SFT/E4 runs switch to the shared 1 x 8 profile.")
            self.__init__(self.root, self.drive_root, self.hf_token, batch_size=1, grad_accum=8)
            self.preflight()

    def preflight_attempt(self):
        """Discarded E3 optimizer update, before E0; no test images or evidence."""
        marker = self.output/"preflight.json"
        if marker.exists():
            print("Preflight already completed for this implementation/profile.")
            return
        cfg = self.configs["E3"]
        set_seed(cfg["seed"])
        model = optimizer = scaler = batch = loss = None
        try:
            model = apply_setup(self.load_base(), cfg)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0)
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            rows = sorted(self.splits["train"], key=lambda row: len(
                self.processor.tokenizer.encode(objects_to_answer(row["objects"]))), reverse=True)
            model.train()
            torch.cuda.reset_peak_memory_stats()
            for attempt in range(20):
                optimizer.zero_grad(set_to_none=True)
                for micro in range(cfg["grad_accum"]):
                    chosen = rows[micro*cfg["batch_size"]:(micro+1)*cfg["batch_size"]]
                    batch = collate([encode_example(self.image(row), self.processor,
                        objects_to_answer(row["objects"])) for row in chosen], self.processor.tokenizer.pad_token_id)
                    batch = {k: v.to("cuda") for k, v in batch.items()}
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = model(**batch).loss / cfg["grad_accum"]
                    assert torch.isfinite(loss).item(), "Preflight loss is nonfinite."
                    scaler.scale(loss).backward()
                    batch = loss = None
                if optimizer_update(model, optimizer, scaler):
                    break
            else:
                raise FloatingPointError("Preflight gradients never became finite.")
            write_json(marker, dict(setup="E3", batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"],
                peak_vram_gb=torch.cuda.max_memory_allocated()/1024**3,
                note="Discarded probe weights; every official SFT run starts from the base model."))
            print("Preflight E3 passed; probe weights discarded.")
        finally:
            model = optimizer = scaler = batch = loss = None
            if "params" in locals():
                params.clear()
            gc.collect()
            torch.cuda.empty_cache()

    def show_samples(self):
        from IPython.display import display
        for row in random.Random(42).sample(self.splits["train"], 6):
            image = self.image(row)
            self.draw_boxes(image, _gt_boxes(row), "lime")
            print(row["image_id"], row["session"])
            display(image)

    @staticmethod
    def draw_boxes(image, boxes, color):
        draw = ImageDraw.Draw(image)
        for box in boxes:
            x1, y1, x2, y2 = [max(0, min(1000, v)) for v in box["bbox"]]
            if x2 <= x1 or y2 <= y1:
                continue
            rect = (x1*image.width/1000, y1*image.height/1000,
                    x2*image.width/1000, y2*image.height/1000)
            draw.rectangle(rect, outline=color, width=3)
            draw.text(rect[:2], box["label"].encode("ascii", "replace").decode(), fill=color)

    def load_run(self, directory, trainable=False):
        directory = Path(directory)
        info = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        model = PeftModel.from_pretrained(self.load_base(), directory / "adapter", is_trainable=trainable)
        restore_projector(model, directory, info["config"]["train_projector"])
        for name, p in model.named_parameters():
            if trainable:
                p.requires_grad_("lora_" in name)
                if p.requires_grad:
                    p.data = p.data.float()
        return model, info

    def score(self, prediction, name, split="test", info=None, out=None):
        out = Path(out) if out else self.root / f"report/evidence/{name.lower()}.json"
        cmd = [sys.executable, str(self.root / "scripts/score.py"), "--pred", str(prediction),
               "--splits-dir", str(self.data_dir / "splits"), "--name", name,
               "--split", split, "--coord", "norm1000", "--out", str(out)]
        if info:
            for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
                cmd += ["--meta", f"{key}={info[key]}"]
        subprocess.run(cmd, check=True)
        result = json.loads(out.read_text(encoding="utf-8"))
        assert result["missing_predictions"] == 0 and result["dataset_version"] == self.meta["dataset_version"]
        if split == "test":
            backup = self.output / "evidence" / out.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, backup)
        return result

    @torch.inference_mode()
    def predict(self, model, name, split, batch_size=4):
        self.processor.tokenizer.padding_side = "left"
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
        rows = self.splits[split]
        path = self.output / "predictions" / f"{name.lower()}_{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"identity": self.identity, "name": name, "split": split,
                    "ids": [row["image_id"] for row in rows], "max_new_tokens": 256}
        manifest_path = path.with_suffix(".manifest.json")
        if path.exists():
            assert manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
        else:
            write_json(manifest_path, manifest)
        done = {}
        if path.exists():
            # A disconnect can truncate the last line; preserve all complete raw outputs.
            lines = path.read_text(encoding="utf-8").splitlines()
            good = []
            for i, line in enumerate(lines):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    assert i == len(lines)-1
                    break
                assert item["image_id"] in manifest["ids"] and item["image_id"] not in done
                done[item["image_id"]] = item["raw"]
                good.append(line)
            path.write_text("".join(line+"\n" for line in good), encoding="utf-8")
        remaining = [row for row in rows if row["image_id"] not in done]
        start = 0
        progress = tqdm(total=len(remaining), desc=f"{name}/{split}")
        with path.open("a", encoding="utf-8") as file:
            while start < len(remaining):
                batch = remaining[start:start+batch_size]
                inputs = generated = None
                try:
                    inputs = self.processor.apply_chat_template(
                        [[user_message(self.image(row))] for row in batch],
                        tokenize=True, add_generation_prompt=True, return_dict=True,
                        return_tensors="pt", padding=True)
                    inputs.pop("token_type_ids", None)
                    inputs = inputs.to("cuda")
                    with torch.autocast("cuda", dtype=torch.float16):
                        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                            eos_token_id=self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
                    texts = self.processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True, clean_up_tokenization_spaces=False)
                except torch.cuda.OutOfMemoryError:
                    inputs = generated = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size//2)
                    print("Inference batch:", batch_size)
                    continue
                for row, raw in zip(batch, texts, strict=True):
                    file.write(json.dumps({"image_id": row["image_id"], "raw": raw}, ensure_ascii=False)+"\n")
                file.flush()
                start += len(batch)
                progress.update(len(batch))
                inputs = generated = None
        progress.close()
        assert len(read_jsonl(path)) == len(rows)
        return path

    def baseline(self):
        model = self.load_base()
        try:
            pred = self.predict(model, "E0", "test")
            result = self.score(pred, "E0")
            for row in read_jsonl(pred)[:5]:
                print(row["image_id"], row["raw"])
            return result
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def save_sft(self, model, optimizer, scheduler, scaler, directory, info, state):
        # A completed checkpoint becomes visible only after every artifact is written.
        directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(directory / "adapter", save_embedding_layers=False)
        if info["config"]["train_projector"]:
            torch.save({n: p.detach().cpu().clone() for n, p in model.named_parameters()
                        if is_projector(n)}, directory / "projector.pt")
        write_json(directory / "run.json", info)
        torch.save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "rng_python": random.getstate(),
                    "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all()}, directory / "training_state.pt")
        write_json(directory / "complete.json", {"step": state["step"]})

    def train_sft(self, name, cfg, fraction=1.0):
        samples = self.splits["train"]
        if fraction != 1.0:
            indices = list(range(len(samples)))
            random.Random(cfg["seed"]).shuffle(indices)
            samples = [samples[i] for i in indices[:int(len(samples)*fraction)]]
        expected = 142 if fraction == 0.1 else 1422
        assert len(samples) == expected
        cfg = deepcopy(cfg)
        manifest = {"identity": self.identity, "config": cfg, "fraction": fraction,
                    "train_ids": [row["image_id"] for row in samples]}
        directory = self.output / "runs" / name.lower()
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
        else:
            write_json(manifest_path, manifest)
        if (directory / "finished.json").exists():
            return directory
        snapshots = sorted((p for p in directory.glob("checkpoint-*") if (p/"complete.json").exists()),
                           key=lambda p: int(p.name.split("-")[-1]))
        checkpoint = snapshots[-1] if snapshots else None
        set_seed(cfg["seed"])
        model = optimizer = scheduler = scaler = batch = loss = None
        try:
            if checkpoint:
                model = PeftModel.from_pretrained(self.load_base(), checkpoint/"adapter", is_trainable=True)
                restore_projector(model, checkpoint, cfg["train_projector"])
                for n, p in model.named_parameters():
                    p.requires_grad_("lora_" in n or (cfg["train_projector"] and is_projector(n)))
                    if p.requires_grad:
                        p.data = p.data.float()
                model.enable_input_require_grads()
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.config.use_cache = False
            else:
                model = apply_setup(self.load_base(), cfg)
            self.processor.tokenizer.padding_side = "right"
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0)
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                num_warmup_steps=math.ceil(cfg["max_steps"]*.05), num_training_steps=cfg["max_steps"])
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            state = dict(step=0, micro_step=0, skipped_updates=0, losses=[], train_seconds=0.0, peak_vram_gb=0.0)
            if checkpoint:
                saved = torch.load(checkpoint/"training_state.pt", map_location="cpu", weights_only=False)
                optimizer.load_state_dict(saved.pop("optimizer"))
                scheduler.load_state_dict(saved.pop("scheduler"))
                scaler.load_state_dict(saved.pop("scaler"))
                random.setstate(saved.pop("rng_python"))
                np.random.set_state(saved.pop("rng_numpy"))
                torch.set_rng_state(saved.pop("rng_torch"))
                torch.cuda.set_rng_state_all(saved.pop("rng_cuda"))
                state = saved
                print("Resume", name, "at", state["step"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            previous_seconds = state["train_seconds"]
            consecutive_skips = 0
            progress = tqdm(total=cfg["max_steps"], initial=state["step"], desc=name)
            while state["step"] < cfg["max_steps"]:
                total_loss = 0.0
                for _ in range(cfg["grad_accum"]):
                    rows = ordered_batch(samples, cfg["seed"], state["micro_step"], cfg["batch_size"])
                    batch = collate([encode_example(self.image(row), self.processor,
                        objects_to_answer(row["objects"])) for row in rows], self.processor.tokenizer.pad_token_id)
                    batch = {key: value.to("cuda") for key, value in batch.items()}
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = model(**batch).loss / cfg["grad_accum"]
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Nonfinite loss; training stopped before saving evidence.")
                    total_loss += loss.detach().item()
                    scaler.scale(loss).backward()
                    state["micro_step"] += 1
                    batch = loss = None
                if not optimizer_update(model, optimizer, scaler):
                    state["skipped_updates"] += 1
                    consecutive_skips += 1
                    if consecutive_skips >= 20:
                        raise FloatingPointError("20 consecutive gradient overflows; training stopped.")
                    continue
                consecutive_skips = 0
                scheduler.step()
                state["step"] += 1
                state["losses"].append(dict(step=state["step"], loss=total_loss))
                progress.update(1)
                progress.set_postfix(loss=f"{total_loss:.4f}")
                if state["step"] % 100 == 0 or state["step"] == cfg["max_steps"]:
                    torch.cuda.synchronize()
                    state["train_seconds"] = previous_seconds + time.perf_counter()-start_time
                    state["peak_vram_gb"] = max(state["peak_vram_gb"], torch.cuda.max_memory_allocated()/1024**3)
                    info = dict(name=name, config=cfg, prompt=PROMPT, coord="norm1000", train_fraction=fraction,
                        n_train_images=len(samples), completed_steps=state["step"],
                        trainable_params=sum(p.numel() for p in params), peak_vram_gb=state["peak_vram_gb"],
                        train_minutes=state["train_seconds"]/60, loss_history=state["losses"],
                        skipped_updates=state["skipped_updates"], dataset_version=self.meta["dataset_version"],
                        model_revision=self.revision)
                    snapshot = directory / f"checkpoint-{state['step']}"
                    self.save_sft(model, optimizer, scheduler, scaler, snapshot, info, state)
                    # Keep one completed snapshot; no removal outside this run's own directory.
                    for old in snapshots:
                        if old != snapshot:
                            shutil.rmtree(old)
                    snapshots = [snapshot]
            progress.close()
            final = snapshots[-1]
            # Final artifacts live in the same completed snapshot, avoiding a second weight copy.
            write_json(directory / "finished.json", {"checkpoint": final.name})
            return directory
        finally:
            model = optimizer = scheduler = scaler = batch = loss = None
            if "params" in locals():
                params.clear()
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def artifact_dir(run):
        marker = json.loads((Path(run)/"finished.json").read_text(encoding="utf-8"))
        name = marker["checkpoint"]
        assert Path(name).name == name
        return Path(run)/name

    def evaluate_run(self, run, name, splits=("val", "test")):
        model, info = self.load_run(self.artifact_dir(run))
        results = {}
        try:
            for split in splits:
                pred = self.predict(model, name, split)
                out = self.output/f"{name.lower()}_val_score.json" if split == "val" else None
                results[split] = self.score(pred, name, split, info, out)
            return results
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def sft_experiments(self):
        self.runs, self.results = {}, {}
        for name in ("E1", "E2", "E3"):
            run = self.train_sft(name, self.configs[name])
            self.runs[name] = run
            self.results[name] = self.evaluate_run(run, name)
        self.best = max(("E1", "E2", "E3"), key=lambda name: self.results[name]["val"]["micro"]["f1"])
        write_json(self.output / "sft_selection.json", dict(best=self.best, selection_split="val",
                   scores={n: r["val"]["micro"]["f1"] for n, r in self.results.items()}))
        print("Best SFT on validation:", self.best)

    def ablation(self):
        cfg = deepcopy(self.configs[self.best])
        run = self.train_sft("E4_data_0.1", cfg, fraction=0.1)
        small = self.evaluate_run(run, "E4_data_0.1", splits=("test",))["test"]
        info = json.loads((self.artifact_dir(self.runs[self.best])/"run.json").read_text(encoding="utf-8"))
        # GUIDE allows one branch to be the original setup. Score the original raw output.
        pred = self.output / "predictions" / f"{self.best.lower()}_test.jsonl"
        full = self.score(pred, "E4_data_1.0", info=info)
        assert full["micro"] == self.results[self.best]["test"]["micro"]
        assert full["f1_ci95"] == self.results[self.best]["test"]["f1_ci95"]
        overlap = max(small["f1_ci95"][0], full["f1_ci95"][0]) <= min(small["f1_ci95"][1], full["f1_ci95"][1])
        analysis = dict(variable="data_size", baseline=self.best, values=[0.1, 1.0],
            train_images=[142, 1422], configuration=cfg, reused_full_run=str(self.runs[self.best]),
            ci95_overlap=overlap, delta_f1=full["micro"]["f1"]-small["micro"]["f1"])
        write_json(self.output / "e4_analysis.json", analysis)
        print("E4 CI95 overlap:", overlap, "; not a significance test of the F1 difference.")

    def grpo(self):
        from transformers import TrainerCallback
        cfg = self.configs["E5"]["grpo"]
        directory = self.output / "runs/e5"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = dict(identity=self.identity, best_sft=self.best, grpo=cfg,
                        train_projector=False, initialization=str(self.artifact_dir(self.runs[self.best])))
        if (directory/"manifest.json").exists():
            assert json.loads((directory/"manifest.json").read_text(encoding="utf-8")) == manifest
        else:
            write_json(directory/"manifest.json", manifest)
        checkpoints = sorted((p for p in directory.glob("checkpoint-*") if (p/"complete.json").exists()),
                             key=lambda p: int(p.name.split("-")[-1]))
        if not (directory/"finished.json").exists() and checkpoints:
            latest = checkpoints[-1]
            saved = json.loads((latest/"run.json").read_text(encoding="utf-8"))
            if saved["completed_steps"] == cfg["max_steps"]:
                finalize_grpo_checkpoint(directory, latest, cfg["max_steps"])
                print("GRPO last step was already complete; final artifacts restored without another update.")
        if not (directory/"finished.json").exists():
            model, source_info = self.load_run(self.artifact_dir(self.runs[self.best]), trainable=True)
            self.processor.tokenizer.padding_side = "left"
            model.enable_input_require_grads()
            model.config.use_cache = False
            owner = self
            class ResourceLog(TrainerCallback):
                def __init__(self):
                    self.started = None
                    self.previous_minutes = 0.0
                    self.previous_peak = 0.0
                    self.skipped_updates = 0
                def on_train_begin(self, args, state, control, **kwargs):
                    self.started = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                def persist(self, state, path):
                    info = dict(name="E5", config=deepcopy(source_info["config"]), grpo=cfg,
                        best_sft=owner.best, prompt=PROMPT, coord="norm1000", completed_steps=state.global_step,
                        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                        train_minutes=self.previous_minutes+(time.perf_counter()-self.started)/60,
                        peak_vram_gb=max(self.previous_peak, torch.cuda.max_memory_allocated()/1024**3),
                        dataset_version=owner.meta["dataset_version"], model_revision=owner.revision,
                        train_projector_in_grpo=False, loss_history=state.log_history,
                        optimizer_skipped_updates=self.skipped_updates)
                    write_json(path/"run.json", info)
                    # Preserve trained SFT projector, even though it is frozen in GRPO.
                    if source_info["config"]["train_projector"]:
                        torch.save({n: p.detach().cpu().clone() for n, p in model.named_parameters()
                                    if is_projector(n)}, path/"projector.pt")
                def on_save(self, args, state, control, **kwargs):
                    self.persist(state, directory/f"checkpoint-{state.global_step}")
                    write_json(directory/f"checkpoint-{state.global_step}"/"complete.json",
                               {"step": state.global_step})
                def on_step_end(self, args, state, control, **kwargs):
                    if trainer.accelerator.optimizer_step_was_skipped:
                        self.skipped_updates += 1
            callback = ResourceLog()
            checkpoints = sorted((p for p in directory.glob("checkpoint-*") if (p/"complete.json").exists()),
                                 key=lambda p: int(p.name.split("-")[-1]))
            resume = checkpoints[-1] if checkpoints else None
            if resume:
                saved = json.loads((resume/"run.json").read_text(encoding="utf-8"))
                callback.previous_minutes, callback.previous_peak = saved["train_minutes"], saved["peak_vram_gb"]
                callback.skipped_updates = saved.get("optimizer_skipped_updates", 0)
            args = grpo_arguments(directory, cfg)
            trainer = None
            try:
                trainer = grpo_trainer_class()(model=model, args=args, reward_funcs=iou_reward,
                    train_dataset=GRPORoadDataset(self.splits["train"], self.data_dir),
                    processing_class=self.processor, callbacks=[callback])
                # TRL 1.9.2 upcasts QLoRA trainables to bf16 unconditionally.
                # T4 training uses fp16 autocast + GradScaler, with fp32 master weights.
                # Restore dtype BEFORE Trainer creates its optimizer.
                for p in model.parameters():
                    if p.requires_grad:
                        p.data = p.data.float()
                if trainer.accelerator.scaler is not None:
                    trainer.accelerator.scaler = torch.amp.GradScaler("cuda", init_scale=1024)
                assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
                trainer.train(resume_from_checkpoint=str(resume) if resume else None)
                assert trainer.state.global_step == cfg["max_steps"]
                final = directory/"final"
                final.mkdir(exist_ok=True)
                model.save_pretrained(final/"adapter", save_embedding_layers=False)
                callback.persist(trainer.state, final)
                write_json(directory/"finished.json", {"checkpoint": "final"})
            finally:
                trainer = model = callback = None
                p = None
                gc.collect()
                torch.cuda.empty_cache()
        self.e5_result = self.evaluate_run(directory, "E5", splits=("test",))["test"]
        self.reward_analysis()

    def reward_analysis(self):
        from lab_reward import compute_reward
        by_id = {row["image_id"]: row for row in self.splits["test"]}
        stats = {}
        for name in (self.best, "E5"):
            values = []
            empty_on_damage = empty = giant = invalid = 0
            for row in read_jsonl(self.output/"predictions"/f"{name.lower()}_test.jsonl"):
                gt = _gt_boxes(by_id[row["image_id"]])
                pred = parse_output(row["raw"])
                invalid += pred is None
                if pred == []:
                    empty += 1
                    empty_on_damage += bool(gt)
                if pred:
                    giant += any((b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1]) >= 800000 for b in pred)
                values.append(compute_reward(row["raw"], gt))
            stats[name] = dict(mean_iou_reward=sum(values)/len(values), invalid_json=invalid,
                               empty_predictions=empty, empty_on_damage=empty_on_damage,
                               images_with_giant_box=giant, giant_area_threshold=0.8)
        sft = self.results[self.best]["test"]
        stats["comparison"] = dict(best_sft=self.best,
            delta_f1=self.e5_result["micro"]["f1"]-sft["micro"]["f1"],
            sft_ci95=sft["f1_ci95"], grpo_ci95=self.e5_result["f1_ci95"])
        write_json(self.output/"e5_analysis.json", stats)
        print(json.dumps(stats, indent=2))

    def export(self):
        assert self.protected_hashes() == self.protected, "An instructor file or REPORT was changed."
        figures = self.root/"report/figures"
        figures.mkdir(parents=True, exist_ok=True)
        predictions = {r["image_id"]: r["raw"] for r in read_jsonl(
            self.output/"predictions"/f"{self.best.lower()}_test.jsonl")}
        ranked = []
        for row in self.splits["test"]:
            pred = parse_output(predictions[row["image_id"]])
            counts, _ = _match(_gt_boxes(row), pred or [])
            tp, fp, fn = (sum(v[i] for v in counts.values()) for i in range(3))
            ranked.append((tp, fp+fn, row, pred or []))
        correct = sorted((x for x in ranked if x[0] > 0 and x[1] == 0), key=lambda x: -x[0])[:3]
        errors = sorted((x for x in ranked if x[1] > 0), key=lambda x: -x[1])[:3]
        chosen = correct+errors
        for item in sorted(ranked, key=lambda x: -x[0]):
            if len(chosen) >= 6:
                break
            if all(item[2]["image_id"] != x[2]["image_id"] for x in chosen):
                chosen.append(item)
        captions = []
        figure_paths = []
        for i, (tp, errors_count, row, pred) in enumerate(chosen):
            image = self.image(row)
            self.draw_boxes(image, _gt_boxes(row), "lime")
            self.draw_boxes(image, pred, "red")
            path = figures/f"prediction_{i+1}.png"
            image.save(path)
            figure_paths.append(path)
            captions.append(dict(file=path.name, image_id=row["image_id"], model=self.best,
                                 tp=tp, fp_plus_fn=errors_count, gt_color="lime", pred_color="red"))
        write_json(figures/"captions.json", captions)
        evidence_paths = [self.root/f"report/evidence/{name}.json" for name in
                          ("e0", "e1", "e2", "e3", "e4_data_0.1", "e4_data_1.0", "e5")]
        summary = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in evidence_paths}
        write_json(self.output/"summary.json", summary)
        archive = self.local/"day10_results.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for path in evidence_paths + figure_paths + [figures/"captions.json"]:
                zip_file.write(path, str(path.relative_to(self.root)))
            for path in self.output.glob("*.json"):
                zip_file.write(path, "analysis/"+path.name)
            for path in (self.output/"runs").rglob("run.json"):
                zip_file.write(path, "analysis/"+str(path.relative_to(self.output)))
            for path in (self.output/"predictions").glob("*.jsonl"):
                zip_file.write(path, "predictions/"+path.name)
        shutil.copy2(archive, self.output/archive.name)
        print("Results ZIP:", archive, "\nDrive copy:", self.output/archive.name)
        print("REPORT.md is unchanged; fill it using summary.json and the real evidence.")
        from google.colab import files
        try:
            files.download(str(archive))
        except Exception as error:
            print("Browser download could not be started; use the saved Drive ZIP:", type(error).__name__)
        return archive
