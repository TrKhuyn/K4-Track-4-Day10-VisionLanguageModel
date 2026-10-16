"""Exercise the notebook's actual Qwen functions using a small local model.

Uses the already-cached real processor if present; never downloads model weights.
The normal Colab setup runs instructor tests only. This optional integration
test is for checking the notebook implementation on CPU before using a T4.
"""
import ast
import copy
import gc
import json
import math
import random
import re
import shutil
import time
import uuid
from pathlib import Path

import pytest
import torch
from PIL import Image

from colab_storage import (CHECKPOINT_MARKER, DriveStore, atomic_json,
                           seal_checkpoint, valid_checkpoint)
from scripts.labcommon import CLASSES
from scripts.score import parse_output

ROOT = Path(__file__).resolve().parents[1]


def notebook_functions(namespace, names):
    notebook = json.loads((ROOT / 'lab_day10.ipynb').read_text(encoding='utf-8'))
    definitions = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    # Model dimensions are small in this test. Keep the real module-count
                    # and trainable-scope checks, adjusting only the 109M parameter bound.
                    if node.name == 'apply_setup':
                        for child in ast.walk(node):
                            if isinstance(child, ast.Assert) and '100000000' in ast.unparse(child.test):
                                child.test = ast.parse('projector_n > 0', mode='eval').body
                    definitions[node.name] = node
    for name in names:
        assert name in definitions, name
        module = ast.fix_missing_locations(ast.Module(body=[definitions[name]], type_ignores=[]))
        exec(compile(module, 'notebook-functions', 'exec'), namespace)


@pytest.mark.parametrize('setup', ['E1', 'E2', 'E3'])
def test_real_processor_and_qwen_save_reload_logits(tmp_path, setup):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoProcessor, Qwen3VLConfig,
                              Qwen3VLForConditionalGeneration)
    from torch.nn.utils.rnn import pad_sequence

    cached = ROOT / 'work/processor_path.txt'
    if not cached.is_file():
        pytest.skip('Optional integration test requires the locally cached Qwen processor.')
    processor_path = Path(cached.read_text(encoding='utf-8').strip())
    if not (processor_path / 'tokenizer.json').is_file():
        pytest.skip('Cached processor is unavailable.')
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=36, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, rope_parameters=dict(rope_type='default', mrope_section=[2, 3, 3])),
        vision_config=dict(depth=24, hidden_size=16, intermediate_size=32,
                           num_heads=2, out_hidden_size=32, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, deepstack_visual_indexes=[5, 11, 17],
                           num_position_embeddings=256),
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
        vision_end_token_id=151653, eos_token_id=151645, pad_token_id=151643)
    config._attn_implementation = 'sdpa'

    def base():
        with torch.random.fork_rng():
            torch.manual_seed(5)
            return Qwen3VLForConditionalGeneration(copy.deepcopy(config))

    cfg = dict(name=setup, setup=setup, max_steps=1, train_projector=setup != 'E1',
               lora=dict(r=2, alpha=4, dropout=0.05))
    store = DriveStore(tmp_path / 'output', tmp_path / 'drive')
    namespace = dict(torch=torch, Path=Path, json=json, math=math, re=re,
                     gc=gc, copy=copy, random=random, shutil=shutil, time=time, uuid=uuid,
                     CLASSES=CLASSES, parse_output=parse_output, PROMPT='Locate road damage.',
                     LoraConfig=LoraConfig, PeftModel=PeftModel, get_peft_model=get_peft_model,
                     AutoProcessor=AutoProcessor, processor=processor, pad_sequence=pad_sequence,
                     load_base_model=base, load_config=lambda name: cfg,
                     atomic_json=atomic_json, seal_checkpoint=seal_checkpoint,
                     valid_checkpoint=valid_checkpoint, CHECKPOINT_MARKER=CHECKPOINT_MARKER,
                     STORE=store, KEEP_STEP_CHECKPOINTS=1, backup_path=store.backup,
                     config_fingerprint=lambda config: 'tiny-notebook-test')
    notebook_functions(namespace, ['yolo_to_box', 'objects_to_answer', 'user_message',
                                  'encode_example', 'collate_fn', 'is_projector',
                                  'lora_pattern', 'apply_setup', 'save_checkpoint',
                                  'load_completed_run', 'restore_checkpoint'])
    assert namespace['yolo_to_box']('crack', [.639062, .786111, .003125, 0])['bbox_2d'] == [637, 786, 641, 787]
    objects = [dict(label='crack', yolo=[.639062, .786111, .003125, 0]),
               dict(label='pothole', yolo=[.5, .5, .2, .2])]
    answer = namespace['objects_to_answer'](objects)
    assert len(json.loads(answer)) == 2
    image = Image.new('RGB', (640, 360), 'gray')
    encoded = namespace['encode_example'](image, namespace['PROMPT'], answer, processor)
    assert int((encoded['input_ids'] == processor.image_token_id).sum()) == 220
    assert processor.tokenizer.decode(encoded['labels'][encoded['labels'] != -100]) == answer + '<|im_end|>'
    batch = namespace['collate_fn']([encoded])
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = namespace['apply_setup'](base(), setup, cfg)
        loss = model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, foreach=False)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            expected = model(**batch).logits
        directory = store.output / setup.lower() / 'final'
        atomic_json(directory.parent / 'train_image_ids.json', ['test-image'])
        run = dict(config=cfg, fingerprint='tiny-notebook-test', complete=True,
                   checkpoint_format=2, completed_steps=1)
        namespace['save_checkpoint'](model, directory, run, [{'step': 1, 'loss': float(loss.detach())}],
                                     {'optimizer': optimizer.state_dict()})
        assert valid_checkpoint(directory)
        restored, _ = namespace['restore_checkpoint'](directory, cfg)
        restored.eval()
        with torch.no_grad():
            actual = restored(**batch).logits
        assert torch.equal(expected, actual), float((expected - actual).abs().max())
    finally:
        torch.set_num_threads(original_threads)
