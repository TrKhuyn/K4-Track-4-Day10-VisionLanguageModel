"""Run GUIDE stages from Colab: %run -i run_lab.py --stage <name>."""
import argparse
import json
import os
from pathlib import Path

from lab_runtime import Lab
from scripts.labcommon import read_jsonl


def restore_selection(lab):
    """Reopen existing SFT results if E4/E5 is run in a later kernel session."""
    lab.runs, lab.results = {}, {}
    for name in ("E1", "E2", "E3"):
        lab.runs[name] = lab.output/"runs"/name.lower()
        artifact = Lab.artifact_dir(lab.runs[name])
        info = json.loads((artifact/"run.json").read_text(encoding="utf-8"))
        assert info["dataset_version"] == lab.meta["dataset_version"]
        assert info["completed_steps"] == info["config"]["max_steps"]
        assert (artifact/"adapter/adapter_config.json").is_file()
        assert any((artifact/"adapter"/file).is_file() and (artifact/"adapter"/file).stat().st_size > 0
                   for file in ("adapter_model.safetensors", "adapter_model.bin")), f"Missing adapter for {name}"
        if info["config"]["train_projector"]:
            assert (artifact/"projector.pt").is_file() and (artifact/"projector.pt").stat().st_size > 0
        lab.results[name] = {}
        for split in ("val", "test"):
            path = (lab.output/f"{name.lower()}_val_score.json" if split == "val"
                    else lab.output/"evidence"/f"{name.lower()}.json")
            score = json.loads(path.read_text(encoding="utf-8"))
            assert score["dataset_version"] == lab.meta["dataset_version"]
            assert score["split"] == split and score["missing_predictions"] == 0
            lab.results[name][split] = score
    # Selection always uses validation, including when resuming E4 or E5.
    lab.best = max(("E1", "E2", "E3"),
                   key=lambda name: lab.results[name]["val"]["micro"]["f1"])


def main(namespace=None, argv=None):
    namespace = namespace if namespace is not None else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(
        "data", "labels", "e0", "raw-e0", "score-e0", "mask", "setup",
        "sft", "summary", "e4", "e5", "export"))
    parser.add_argument("--root", default=str(namespace.get("ROOT", Path.cwd())))
    parser.add_argument("--drive-root", default=str(namespace.get(
        "DRIVE_ROOT", "/content/drive/MyDrive/day10-vlm")))
    parser.add_argument("--batch-size", type=int, default=namespace.get("TRAIN_BATCH_SIZE", 2))
    parser.add_argument("--grad-accum", type=int, default=namespace.get("GRAD_ACCUM", 4))
    args = parser.parse_args(argv)
    key = (str(Path(args.root).resolve()), str(Path(args.drive_root)), args.batch_size, args.grad_accum)
    lab = namespace.get("_DAY10_LAB")
    if lab is None or namespace.get("_DAY10_REQUEST") != key:
        lab = Lab(args.root, args.drive_root, hf_token=os.environ.get("HF_TOKEN"),
                  batch_size=args.batch_size, grad_accum=args.grad_accum)
        namespace["_DAY10_LAB"] = lab
        namespace["_DAY10_REQUEST"] = key
    stage = args.stage
    if stage == "data":
        print(json.dumps(lab.meta, ensure_ascii=False, indent=2))
        lab.show_samples()
    elif stage == "labels":
        lab.preflight()
        lab.check_labels()
    elif stage == "e0":
        lab.preflight()
        lab.baseline()
    elif stage == "raw-e0":
        for row in read_jsonl(lab.output/"predictions/e0_test.jsonl")[:5]:
            print(row["image_id"], row["raw"], sep="\n")
    elif stage == "score-e0":
        score = json.loads((lab.output/"evidence/e0.json").read_text(encoding="utf-8"))
        print("E0 F1:", score["micro"]["f1"], "CI95:", score["f1_ci95"],
              "parse_valid_rate:", score["parse_valid_rate"])
    elif stage == "mask":
        lab.check_labels()
    elif stage == "setup":
        for name in ("E1", "E2", "E3"):
            print(name, lab.configs[name])
    elif stage == "sft":
        lab.preflight()
        lab.sft_experiments()
    elif stage == "summary":
        restore_selection(lab)
        for name in ("E1", "E2", "E3"):
            score = lab.results[name]["test"]
            print(name, "F1:", score["micro"]["f1"], "CI95:", score["f1_ci95"],
                  "parse_valid_rate:", score["parse_valid_rate"], "resources:", score["meta"])
        e0 = json.loads((lab.output/"evidence/e0.json").read_text(encoding="utf-8"))
        e1 = lab.results["E1"]["test"]
        if e1["parse_valid_rate"] < 0.95 or e1["micro"]["f1"]-e0["micro"]["f1"] < 0.10:
            print("Actual E1 results have not met the rubric thresholds. Check the evidence before submission.")
    elif stage in ("e4", "e5"):
        lab.preflight()
        restore_selection(lab)
        if stage == "e4":
            lab.ablation()
        else:
            lab.grpo()
    elif stage == "export":
        restore_selection(lab)
        # Recreate the local evidence directory when using a fresh clone/kernel.
        import shutil
        target = lab.root/"report/evidence"
        target.mkdir(parents=True, exist_ok=True)
        for path in (lab.output/"evidence").glob("*.json"):
            shutil.copy2(path, target/path.name)
        namespace["results_zip"] = lab.export()


if __name__ == "__main__":
    # IPython's %run -i preserves this namespace between notebook stages,
    # so GPU models keep the same lifecycle as the tested Lab implementation.
    _DAY10_STAGE_OK = False
    main(globals())
    _DAY10_STAGE_OK = True
