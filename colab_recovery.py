"""Restore completed E1--E3 artifacts from an older Drive run before E4.

This file is embedded as one Colab cell by ``prepare_colab.py``.  It only
copies missing files, so a current run always takes priority over an older one.
"""
from pathlib import Path
import shutil


def _copy_missing(source, destination):
    """Copy a file or directory tree without replacing an existing artifact."""
    source, destination = Path(source), Path(destination)
    if source.is_file():
        files = [(source, destination)]
    elif source.is_dir():
        files = [(path, destination / path.relative_to(source))
                 for path in sorted(source.rglob('*')) if path.is_file()]
    else:
        return 0
    copied = 0
    for old, new in files:
        if old.name.endswith(('.tmp', '.partial', '.uploading')) or new.exists():
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        temporary = new.with_name(new.name + '.recovering')
        shutil.copy2(old, temporary)
        temporary.replace(new)
        copied += 1
    return copied


def restore_completed_baselines(source_root):
    """Restore only the final E1--E3 artifacts that have test evidence."""
    source_root = Path(source_root).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(f'Không thấy run cũ: {source_root}')
    copied = 0
    recovered = []
    mode = 'full' if RUN_MODE == 'full' else 'smoke'
    for name in ('E1', 'E2', 'E3'):
        key = name.lower()
        source_evidence = source_root / 'report' / 'evidence' / f'{key}.json'
        if not source_evidence.is_file():
            print(f'{name}: run cũ không có evidence, không khôi phục.')
            continue
        artifacts = (
            (source_root / 'work' / mode / key / 'final', run_directory(name) / 'final'),
            (source_root / 'predictions' / RUN_MODE / f'{key}_val.jsonl', prediction_path(name, 'val')),
            (source_root / 'predictions' / RUN_MODE / f'{key}_val.manifest.json',
             prediction_path(name, 'val').with_suffix('.manifest.json')),
            (source_root / 'predictions' / RUN_MODE / f'{key}_val.metrics.json',
             prediction_path(name, 'val').with_suffix('.metrics.json')),
            (source_root / 'predictions' / RUN_MODE / f'{key}_test.jsonl', prediction_path(name, 'test')),
            (source_root / 'predictions' / RUN_MODE / f'{key}_test.manifest.json',
             prediction_path(name, 'test').with_suffix('.manifest.json')),
            (source_evidence, EVIDENCE_ROOT / f'{key}.json'),
        )
        for source, destination in artifacts:
            copied += _copy_missing(source, destination)
        run = load_completed_run(run_directory(name) / 'final', load_config(name))
        if run is None:
            raise RuntimeError(
                f'{name}: evidence có mặt nhưng checkpoint final không đầy đủ hoặc không khớp cấu hình. '
                'Dùng RUN_LABEL/cấu hình của run cũ, hoặc chọn run cũ cùng phiên bản notebook.')
        recovered.append(name)
    print(f'Khôi phục trước E4 từ {source_root}: {copied} file; baseline hợp lệ: {recovered or "không có"}.')
    return recovered


# None: dùng SAVED_ROOT, vốn đã được khôi phục tự động ở đầu notebook.
# Muốn dùng run có nhãn khác, đổi thành Path('/content/drive/MyDrive/lab_day10_colab_runs/<run_cu>').
RECOVERY_SOURCE_ROOT = SAVED_ROOT if RECOVERY_SOURCE_ROOT is None else Path(RECOVERY_SOURCE_ROOT)
if 'E4' in RUNS:
    restore_completed_baselines(RECOVERY_SOURCE_ROOT)
else:
    print('Không chạy E4 nên không cần khôi phục baseline cũ.')
