"""Run cached E4 -> scorer -> plots -> ZIP in a temporary synthetic workspace.

This checks notebook control flow and file contracts; no synthetic result is
written to the repository report/evidence directory or used as training evidence.
"""
import ast
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
import yaml
from PIL import Image

from colab_storage import (CHECKPOINT_MARKER, DriveStore, atomic_json,
                           seal_checkpoint, valid_checkpoint)
from scripts.labcommon import CLASSES, read_jsonl, write_jsonl
from scripts.score import evaluate, parse_output

ROOT = Path(__file__).resolve().parents[1]


def test_cached_e4_scoring_figures_and_report_export(tmp_path, monkeypatch):
    notebook = json.loads((ROOT / 'lab_day10.ipynb').read_text(encoding='utf-8'))
    cells = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    output, repo, data = tmp_path / 'output', tmp_path / 'repo', tmp_path / 'data'
    store = DriveStore(output, tmp_path / 'drive')
    for folder in ('configs', 'scripts'):
        shutil.copytree(ROOT / folder, repo / folder, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / 'REPORT.md', repo / 'REPORT.md')
    output.mkdir()
    shutil.copy2(ROOT / 'REPORT.md', output / 'REPORT.md')
    rows = [dict(image_id=str(i), image_path=f'images/{i}.jpg', width=640, height=360,
                 objects=[dict(label='pothole', yolo=[.5, .5, .2, .2])]) for i in range(6)]
    (data / 'images').mkdir(parents=True)
    for row in rows:
        Image.new('RGB', (640, 360), 'gray').save(data / row['image_path'])
    for split in ('train', 'val', 'test'):
        write_jsonl(data / f'splits/{split}.jsonl', rows)
    atomic_json(data / 'splits/meta.json', {'dataset_version': 'synthetic-unit-test'})
    report = output / 'report'
    evidence = report / 'evidence'
    evidence.mkdir(parents=True)
    work = output / 'work/full'

    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    namespace = dict(Path=Path, json=json, math=math, copy=copy, shutil=shutil, subprocess=subprocess,
                     sys=sys, yaml=yaml, plt=plt, CLASSES=CLASSES, read_jsonl=read_jsonl,
                     parse_output=parse_output, evaluate=evaluate, fingerprint=fingerprint,
                     config_fingerprint=fingerprint, valid_checkpoint=valid_checkpoint,
                     atomic_json=atomic_json, STORE=store, backup_path=store.backup,
                     CHECKPOINT_MARKER=CHECKPOINT_MARKER,
                     REPO_ROOT=repo, DATA_ROOT=data, OUTPUT_ROOT=output, REPORT_ROOT=report,
                     EVIDENCE_ROOT=evidence, WORK_ROOT=output / 'work',
                     PREDICTION_ROOT=output / 'predictions', SAVED_ROOT=store.saved,
                     RUN_MODE='full', RUNS=('E0', 'E1', 'E2', 'E3', 'E4'),
                     E4_BASE_SETUP='auto', DATASET_VERSION='synthetic-unit-test',
                     VAL_LIMIT=40, MAX_NEW_TOKENS=256, TRAIN_BATCH_SIZE=1, GRAD_ACCUM=8,
                     MAKE_FIGURES=True, SAVE_TO_DRIVE=True, train_samples=rows,
                     val_samples=rows, test_samples=rows, RESULTS={},
                     display=lambda value: None, Markdown=lambda value: value,
                     load_image=lambda row: Image.open(data / row['image_path']).convert('RGB'))
    selected = {'prediction_path', 'prediction_manifest', 'valid_prediction_file',
                'score_predictions', 'run_directory', 'load_config', 'load_completed_run',
                'infer_checkpoint', 'execute_setup', 'choose_e4_base', 'evidence_table',
                'visualize_predictions', 'draw_boxes', 'yolo_to_box', 'objects_to_answer'}
    for source in cells:
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in selected:
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'notebook-export-test', 'exec'), namespace)
    assert selected <= namespace.keys()
    monkeypatch.setattr(plt, 'show', lambda: None)

    def seed_run(name, config, good_val):
        directory = namespace['run_directory'](name) / 'final'
        (directory / 'adapter').mkdir(parents=True)
        (directory / 'adapter/adapter_config.json').write_text('{}')
        (directory / 'adapter/adapter_model.safetensors').write_bytes(b'unit-test-only')
        (directory / 'training_state.pt').write_bytes(b'unit-test-only')
        (directory / 'loss.jsonl').write_text('{"step":1,"loss":1.0}\n')
        if config.get('train_projector'):
            (directory / 'extra_trainable.pt').write_bytes(b'unit-test-only')
        run = dict(config=config, fingerprint=fingerprint(config), run_id=name + '-test',
                   complete=True, checkpoint_format=2, completed_steps=config['max_steps'],
                   trainable_params=100, peak_vram_gb=1., train_minutes=1.,
                   dataset_version=namespace['DATASET_VERSION'])
        atomic_json(directory / 'run.json', run)
        seal_checkpoint(directory, run)
        for split in ('val', 'test'):
            path = namespace['prediction_path'](name, split)
            good_count = good_val if split == 'val' else 3
            predictions = [{'image_id': row['image_id'], 'raw': (
                '[{"label":"pothole","bbox_2d":[400,400,600,600]}]' if i < good_count else '[]')}
                for i, row in enumerate(rows)]
            write_jsonl(path, predictions)
            atomic_json(path.with_suffix('.manifest.json'), namespace['prediction_manifest'](run, rows, split))
        return directory, run

    directories = {}
    for name, good_val in [('E1', 2), ('E2', 4), ('E3', 1)]:
        config = namespace['load_config'](name)
        directory, run = seed_run(name, config, good_val)
        directories[name] = directory
        # Exact notebook inference uses cached predictions and must not load a model.
        namespace['restore_checkpoint'] = lambda *args: pytest.fail('Cached predictions must be reused')
        namespace['infer_checkpoint'](directory, config, 'val')
        namespace['score_predictions'](namespace['prediction_path'](name, 'test'), name, run)
    assert namespace['choose_e4_base']() == 'E2'
    config = namespace['load_config']('E2')
    config.update(name='E4_data_size_0.1', train_fraction=.1)
    directory, _ = seed_run('E4_data_size_0.1', config, 2)
    namespace['train_experiment'] = lambda requested: directory
    e4_source = next(source for source in cells if source.startswith('def choose_e4_base():'))
    exec(compile(e4_source, 'notebook-e4-test', 'exec'), namespace)
    assert namespace['RESULTS']['E4_data_size_0.1']['test']['name'] == 'E4_data_size_0.1'
    assert namespace['RESULTS']['E4_data_size_1.0']['meta']['base_setup'] == 'E2'
    e0_config = dict(name='E0', setup='E0', train_fraction=1., max_steps=0, train_projector=False)
    _, e0_run = seed_run('E0', e0_config, 0)
    namespace['score_predictions'](namespace['prediction_path']('E0', 'test'), 'E0', e0_run)
    plotting = next(source for source in cells if source.startswith("if RUN_MODE == 'full':\n    evidence_table()"))
    exec(compile(plotting, 'notebook-plots-test', 'exec'), namespace)
    exporting = next(source for source in cells if source.startswith('# Đồng bộ evidence'))
    exec(compile(exporting, 'notebook-export-test', 'exec'), namespace)
    with zipfile.ZipFile(output / 'report_bundle.zip') as archive:
        names = archive.namelist()
        assert 'report/evidence/e4_data_size_0.1.json' in names
        assert 'report/evidence/e4_data_size_1.0.json' in names
        assert 'report/results_table.md' in names
        assert sum(name.endswith('.png') for name in names) >= 30
        assert not any(name.endswith(('.pt', '.safetensors')) for name in names)
    assert (store.saved / 'report_bundle.zip').exists()
    assert not (output / 'report_bundle.zip.tmp').exists()
