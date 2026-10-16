"""Verify the exact embedded Colab code without downloading the 4B model."""
import ast
import contextlib
import copy
import gc
import hashlib
import json
import random
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from colab_storage import (CHECKPOINT_MARKER, DriveStore, atomic_json,
                           seal_checkpoint, valid_checkpoint)
from prepare_colab import synced_notebook
from scripts.labcommon import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_embedded_code_is_current_and_has_pre_e4_recovery():
    notebook = json.loads((ROOT / 'lab_day10.ipynb').read_text(encoding='utf-8'))
    assert synced_notebook(copy.deepcopy(notebook)) == notebook
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            ast.parse(source)
            assert 'E4_ARTIFACT_RECOVERY' not in source
            assert 'nthanhwork' not in source
    recovery_cells = [cell for cell in notebook['cells']
                      if cell['metadata'].get('colab_source') == 'colab_recovery.py']
    assert len(recovery_cells) == 1
    assert ''.join(recovery_cells[0]['source']) == (ROOT / 'colab_recovery.py').read_text(encoding='utf-8')
    recovery = json.loads((ROOT / 'lab_day10_e4_recovery.ipynb').read_text(encoding='utf-8'))
    assert any('E4_ARTIFACT_RECOVERY' in ''.join(c['source']) for c in recovery['cells'])


def test_evidence_skips_e1_e2_e3_and_recovery_copies_only_missing_artifacts(tmp_path):
    notebook = json.loads((ROOT / 'lab_day10.ipynb').read_text(encoding='utf-8'))
    cells = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    evidence = tmp_path / 'output/report/evidence'
    evidence.mkdir(parents=True)
    calls = []
    namespace = {'EVIDENCE_ROOT': evidence, 'RUNS': ('E1', 'E2', 'E3'), 'RESULTS': {},
                 'execute_setup': lambda name: calls.append(name)}
    for name in ('E1', 'E2', 'E3'):
        (evidence / f'{name.lower()}.json').write_text('{}')
        source = next(s for s in cells if s.startswith(f"if '{name}' in RUNS and"))
        exec(compile(source, 'notebook-evidence-skip-test', 'exec'), namespace)
    assert calls == []

    tree = ast.parse((ROOT / 'colab_recovery.py').read_text(encoding='utf-8'))
    recovery_namespace = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'colab-recovery-test', 'exec'),
                 recovery_namespace)
    old, output = tmp_path / 'old-run', tmp_path / 'new-output'
    (old / 'report/evidence').mkdir(parents=True)
    (old / 'report/evidence/e1.json').write_text('{"name":"E1"}')
    final = old / 'work/full/e1/final'
    (final / 'adapter').mkdir(parents=True)
    (final / 'adapter/adapter_model.safetensors').write_bytes(b'old-weights')
    (final / 'run.json').write_text('{}')
    prediction_root = old / 'predictions/full'
    prediction_root.mkdir(parents=True)
    for suffix in ('val.jsonl', 'val.manifest.json', 'val.metrics.json', 'test.jsonl', 'test.manifest.json'):
        (prediction_root / f'e1_{suffix}').write_text(suffix)
    current_val = output / 'predictions/full/e1_val.jsonl'
    current_val.parent.mkdir(parents=True)
    current_val.write_text('current-result')
    recovery_namespace.update(
        RUN_MODE='full', EVIDENCE_ROOT=output / 'report/evidence',
        run_directory=lambda name: output / f'work/full/{name.lower()}',
        prediction_path=lambda name, split: output / f'predictions/full/{name.lower()}_{split}.jsonl',
        load_config=lambda name: {'name': name},
        load_completed_run=lambda directory, config: {'ok': True} if (Path(directory) / 'run.json').is_file() else None,
    )
    assert recovery_namespace['restore_completed_baselines'](old) == ['E1']
    assert current_val.read_text() == 'current-result'
    assert (output / 'work/full/e1/final/adapter/adapter_model.safetensors').read_bytes() == b'old-weights'
    assert (output / 'report/evidence/e1.json').is_file()
    assert recovery_namespace['restore_completed_baselines'](old) == ['E1']


def make_checkpoint(directory, step=2):
    (directory / 'adapter').mkdir(parents=True)
    (directory / 'adapter/adapter_config.json').write_text('{}')
    (directory / 'adapter/adapter_model.safetensors').write_bytes(b'fake-test-weights')
    (directory / 'training_state.pt').write_bytes(b'fake-test-state')
    run = {'completed_steps': step, 'fingerprint': 'test', 'checkpoint_format': 2,
           'config': {'train_projector': False}}
    atomic_json(directory / 'run.json', run)
    seal_checkpoint(directory, run)


def test_drive_restore_rejects_incomplete_snapshot_and_repairs_local_copy(tmp_path):
    output, drive = tmp_path / 'output', tmp_path / 'drive'
    snapshot = output / 'work/e1/step-0002'
    make_checkpoint(snapshot)
    store = DriveStore(output, drive)
    store.backup(snapshot)
    assert valid_checkpoint(drive / 'work/e1/step-0002')
    # Simulate an interrupted later Drive upload.
    partial = drive / 'work/e1/step-0003'
    partial.mkdir()
    (partial / 'training_state.pt.uploading').write_bytes(b'incomplete')
    atomic_json(partial / 'run.json', {'checkpoint_format': 2})
    restored = tmp_path / 'fresh'
    broken = restored / 'work/e1/step-0002/adapter/adapter_model.safetensors'
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b'broken')
    assert DriveStore(restored, drive).restore() > 0
    assert valid_checkpoint(restored / 'work/e1/step-0002', 'test')
    assert not (restored / 'work/e1/step-0003').exists()
    assert (drive / 'backup_status.json').is_file()


def test_failed_drive_upload_does_not_commit_or_remove_previous_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)
    store = DriveStore(tmp_path / 'output', tmp_path / 'drive')
    first = store.output / 'work/e1/step-0001'
    make_checkpoint(first, step=1)
    store.backup(first)
    second = first.with_name('step-0002')
    make_checkpoint(second)
    original = shutil.copy2

    def fail_state(source, destination):
        if Path(source).name == 'training_state.pt':
            raise OSError('simulated Drive error')
        return original(source, destination)

    monkeypatch.setattr(shutil, 'copy2', fail_state)
    with pytest.raises(OSError):
        store.backup(second)
    assert valid_checkpoint(store.saved / 'work/e1/step-0001')
    assert not valid_checkpoint(store.saved / 'work/e1/step-0002')


def test_reusing_published_snapshot_keeps_commit_marker(monkeypatch, tmp_path):
    store = DriveStore(tmp_path / 'output', tmp_path / 'drive')
    checkpoint = store.output / 'work/e1/step-0002'
    make_checkpoint(checkpoint)
    store.backup(checkpoint)
    monkeypatch.setattr(shutil, 'copy2', lambda *args: pytest.fail('Unchanged snapshot must not be uploaded again'))
    store.backup(checkpoint)
    assert valid_checkpoint(store.saved / 'work/e1/step-0002')


@pytest.mark.parametrize('grad_accum', [1, 2])
def test_resume_training_matches_uninterrupted_optimizer_and_dropout(monkeypatch, tmp_path, grad_accum):
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel, set_seed
    from safetensors.torch import load_file

    # Keep the optimizer, scheduler, scaler, sampler, dropout and PEFT serialization real.
    autocast, scaler = torch.autocast, torch.amp.GradScaler
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: autocast('cpu', enabled=False))
    monkeypatch.setattr(torch.amp, 'GradScaler', lambda *a, **k: scaler('cpu', init_scale=1024))
    for name in ('synchronize', 'reset_peak_memory_stats', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda states: None)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    class Samples(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            ids = torch.tensor([1, self.rows[index]['value'], 3, 4])
            return {'input_ids': ids, 'labels': ids.clone()}

    class Processor:
        def save_pretrained(self, directory):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'processor.json').write_text('{}')

    def load_base():
        # Model weights are the same on a fresh load, independent of prior RNG consumption.
        with torch.random.fork_rng():
            torch.manual_seed(7)
            return GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8,
                vocab_size=16, n_positions=8, resid_pdrop=0.2, attn_pdrop=0.2))

    def setup(model, name, config):
        return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.2,
                             target_modules=['c_attn'], task_type='CAUSAL_LM'))

    config = {'name': 'E1', 'setup': 'E1', 'seed': 42, 'lr': 0.001, 'max_steps': 5,
              'grad_accum': grad_accum, 'batch_size': 2, 'train_fraction': 1., 'train_projector': False}

    def namespace(output, drive):
        store = DriveStore(output, drive)
        ns = dict(copy=copy, gc=gc, json=json, random=random, shutil=shutil, time=time,
                  uuid=uuid, Path=Path, torch=torch, atomic_json=atomic_json,
                  seal_checkpoint=seal_checkpoint, valid_checkpoint=valid_checkpoint,
                  CHECKPOINT_MARKER=CHECKPOINT_MARKER, STORE=store, backup_path=store.backup,
                  read_jsonl=read_jsonl, set_seed=set_seed, load_base_model=load_base,
                  apply_setup=setup, processor=Processor(), is_projector=lambda name: False,
                  DataLoader=torch.utils.data.DataLoader, RoadDataset=Samples,
                  collate_fn=lambda rows: {k: torch.stack([r[k] for r in rows]) for k in rows[0]},
                  to_device=lambda batch: batch,
                  train_samples=[{'image_id': str(i), 'value': 5+i} for i in range(7)],
                  run_directory=lambda name: output / name.lower(),
                  config_fingerprint=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
                  DATASET_VERSION='test', MODEL_ID='tiny-test', MODEL_COMMIT='test', PROMPT='test',
                  RESUME_TRAINING=True, RESTART_INCOMPLETE=True, CHECKPOINT_EVERY_STEPS=2,
                  CHECKPOINT_EVERY_SECONDS=10**9, LOG_BACKUP_SECONDS=10**9,
                  KEEP_STEP_CHECKPOINTS=1)
        exec(compile((ROOT / 'colab_training.py').read_text(encoding='utf-8'), 'colab_training.py', 'exec'), ns)
        return ns

    class Disconnect(Exception):
        pass

    try:
        full = namespace(tmp_path / 'full', tmp_path / 'full-drive')
        expected = full['train_experiment'](config)
        interrupted = namespace(tmp_path / 'interrupted', tmp_path / 'resume-drive')
        backup = interrupted['backup_path']

        def disconnect_after_saved_step(path):
            backup(path)
            if Path(path).name == 'step-0002':
                raise Disconnect()

        interrupted['backup_path'] = disconnect_after_saved_step
        with pytest.raises(Disconnect):
            interrupted['train_experiment'](config)
        resumed = namespace(tmp_path / 'fresh-session', tmp_path / 'resume-drive')
        resumed['STORE'].restore()
        actual = resumed['train_experiment'](config)
        expected_weights = load_file(expected / 'adapter/adapter_model.safetensors')
        actual_weights = load_file(actual / 'adapter/adapter_model.safetensors')
        assert all(torch.equal(value, actual_weights[key]) for key, value in expected_weights.items())
        expected_losses = [r['loss'] for r in read_jsonl(expected / 'loss.jsonl')]
        assert [r['loss'] for r in read_jsonl(actual / 'loss.jsonl')] == expected_losses
        assert len(expected_losses) == 5
        assert valid_checkpoint(actual)
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('torn_last_line', [b'{"image_id":', b'{"raw":"\xe1\xbb'])
def test_inference_resumes_drive_prefix_and_ignores_torn_last_line(tmp_path, monkeypatch, torn_last_line):
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: contextlib.nullcontext())
    output, drive = tmp_path / 'output', tmp_path / 'drive'
    store = DriveStore(output, drive)
    output.mkdir()
    path = output / 'e0_test.jsonl'
    manifest = {'split': 'test', 'run_id': 'test'}
    atomic_json(path.with_suffix('.progress.manifest.json'), manifest)
    path.with_suffix('.progress.jsonl').write_bytes(b'{"image_id":"0","raw":"[]"}\n' + torn_last_line)
    processed = []

    class Processor:
        tokenizer = SimpleNamespace(padding_side='left', pad_token_id=0)

        def apply_chat_template(self, messages, **kwargs):
            processed.extend(messages)
            return {'input_ids': torch.ones((len(messages), 2), dtype=torch.long)}

        def batch_decode(self, tokens, **kwargs):
            return ['[]'] * len(tokens)

    model = SimpleNamespace(eval=lambda: None, gradient_checkpointing_disable=lambda: None,
                            generate=lambda **kw: torch.ones((len(kw['input_ids']), 3), dtype=torch.long))
    rows = [{'image_id': str(i)} for i in range(4)]
    ns = dict(torch=torch, Path=Path, json=json, time=time, atomic_json=atomic_json,
              processor=Processor(), backup_path=store.backup, INFER_BATCH_SIZE=2,
              PREDICTION_BACKUP_SECONDS=60, MAX_NEW_TOKENS=16,
              user_message=lambda image: image, load_image=lambda row: row['image_id'],
              to_device=lambda batch: batch,
              valid_prediction_file=lambda path, rows, manifest: len(read_jsonl(path)) == len(rows))
    exec(compile((ROOT / 'colab_inference.py').read_text(encoding='utf-8'), 'colab_inference.py', 'exec'), ns)
    ns['predict_jsonl'](model, rows, path, manifest)
    assert processed == [['1'], ['2'], ['3']]
    assert [r['image_id'] for r in read_jsonl(path)] == ['0', '1', '2', '3']
    assert (drive / path.name).read_bytes() == path.read_bytes()
