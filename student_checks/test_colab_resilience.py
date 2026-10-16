"""Failure paths that could otherwise interrupt a long Colab run."""
import ast
import contextlib
import errno
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import colab_model
from colab_storage import DriveStore, atomic_json, retry_io
from scripts.labcommon import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_drive_retries_transient_error_but_not_full_disk(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr(time, 'sleep', sleeps.append)

    def transient():
        calls.append(1)
        if len(calls) < 3:
            raise OSError(errno.EIO, 'simulated mounted-Drive error')
        return 'saved'

    assert retry_io(transient) == 'saved'
    assert sleeps == [1, 2]
    calls.clear()

    def full_disk():
        calls.append(1)
        raise OSError(errno.ENOSPC, 'disk full')

    with pytest.raises(OSError):
        retry_io(full_disk)
    assert calls == [1]


def test_model_http_retry_and_fail_fast_on_authentication(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(time, 'sleep', sleeps.append)

    class HTTPError(Exception):
        def __init__(self, status):
            self.response = SimpleNamespace(status_code=status)

    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise HTTPError(429)
        return 'downloaded'

    assert colab_model.retry_download(operation) == 'downloaded'
    assert sleeps == [2, 4]
    calls.clear()

    def unauthorized():
        calls.append(1)
        raise HTTPError(401)

    with pytest.raises(HTTPError):
        colab_model.retry_download(unauthorized)
    assert calls == [1]


def test_backup_retries_directory_creation_before_copy(monkeypatch, tmp_path):
    store = DriveStore(tmp_path / 'output', tmp_path / 'drive')
    source = store.output / 'logs/progress.json'
    atomic_json(source, {'step': 1})
    mkdir = Path.mkdir
    calls = []
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)

    def temporary_failure(path, *args, **kwargs):
        if path == store.saved / 'logs' and not calls:
            calls.append(1)
            raise OSError(errno.EIO, 'temporary Drive mkdir failure')
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'mkdir', temporary_failure)
    store.backup(source)
    assert json.loads((store.saved / 'logs/progress.json').read_text()) == {'step': 1}
    assert calls == [1]


def test_pinned_snapshot_is_reused_without_network(monkeypatch, tmp_path):
    import huggingface_hub
    commit = 'a' * 40
    snapshot = tmp_path / 'snapshots' / commit
    snapshot.mkdir(parents=True)
    names = ('config.json', 'generation_config.json', 'preprocessor_config.json',
             'video_preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json',
             'vocab.json', 'merges.txt', 'chat_template.json', 'model.safetensors')
    for name in names:
        (snapshot / name).write_text('{}')
    atomic_json(snapshot / 'model.safetensors.index.json', {'weight_map': {'weight': 'model.safetensors'}})
    monkeypatch.setattr(huggingface_hub, 'try_to_load_from_cache',
                        lambda model, name, **kwargs: str(snapshot / name))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download',
                        lambda *args, **kwargs: pytest.fail('Pinned cached model must not call the network'))
    path, revision = colab_model.prepare_model_snapshot('test/model', commit, tmp_path)
    assert path == snapshot and revision == commit


@pytest.mark.parametrize('fail_single', [False, True])
def test_inference_oom_retries_same_ids_without_duplicates(monkeypatch, tmp_path, fail_single):
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    store = DriveStore(tmp_path / 'output', tmp_path / 'drive')
    store.output.mkdir()
    path = store.output / 'predictions.jsonl'
    rows = [{'image_id': str(i)} for i in range(5)]
    manifest = {'split': 'test', 'run_id': 'test'}
    generated = []

    class Processor:
        tokenizer = SimpleNamespace(padding_side='left', pad_token_id=0)

        def apply_chat_template(self, messages, **kwargs):
            return {'input_ids': torch.tensor([[int(message[0]), 9] for message in messages])}

        def batch_decode(self, tokens, **kwargs):
            return ['[]'] * len(tokens)

    def generate(**kwargs):
        batch = kwargs['input_ids']
        if len(batch) > 1 or fail_single:
            raise torch.cuda.OutOfMemoryError('simulated CUDA OOM')
        generated.extend(batch[:, 0].tolist())
        return torch.ones((len(batch), 3), dtype=torch.long)

    model = SimpleNamespace(eval=lambda: None, gradient_checkpointing_disable=lambda: None, generate=generate)
    namespace = dict(torch=torch, Path=Path, json=json, time=time, atomic_json=atomic_json,
                     processor=Processor(), backup_path=store.backup, INFER_BATCH_SIZE=4,
                     PREDICTION_BACKUP_SECONDS=60, MAX_NEW_TOKENS=16,
                     user_message=lambda image: image, load_image=lambda row: row['image_id'],
                     to_device=lambda batch: batch,
                     valid_prediction_file=lambda path, rows, manifest: len(read_jsonl(path)) == len(rows))
    exec((ROOT / 'colab_inference.py').read_text(encoding='utf-8'), namespace)
    if fail_single:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            namespace['predict_jsonl'](model, rows, path, manifest)
        assert not path.exists()
        assert not path.with_suffix('.manifest.json').exists()
        assert (store.saved / 'predictions.progress.jsonl').exists()
    else:
        namespace['predict_jsonl'](model, rows, path, manifest)
        assert generated == [0, 1, 2, 3, 4]
        assert [row['image_id'] for row in read_jsonl(path)] == ['0', '1', '2', '3', '4']


def test_gpu_probe_executes_before_long_baseline():
    notebook = json.loads((ROOT / 'lab_day10.ipynb').read_text(encoding='utf-8'))
    nodes = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            nodes.extend(ast.parse(''.join(cell['source'])).body)
    calls = []
    for node in nodes:
        # Do not inspect function bodies: the check concerns actual top-level execution.
        for child in ast.walk(node) if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) else []:
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                calls.append(child.func.id)
    assert calls.count('preflight_training') == 1
    assert calls.count('execute_e0') == 1
    assert calls.index('preflight_training') < calls.index('execute_e0')
