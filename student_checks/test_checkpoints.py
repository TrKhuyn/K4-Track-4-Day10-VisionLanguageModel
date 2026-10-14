import json
import pytest
from lab_runtime import finalize_grpo_checkpoint, write_json


def checkpoint(directory, step=60, projector=True):
    path=directory/f'checkpoint-{step}'
    path.mkdir(parents=True)
    write_json(path/'complete.json',{'step':step})
    write_json(path/'run.json',{'completed_steps':step,'config':{'train_projector':projector}})
    write_json(path/'adapter_config.json',{'peft_type':'LORA'})
    (path/'adapter_model.safetensors').write_bytes(b'unit-test-placeholder')
    if projector: (path/'projector.pt').write_bytes(b'unit-test-projector')
    return path


def test_final_grpo_step_can_be_exported_without_training_again(tmp_path):
    path=checkpoint(tmp_path)
    finalize_grpo_checkpoint(tmp_path,path,60)
    assert json.loads((tmp_path/'finished.json').read_text())=={'checkpoint':'final'}
    assert (tmp_path/'final/adapter/adapter_model.safetensors').read_bytes()==(path/'adapter_model.safetensors').read_bytes()
    assert (tmp_path/'final/projector.pt').read_bytes()==(path/'projector.pt').read_bytes()


def test_incomplete_projector_never_gets_a_finished_marker(tmp_path):
    path=checkpoint(tmp_path)
    (path/'projector.pt').unlink()
    with pytest.raises(AssertionError): finalize_grpo_checkpoint(tmp_path,path,60)
    assert not (tmp_path/'finished.json').exists()


def test_intermediate_checkpoint_cannot_be_labeled_as_final(tmp_path):
    path=checkpoint(tmp_path,step=20)
    with pytest.raises(AssertionError): finalize_grpo_checkpoint(tmp_path,path,60)
    assert not (tmp_path/'finished.json').exists()


def test_frozen_projector_is_optional_only_for_e1_source(tmp_path):
    path=checkpoint(tmp_path,projector=False)
    finalize_grpo_checkpoint(tmp_path,path,60)
    assert (tmp_path/'finished.json').exists()
    assert not (tmp_path/'final/projector.pt').exists()
