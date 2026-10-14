import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import run_lab


def test_runner_keeps_one_lab_for_notebook_stages(monkeypatch,tmp_path):
    created=[]
    class FakeLab:
        def __init__(self,*args,**kwargs):
            created.append(kwargs)
            self.configs={name:{'batch_size':kwargs['batch_size']} for name in ('E1','E2','E3')}
    monkeypatch.setattr(run_lab,'Lab',FakeLab)
    namespace={'ROOT':tmp_path,'DRIVE_ROOT':tmp_path/'drive','TRAIN_BATCH_SIZE':1,'GRAD_ACCUM':8}
    run_lab.main(namespace,['--stage','setup'])
    run_lab.main(namespace,['--stage','setup'])
    assert len(created)==1
    assert created[0]['batch_size']==1 and created[0]['grad_accum']==8


def test_runner_e0_preflights_before_long_baseline(monkeypatch,tmp_path):
    calls=[]
    class FakeLab:
        def __init__(self,*args,**kwargs): pass
        def preflight(self): calls.append('preflight')
        def baseline(self): calls.append('baseline')
    monkeypatch.setattr(run_lab,'Lab',FakeLab)
    run_lab.main({'ROOT':tmp_path},['--stage','e0'])
    assert calls==['preflight','baseline']


def test_saved_selection_uses_validation_and_recovers_drive_evidence(tmp_path):
    lab=SimpleNamespace(output=tmp_path,meta={'dataset_version':'d10-test'})
    for name,val,test in (('E1',.9,.1),('E2',.6,.95),('E3',.5,.8)):
        directory=tmp_path/'runs'/name.lower()
        directory.mkdir(parents=True)
        (directory/'finished.json').write_text(json.dumps({'checkpoint':'final'}))
        artifact=directory/'final'
        (artifact/'adapter').mkdir(parents=True)
        (artifact/'run.json').write_text(json.dumps({'dataset_version':'d10-test',
            'completed_steps':1,'config':{'max_steps':1,'train_projector':False}}))
        (artifact/'adapter/adapter_config.json').write_text('{}')
        (artifact/'adapter/adapter_model.safetensors').write_bytes(b'unit-test-only')
        for split,f1 in (('val',val),('test',test)):
            path=(tmp_path/f'{name.lower()}_val_score.json' if split=='val'
                  else tmp_path/'evidence'/f'{name.lower()}.json')
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({'dataset_version':'d10-test','split':split,
                                       'missing_predictions':0,'micro':{'f1':f1}}))
    run_lab.restore_selection(lab)
    assert lab.best=='E1'
    assert set(lab.runs)=={'E1','E2','E3'}


def test_ipython_run_i_keeps_state_and_marks_success(monkeypatch,tmp_path):
    shell=pytest.importorskip('IPython.core.interactiveshell').InteractiveShell.instance(
        ipython_dir=str(tmp_path/'ipython'))
    import lab_runtime
    created=[]
    class FakeLab:
        def __init__(self,*args,**kwargs):
            created.append(kwargs)
            self.configs={name:{} for name in ('E1','E2','E3')}
    monkeypatch.setattr(lab_runtime,'Lab',FakeLab)
    script=Path(run_lab.__file__).resolve()
    shell.user_ns.update(ROOT=tmp_path,DRIVE_ROOT=tmp_path/'drive',TRAIN_BATCH_SIZE=2,GRAD_ACCUM=4)
    shell.user_ns.pop('_DAY10_LAB',None)
    for _ in range(2):
        shell.user_ns['_DAY10_STAGE_OK']=False
        shell.run_line_magic('run',f'-i "{script}" --stage setup')
        assert shell.user_ns['_DAY10_STAGE_OK'] is True
    assert len(created)==1
    assert '_DAY10_LAB' in shell.user_ns
    shell.user_ns.pop('_DAY10_LAB',None)


def test_ipython_stage_failure_cannot_be_marked_success(monkeypatch,tmp_path):
    shell=pytest.importorskip('IPython.core.interactiveshell').InteractiveShell.instance(
        ipython_dir=str(tmp_path/'ipython'))
    import lab_runtime
    class FakeLab:
        def __init__(self,*args,**kwargs): raise ValueError('simulated initialization failure')
    monkeypatch.setattr(lab_runtime,'Lab',FakeLab)
    shell.user_ns.update(ROOT=tmp_path,DRIVE_ROOT=tmp_path/'drive')
    shell.user_ns.pop('_DAY10_LAB',None)
    shell.user_ns['_DAY10_STAGE_OK']=False
    shell.run_line_magic('run',f'-i "{Path(run_lab.__file__).resolve()}" --stage data')
    assert shell.user_ns['_DAY10_STAGE_OK'] is False
