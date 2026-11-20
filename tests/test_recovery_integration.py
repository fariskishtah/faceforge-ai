from pathlib import Path
import pytest
import hashlib
from PIL import Image
from faceforge.inference import load, latent, render, interpolate

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/kaggle_recovery/run/generator.pt'
EXPECTED='45c25da47ad0e5a30277eb2107933a58f883e54179b211fe60dd78ac02f27f67'

@pytest.mark.skipif(not MODEL.is_file(), reason="private recovery checkpoint unavailable")
def test_real_checkpoint_inference_and_seed():
    assert MODEL.is_file(), 'Recovered generator checkpoint is required for integration verification'
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==EXPECTED
    g,cfg=load(MODEL)
    assert cfg['resolution']==64 and cfg['z_dim']==128
    a=render(g,latent(123,2,128));b=render(g,latent(123,2,128));other=render(g,latent(124,2,128))
    assert a[0].size==(64,64) and a[0].mode=='RGB'
    assert a[0].tobytes()==b[0].tobytes()
    assert a[0].tobytes()!=other[0].tobytes()
    assert len(render(g,interpolate(latent(1,1,128),latent(2,1,128),5)))==5

@pytest.mark.skipif(not (ROOT/"artifacts/kaggle_recovery/generated/seed_123_00.png").is_file(), reason="private sample unavailable")
def test_exported_png_metadata():
    p=ROOT/'artifacts/kaggle_recovery/generated/seed_123_00.png'
    with Image.open(p) as im:
        assert im.size==(64,64)
        assert 'AI-generated' in im.info['Description']
        assert EXPECTED in im.info['FaceForge']

@pytest.mark.skipif(not (ROOT/"artifacts/kaggle_extend/run/generator.pt").is_file() or not (ROOT/"artifacts/kaggle_eval/evaluation.json").is_file(), reason="private research artifacts unavailable")
def test_extended_checkpoint_and_comparison():
    import json
    newer=ROOT/'artifacts/kaggle_extend/run/generator.pt'
    assert hashlib.sha256(newer.read_bytes()).hexdigest()=='1998b102398d0da2f82f5438d560966acdcd285f93b0d611acbcc170be2249d6'
    g_old,old_cfg=load(MODEL);g_new,new_cfg=load(newer)
    assert old_cfg['z_dim']==new_cfg['z_dim']==128
    z=latent(20261120,8,128)
    old=render(g_old,z);new=render(g_new,z)
    assert len(old)==len(new)==8 and old[0].tobytes()!=new[0].tobytes()
    scores=json.loads((ROOT/'artifacts/kaggle_eval/evaluation.json').read_text())
    assert scores['recovery']['n_real']==scores['extended']['n_real']==1000
    assert scores['recovery']['n_fake']==scores['extended']['n_fake']==1000
    assert scores['extended']['fid']<scores['recovery']['fid']
    assert scores['extended']['kid']<scores['recovery']['kid']
