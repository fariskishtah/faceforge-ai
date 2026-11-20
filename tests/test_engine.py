import torch
from PIL import Image
from faceforge.model import Generator,Discriminator
from faceforge.train import step,train
from faceforge.data import manifest,Faces
from faceforge.inference import load,latent,render,interpolate

def test_models_and_updates():
    torch.manual_seed(3); g=Generator(8,4); d=Discriminator(4); z=torch.randn(2,8,1,1); x=g(z)
    assert x.shape==(2,3,64,64) and x.abs().max()<=1
    assert d(x).shape==(2,)
    a=next(g.parameters()).detach().clone(); b=next(d.parameters()).detach().clone()
    og=torch.optim.Adam(g.parameters()); od=torch.optim.Adam(d.parameters()); m=step(g,d,og,od,torch.randn(2,3,64,64),8)
    assert all(torch.isfinite(torch.tensor(v)) for v in m.values())
    assert not torch.equal(a,next(g.parameters())) and not torch.equal(b,next(d.parameters()))
    assert all(p.grad is None for p in d.parameters())

def test_data_checkpoint_resume_inference(tmp_path):
    root=tmp_path/'images'; root.mkdir()
    for i in range(4): Image.new('RGB',(90,100),(i*50,10,20)).save(root/f'{i}.png')
    m=tmp_path/'manifest.json'; result=manifest(root,m); assert len(result['rows'])==4
    assert Faces(m)[0].shape==(3,64,64)
    out=tmp_path/'run'; train(str(m),str(out),epochs=1,batch=2,z_dim=8,width=4)
    train(str(m),str(out),epochs=2,batch=2,z_dim=8,width=4,resume=str(out/'checkpoint.pt'))
    g,c=load(out/'checkpoint.pt'); z=latent(7,2,8); assert torch.equal(z,latent(7,2,8))
    assert len(render(g,z))==2 and interpolate(z[:1],z[1:],3).shape==(3,8,1,1)
    assert (out/'fixed_epoch_0001.png').exists() and (out/'fixed_epoch_0002.png').exists()

def test_invalid_batches_and_interpolation():
    import pytest
    with pytest.raises(ValueError): latent(1,0,8)
    with pytest.raises(ValueError): latent(1,33,8)
    with pytest.raises(ValueError): interpolate(torch.zeros(1,8,1,1),torch.ones(1,8,1,1),1)

def test_nonfinite_loss_fails():
    import pytest
    g=Generator(8,4);d=Discriminator(4)
    with pytest.raises(FloatingPointError):
        step(g,d,torch.optim.Adam(g.parameters()),torch.optim.Adam(d.parameters()),torch.full((2,3,64,64),float('nan')),8)
