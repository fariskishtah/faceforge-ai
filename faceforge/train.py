"""Resumable epoch-level training with fixed latent progress samples."""
import json, random, time, math, hashlib
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from .data import Faces
from .model import Generator,Discriminator,initialize

def step(g,d,og,od,real,z_dim):
    n=real.size(0); device=real.device; criterion=nn.BCEWithLogitsLoss()
    od.zero_grad(set_to_none=True); og.zero_grad(set_to_none=True)
    z=torch.randn(n,z_dim,1,1,device=device); fake=g(z)
    dl=criterion(d(real),torch.ones(n,device=device))+criterion(d(fake.detach()),torch.zeros(n,device=device))
    if not torch.isfinite(dl): raise FloatingPointError('Non-finite discriminator loss')
    dl.backward(); dn=math.sqrt(sum(p.grad.detach().square().sum().item() for p in d.parameters() if p.grad is not None)); od.step()
    od.zero_grad(set_to_none=True)
    for p in d.parameters(): p.requires_grad_(False)
    gl=criterion(d(fake),torch.ones(n,device=device))
    if not torch.isfinite(gl): raise FloatingPointError('Non-finite generator loss')
    gl.backward(); gn=math.sqrt(sum(p.grad.detach().square().sum().item() for p in g.parameters() if p.grad is not None)); og.step()
    for p in d.parameters(): p.requires_grad_(True)
    return {'d_loss':dl.item(),'g_loss':gl.item(),'d_grad_norm':dn,'g_grad_norm':gn}

def train(manifest, out, epochs=10, batch=64, seed=42, z_dim=128, width=64, lr=.0002, device='cpu', resume=None, d_lr=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); random.seed(seed); torch.manual_seed(seed)
    data=Faces(manifest); loader=DataLoader(data,batch_size=batch,shuffle=True,drop_last=True,num_workers=0)
    if not len(loader): raise ValueError('Not enough training images for one batch')
    g=Generator(z_dim,width).to(device); d=Discriminator(width).to(device); g.apply(initialize); d.apply(initialize)
    d_lr=lr if d_lr is None else d_lr
    og=torch.optim.Adam(g.parameters(),lr=lr,betas=(.5,.999)); od=torch.optim.Adam(d.parameters(),lr=d_lr,betas=(.5,.999))
    fixed=torch.randn(16,z_dim,1,1,generator=torch.Generator().manual_seed(seed+1),device='cpu').to(device)
    start=0; global_step=0
    config={'manifest':str(Path(manifest).resolve()),'epochs':epochs,'batch':batch,'seed':seed,'z_dim':z_dim,'width':width,'lr':lr,'d_lr':d_lr,'device':device,'resolution':64,'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest()}
    if resume:
        c=torch.load(resume,map_location=device,weights_only=False)
        for k in ('manifest','batch','seed','z_dim','width','lr','d_lr','resolution','manifest_sha256'):
            if c['config'].get(k)!=config[k]: raise ValueError(f'Resume configuration mismatch: {k}')
        g.load_state_dict(c['generator']); d.load_state_dict(c['discriminator']); og.load_state_dict(c['og']); od.load_state_dict(c['od']); start=c['epoch']; global_step=c['step']; fixed=c['fixed'].to(device); torch.set_rng_state(c['rng'].cpu()); random.setstate(c['python_rng']);
        if device.startswith('cuda') and c.get('cuda_rng') is not None: torch.cuda.set_rng_state_all([state.cpu() for state in c['cuda_rng']])
    elif (out/'metrics.jsonl').exists(): raise FileExistsError('Output exists; use a new output or --resume')
    t=time.time()
    for epoch in range(start,epochs):
        for real in loader:
            metrics=step(g,d,og,od,real.to(device),z_dim); global_step+=1
            with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps({'epoch':epoch+1,'step':global_step,**metrics})+'\n')
        g.eval()
        with torch.no_grad(): save_image(g(fixed).cpu(),out/f'fixed_epoch_{epoch+1:04}.png',nrow=4,normalize=True,value_range=(-1,1))
        g.train()
        c={'generator':g.state_dict(),'discriminator':d.state_dict(),'og':og.state_dict(),'od':od.state_dict(),'epoch':epoch+1,'step':global_step,'config':config,'fixed':fixed.cpu(),'rng':torch.get_rng_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else None,'duration_this_invocation_s':time.time()-t}
        tmp=out/'checkpoint.tmp'; torch.save(c,tmp); tmp.replace(out/'checkpoint.pt')
        print(f'epoch={epoch+1} step={global_step} d={metrics["d_loss"]:.4f} g={metrics["g_loss"]:.4f}',flush=True)
