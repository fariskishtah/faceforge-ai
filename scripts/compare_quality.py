"""Reproducible diagnostics, not substitutes for FID/KID or human review."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torchvision.utils import save_image
from faceforge.inference import load,latent

def measure(path,seed,count,out):
    g,cfg=load(path); z=latent(seed,count,cfg['z_dim']); start=time.perf_counter()
    with torch.inference_mode(): x=g(z)
    elapsed=time.perf_counter()-start
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    save_image(x,output/'fixed_grid.png',nrow=8,normalize=True,value_range=(-1,1))
    result={'checkpoint':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'seed':seed,'count':count,'resolution':cfg['resolution'],'parameters':sum(p.numel() for p in g.parameters()),'latency_s':elapsed,'range':[x.min().item(),x.max().item()],'mean':x.mean().item(),'std':x.std().item(),'pixel_pairwise_l1':torch.pdist(x.flatten(1),p=1).mean().item()/x[0].numel(),'horizontal_adjacent_l1':(x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean().item(),'vertical_adjacent_l1':(x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean().item()}
    (output/'diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20261120);p.add_argument('--count',type=int,default=32);a=p.parse_args();measure(a.checkpoint,a.seed,a.count,a.out)
