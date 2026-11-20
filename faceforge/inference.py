"""Generator-only inference. Load trusted local checkpoint files only."""
import torch
from torchvision.transforms.functional import to_pil_image
from .model import Generator

def load(path,device='cpu'):
    c=torch.load(path,map_location=device,weights_only=False)
    cfg=c['config']; g=Generator(cfg['z_dim'],cfg['width']).to(device); g.load_state_dict(c['generator']); g.eval(); return g,cfg

def latent(seed,n,z_dim):
    if not 1<=n<=32: raise ValueError('Batch size must be between 1 and 32')
    return torch.randn(n,z_dim,1,1,generator=torch.Generator().manual_seed(int(seed)))

def render(g,z):
    with torch.inference_mode(): x=g(z.to(next(g.parameters()).device)).cpu()
    return [to_pil_image(im.add(1).div(2).clamp(0,1)) for im in x]

def interpolate(a,b,steps):
    if not 2<=steps<=32: raise ValueError('Steps must be between 2 and 32')
    return torch.cat([(1-t)*a+t*b for t in torch.linspace(0,1,steps)],dim=0)
