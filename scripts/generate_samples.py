"""Generate reproducible PNGs from a trusted FaceForge checkpoint."""
import argparse,hashlib,time
from pathlib import Path
from PIL.PngImagePlugin import PngInfo
from faceforge.inference import load,latent,render
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--count',type=int,default=4);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
g,c=load(a.checkpoint);start=time.perf_counter();images=render(g,latent(a.seed,a.count,c['z_dim']));digest=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
for i,image in enumerate(images):
    info=PngInfo();info.add_text('Description','AI-generated synthetic face imagery; may resemble real individuals.');info.add_text('FaceForge','seed=%s; batch_index=%s; checkpoint_sha256=%s'%(a.seed,i,digest));image.save(out/f'seed_{a.seed}_{i:02}.png',pnginfo=info)
print(f'{len(images)} images in {time.perf_counter()-start:.3f}s; checkpoint SHA-256 {digest}')
