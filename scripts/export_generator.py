"""Export inference weights from a trusted checkpoint, with SHA-256 digest."""
import argparse,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('output'); a=p.parse_args()
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
torch.save({'generator':c['generator'],'config':{'z_dim':c['config']['z_dim'],'width':c['config']['width'],'resolution':64}},a.output)
print('sha256',hashlib.sha256(Path(a.output).read_bytes()).hexdigest())
