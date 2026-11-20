import argparse
from faceforge.data import manifest
p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--output',default='data/manifest.json'); p.add_argument('--seed',type=int,default=42)
a=p.parse_args(); result=manifest(a.root,a.output,a.seed); print(f"Valid unique: {len(result['rows'])}; rejected: {len(result['rejected'])}")
