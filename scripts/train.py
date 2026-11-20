import argparse
from faceforge.train import train
p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch',type=int,default=64); p.add_argument('--seed',type=int,default=42); p.add_argument('--z-dim',type=int,default=128); p.add_argument('--width',type=int,default=64); p.add_argument('--lr',type=float,default=.0002); p.add_argument('--d-lr',type=float); p.add_argument('--device',default='cpu'); p.add_argument('--resume')
a=p.parse_args(); train(**vars(a))
