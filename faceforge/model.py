"""A compact 64x64 DCGAN. Discriminator returns logits, not probabilities."""
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim=128, width=64):
        super().__init__()
        self.z_dim, self.width = z_dim, width
        layers = []
        channels = [z_dim, width*8, width*4, width*2, width, 3]
        for i, (a,b) in enumerate(zip(channels, channels[1:])):
            layers.append(nn.ConvTranspose2d(a,b,4,1 if i == 0 else 2,0 if i == 0 else 1,bias=False))
            layers.extend([nn.BatchNorm2d(b),nn.ReLU(True)] if i < 4 else [nn.Tanh()])
        self.net = nn.Sequential(*layers)
    def forward(self,z):
        if z.ndim == 2: z = z[:,:,None,None]
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        layers=[]
        channels=[3,width,width*2,width*4,width*8,1]
        for i,(a,b) in enumerate(zip(channels,channels[1:])):
            layers.append(nn.Conv2d(a,b,4,1 if i == 4 else 2,0 if i == 4 else 1,bias=False))
            if 0 < i < 4: layers.append(nn.BatchNorm2d(b))
            if i < 4: layers.append(nn.LeakyReLU(0.2,True))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).flatten()

def initialize(m):
    if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)): nn.init.normal_(m.weight,0,0.02)
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.normal_(m.weight,1,0.02); nn.init.zeros_(m.bias)
