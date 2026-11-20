"""Image validation and deterministic preprocessing."""
from pathlib import Path
import hashlib, json
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

EXT={'.jpg','.jpeg','.png','.webp'}
def manifest(root, output, seed=42):
    import random
    root=Path(root).resolve(); rows=[]; seen=set(); rejected=[]
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() not in EXT or not p.is_file(): continue
        try:
            with Image.open(p) as im: im.verify()
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            if h in seen: rejected.append({'path':str(p.relative_to(root)),'reason':'exact duplicate'}); continue
            seen.add(h)
            with Image.open(p) as im: size=im.size
            rows.append({'path':str(p.relative_to(root)),'sha256':h,'size':size})
        except Exception as e: rejected.append({'path':str(p.relative_to(root)),'reason':str(e)})
    random.Random(seed).shuffle(rows)
    # This split is image-level, not identity- or near-duplicate-disjoint.
    for i,r in enumerate(rows): r['split']='test' if i < len(rows)//10 else 'train'
    out={'root':str(root),'seed':seed,'rows':rows,'rejected':rejected,'warning':'Image-level split only; no identity or perceptual deduplication.'}
    Path(output).write_text(json.dumps(out,indent=2)); return out

class Faces(Dataset):
    def __init__(self,path,split='train'):
        data=json.loads(Path(path).read_text()); self.root=Path(data['root']); self.rows=[r for r in data['rows'] if r['split']==split]
        self.transform=transforms.Compose([transforms.Lambda(lambda im: ImageOps.exif_transpose(im).convert('RGB')),transforms.CenterCrop( min(178, 178) ),transforms.Resize((64,64),antialias=True),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.root/self.rows[i]['path']) as im: return self.transform(im)
