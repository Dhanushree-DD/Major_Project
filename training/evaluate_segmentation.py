"""Evaluate a trained fetal-brain/head segmentation model.

Dataset layout:
  data/images/*.jpg|png
  data/masks/*.png

Reports Dice and IoU. For repeated frames from the same patient/exam, use
--groups to provide a CSV with columns `image,group` so evaluation is grouped.
"""
import argparse
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))
from ai.segmentation import UNet

EXTS={".jpg",".jpeg",".png",".bmp"}
TFM=transforms.Compose([transforms.Grayscale(1),transforms.Resize((256,256)),transforms.ToTensor()])

class DS(Dataset):
    def __init__(self, image_dir, mask_dir, names=None):
        self.image_dir=Path(image_dir); self.mask_dir=Path(mask_dir)
        allp=sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in EXTS])
        self.paths=[p for p in allp if names is None or p.name in names]
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        p=self.paths[i]; m=self.mask_dir/(p.stem+'.png')
        if not m.exists(): raise FileNotFoundError(f'Mask missing: {m}')
        return TFM(Image.open(p).convert('RGB')), (TFM(Image.open(m).convert('L'))>0.5).float(), p.name

def metrics(pred, target, eps=1e-7):
    p=pred.bool(); y=target.bool()
    inter=(p & y).sum().item(); union=(p | y).sum().item()
    dice=(2*inter+eps)/(p.sum().item()+y.sum().item()+eps)
    iou=(inter+eps)/(union+eps)
    return dice,iou

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--images',required=True); ap.add_argument('--masks',required=True)
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--groups',default=None)
    ap.add_argument('--threshold',type=float,default=.5); args=ap.parse_args()
    names=None
    if args.groups:
        with open(args.groups,newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        names={r['image'] for r in rows}
    ds=DS(args.images,args.masks,names)
    if not ds: raise RuntimeError('No evaluation images found')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=UNet().to(device); model.load_state_dict(ckpt.get('model_state_dict',ckpt)); model.eval()
    dl=DataLoader(ds,batch_size=1,shuffle=False)
    scores=[]
    with torch.no_grad():
        for x,y,name in dl:
            prob=torch.sigmoid(model(x.to(device))).cpu()
            d,i=metrics(prob>=args.threshold,y)
            scores.append((d,i,name[0]))
    print(f'images={len(scores)}')
    print(f'Dice={np.mean([s[0] for s in scores]):.4f}')
    print(f'IoU={np.mean([s[1] for s in scores]):.4f}')
    for d,i,n in scores:
        print(f'{n}\tDice={d:.4f}\tIoU={i:.4f}')
if __name__=='__main__': main()
