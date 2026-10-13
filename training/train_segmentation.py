"""Train the optional U-Net anatomical segmentation model.

Dataset layout:
  data/images/*.jpg|png
  data/masks/*.png

Masks should be binary, with foreground > 0. Use patient-level splitting when
multiple frames from the same examination/patient exist.
"""
import argparse, os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))
from ai.segmentation import UNet


class MaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.images = sorted([p for p in Path(image_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        self.mask_dir = Path(mask_dir)
        self.to_x = transforms.Compose([transforms.Grayscale(1), transforms.Resize((256,256)), transforms.ToTensor()])
        self.to_y = transforms.Compose([transforms.Grayscale(1), transforms.Resize((256,256), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        p = self.images[i]
        mask = self.mask_dir / (p.stem + ".png")
        return self.to_x(Image.open(p).convert("RGB")), (self.to_y(Image.open(mask).convert("L")) > 0.5).float()


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1,2,3))
    denom = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return (1 - ((2*inter + eps)/(denom + eps))).mean()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--masks", required=True)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="webapp/models/segmentation_unet.pth")
    args=ap.parse_args()
    ds=MaskDataset(args.images,args.masks)
    if len(ds)==0: raise RuntimeError("No segmentation images found")
    n_val=max(1,int(0.2*len(ds))); n_train=len(ds)-n_val
    train,val=random_split(ds,[n_train,n_val],generator=torch.Generator().manual_seed(42))
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=UNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
    bce=torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train(); total=0
        for x,y in DataLoader(train,args.batch_size,shuffle=True):
            x,y=x.to(device),y.to(device); opt.zero_grad(); z=model(x)
            loss=bce(z,y)+dice_loss(z,y); loss.backward(); opt.step(); total+=loss.item()
        print(f"epoch {epoch+1}/{args.epochs} loss={total/max(1,len(train)//args.batch_size):.4f}")
    os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
    torch.save({"model_state_dict":model.state_dict(),"image_size":256},args.out)
    print("saved",args.out)
if __name__=="__main__": main()
