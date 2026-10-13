"""Fetal-head/brain ROI segmentation helpers.

Two modes are supported:
1. A trained U-Net checkpoint, when SEGMENTATION_MODEL_PATH points to one.
2. A clearly labelled classical CV preview fallback, useful for demonstrating
   the web integration before a pixel-labelled segmentation model is trained.

The fallback is NOT a clinical segmentation model and must not be used for
research metrics.
"""
import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

try:
    import cv2
except ImportError:
    cv2 = None

import torch
import torch.nn as nn
from torchvision import transforms


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Small dependency-light U-Net for 2D fetal ultrasound ROI masks."""
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        self.e1 = DoubleConv(in_channels, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.b = DoubleConv(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.d4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    @staticmethod
    def _cat(x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        d4 = self.d4(self._cat(self.u4(b), e4))
        d3 = self.d3(self._cat(self.u3(d4), e3))
        d2 = self.d2(self._cat(self.u2(d3), e2))
        d1 = self.d1(self._cat(self.u1(d2), e1))
        return self.out(d1)


SEG_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def load_segmentation_model(path, device="cpu"):
    if not path or not os.path.isfile(path):
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model = UNet()
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _fallback_mask(image):
    """Detect fetal cranial calvarium contour in obstetric ultrasound scans."""
    gray = np.array(image.convert("L"))
    if cv2 is None:
        h, w = gray.shape
        yy, xx = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        return (((xx - cx) / (0.35 * w)) ** 2 + ((yy - cy) / (0.35 * h)) ** 2 <= 1)

    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    margin_y, margin_x = int(h * 0.04), int(w * 0.04)
    inner = blur[margin_y:h - margin_y, margin_x:w - margin_x]

    # Threshold bright cranial bone structure
    _, thresh = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    padded = np.zeros_like(gray)
    padded[margin_y:h - margin_y, margin_x:w - margin_x] = thresh

    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = np.array([w / 2.0, h / 2.0])
    best_hull = None
    best_score = -1

    for c in contours:
        area = cv2.contourArea(c)
        if area < (h * w * 0.04) or area > (h * w * 0.85):
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        M = cv2.moments(hull)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist = np.linalg.norm(np.array([cx, cy]) - center) / (w * 0.5)

        ratio = 0.6
        if len(hull) >= 5:
            try:
                (ex, ey), (d1, d2), _ = cv2.fitEllipse(hull)
                if max(d1, d2) > 0:
                    ratio = min(d1, d2) / max(d1, d2)
            except Exception:
                pass

        score = hull_area * (0.4 + 0.6 * solidity) * (0.3 + 0.7 * ratio) / (1.0 + 0.8 * dist)
        if score > best_score:
            best_score = score
            best_hull = hull

    mask = np.zeros_like(gray, dtype=np.uint8)
    if best_hull is not None and len(best_hull) >= 5:
        try:
            ellipse = cv2.fitEllipse(best_hull)
            (ex, ey), (d1, d2), _ = ellipse
            if 0.1 * w < d1 < 0.9 * w and 0.1 * h < d2 < 0.9 * h:
                cv2.ellipse(mask, ellipse, 255, thickness=-1)
            else:
                cv2.drawContours(mask, [best_hull], -1, 255, thickness=-1)
        except Exception:
            cv2.drawContours(mask, [best_hull], -1, 255, thickness=-1)
    elif contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        cv2.drawContours(mask, [hull], -1, 255, thickness=-1)
    else:
        yy, xx = np.ogrid[:h, :w]
        mask = ((((xx - w / 2) / (0.35 * w)) ** 2 + ((yy - h / 2) / (0.35 * h)) ** 2 <= 1) * 255).astype(np.uint8)

    return mask > 0


def predict_mask(image, model=None, device="cpu"):
    w, h = image.size
    if model is None:
        mask = _fallback_mask(image)
        return mask.astype(np.uint8), "demo_classical_roi"
    x = SEG_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = (prob >= 0.5).astype(np.uint8)
    mask_img = Image.fromarray(mask * 255).resize((w, h), Image.Resampling.NEAREST)
    return (np.array(mask_img) > 127).astype(np.uint8), "trained_unet"


def make_segmentation_overlay(image, mask):
    """Render professional clinical cyan overlay with anti-aliased cranial boundary."""
    rgb = np.array(image.convert("RGB"))
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    # Medical Cyan tone: (14, 165, 233) with soft 25% opacity
    overlay = rgb.copy().astype(np.float32)
    cyan_fill = np.array([14, 165, 233], dtype=np.float32)
    mask_3d = (mask_u8 > 0)[:, :, np.newaxis]
    overlay = np.where(mask_3d, overlay * 0.75 + cyan_fill * 0.25, overlay)
    res = np.clip(overlay, 0, 255).astype(np.uint8)

    # Smooth anti-aliased 2px medical border: vibrant electric cyan (56, 189, 248)
    if cv2 is not None:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(res, contours, -1, (56, 189, 248), 2, lineType=cv2.LINE_AA)

    return Image.fromarray(res)


def crop_to_mask(image, mask, margin=12):
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return image
    x1, x2 = max(0, int(xs.min()) - margin), min(image.width, int(xs.max()) + margin + 1)
    y1, y2 = max(0, int(ys.min()) - margin), min(image.height, int(ys.max()) + margin + 1)
    return image.crop((x1, y1, x2, y2))
