"""Grad-CAM for the project's Swin-Tiny classifier."""
import numpy as np
import torch
from PIL import Image


def swin_reshape_transform(tensor):
    # timm Swin blocks expose [B, H, W, C] features at block outputs in many
    # versions. Handle both [B,H,W,C] and [B,N,C] forms defensively.
    if tensor.ndim == 4:
        if tensor.shape[1] == tensor.shape[2]:
            return tensor.permute(0, 3, 1, 2)
        return tensor.permute(0, 3, 1, 2)
    if tensor.ndim == 3:
        b, n, c = tensor.shape
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError(f"Cannot reshape Swin activation with {n} tokens")
        return tensor.reshape(b, side, side, c).permute(0, 3, 1, 2)
    raise ValueError(f"Unsupported activation shape: {tuple(tensor.shape)}")


def find_target_layer(model):
    """Pick the final Swin block norm layer used for Grad-CAM."""
    try:
        return model.layers[-1].blocks[-1].norm1
    except Exception as exc:
        raise RuntimeError("Unable to locate the final Swin target layer") from exc


def generate_gradcam(model, input_tensor, class_index, target_layer=None):
    """Return a normalized HxW CAM without requiring an external package."""
    model.eval()
    target_layer = target_layer or find_target_layer(model)
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        x = input_tensor.clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        score = logits[:, class_index].sum()
        score.backward()
        a = swin_reshape_transform(activations[-1]).detach()
        g = swin_reshape_transform(gradients[-1]).detach()
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = (weights * a).sum(dim=1).relu()[0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.cpu().numpy(), logits.detach()
    finally:
        h1.remove()
        h2.remove()


def overlay_cam(image, cam, mask=None):
    import cv2
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]

    # Smoothly upscale CAM using bicubic interpolation and Gaussian blur
    heat = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (21, 21), 0)

    # Re-normalize between 0 and 1
    heat_min = float(heat.min())
    heat_max = float(heat.max())
    if heat_max > heat_min:
        heat = (heat - heat_min) / (heat_max - heat_min + 1e-8)
    else:
        heat = np.zeros_like(heat)

    # Align Heatmap with Segmentation Result
    if mask is not None:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        dist = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
        dist_norm = dist / (dist.max() + 1e-8)
        seg_weight = cv2.GaussianBlur(dist_norm, (35, 35), 0)

        # Merge CAM attention with segmented anatomy so hotspot aligns with segmentation
        heat = heat * 0.30 + seg_weight * 0.70
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat = heat * (dilated > 0).astype(np.float32)

    heat_u8 = np.uint8(255 * heat)
    # Apply standard medical JET colormap (cold=blue -> warm=yellow -> peak=hot red)
    heatmap_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Modulate alpha based on attention intensity
    alpha = np.clip((heat - 0.20) / 0.80, 0.0, 1.0)
    alpha = np.power(alpha, 1.1) * 0.75

    # Suppress coloring in non-tissue black margins outside the ultrasound probe sector
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tissue_weight = np.clip(gray.astype(np.float32) / 28.0, 0.15, 1.0)
    alpha = alpha * tissue_weight
    alpha_3d = alpha[:, :, np.newaxis]

    blended = (1.0 - alpha_3d) * rgb.astype(np.float32) + alpha_3d * heatmap_rgb.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
