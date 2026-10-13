# Segmentation + Grad-CAM Upgrade

This upgrade adds two research-facing components to the existing Swin-Tiny Flask application:

1. **Anatomical segmentation / ROI** using an optional U-Net checkpoint.
2. **Swin Transformer Grad-CAM** to visualize image regions that contributed to the predicted class.

## Important: current state

The uploaded repository contains the Swin classifier checkpoint as a **Git-LFS pointer**. Run `git lfs pull` in a normal Git clone (or replace the pointer with the real `.pth` file) before running the classifier.

The repository does **not** contain pixel-level segmentation labels or a trained segmentation checkpoint. Therefore the web app falls back to a clearly labelled `demo_classical_roi` visualization until a real U-Net checkpoint is trained.

Do **not** report the demo ROI as segmentation accuracy or use it for clinical claims.

## Train a real U-Net

Prepare paired data:

```text
data/
  images/
    case001.jpg
    case002.jpg
  masks/
    case001.png
    case002.png
```

Each mask must be binary: background = 0, fetal head/brain ROI = non-zero.

Then run:

```bash
python training/train_segmentation.py --images data/images --masks data/masks --epochs 30 --out webapp/models/segmentation_unet.pth
```

For research, replace the random split with a **patient-level split** if multiple images belong to the same patient/examination.

The app automatically loads `webapp/models/segmentation_unet.pth` when present. You can override the path with:

```bash
SEGMENTATION_MODEL_PATH=/path/to/model.pth
```

## What the app will show

```text
Ultrasound
   ├── Existing Swin-Tiny classifier → class + confidence
   ├── U-Net → anatomical mask / ROI
   └── Grad-CAM → classifier explanation heatmap
```

## Research gap this enables

The important experiment is not simply “we added a heatmap”. Compare:

- **Baseline:** raw image → Swin-Tiny
- **Segmentation-guided:** segmented ROI → Swin-Tiny
- **Segmentation-guided + Grad-CAM:** ROI classifier + explanation

Report accuracy, macro-F1, balanced accuracy, per-class recall, confusion matrix and AUROC where appropriate. For the segmentation model report Dice and IoU. For explainability, consider quantitative localization/faithfulness metrics rather than only showing heatmaps.

## Medical/research note

Grad-CAM indicates regions associated with the model's prediction; it does not prove that a highlighted region is the biological cause of an abnormality. The system is for research/decision-support evaluation, not autonomous diagnosis.
