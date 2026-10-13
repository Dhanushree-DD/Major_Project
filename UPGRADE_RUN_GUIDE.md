# Fetal Brain Abnormality Project — Upgrade Run Guide

## 1. Get the real Swin checkpoint

The repository uses Git LFS for `fetal_brain_swin_best.pth`. A Git-LFS pointer is not model weights.

```bash
git lfs install
git lfs pull
```

Verify the file is about 110 MB, not a ~130-byte text pointer.

## 2. Install

```bash
pip install -r requirements.txt
```

For the Flask app:

```bash
cd webapp
python app.py
```

Then open `http://127.0.0.1:5000`.

## 3. Add a real segmentation dataset

```text
data/
  images/
    case001.jpg
    case002.jpg
  masks/
    case001.png
    case002.png
```

Each mask should be binary: foreground > 0.

## 4. Train U-Net

From the project root:

```bash
python training/train_segmentation.py --images data/images --masks data/masks --epochs 30 --out webapp/models/segmentation_unet.pth
```

For repeated frames from the same patient/exam, create patient-level splits rather than random image splits before reporting research results.

## 5. Evaluate segmentation

```bash
python training/evaluate_segmentation.py --images data/images --masks data/masks --checkpoint webapp/models/segmentation_unet.pth
```

Report Dice and IoU.

## 6. Web-app behavior

- Without a U-Net checkpoint: segmentation is shown as a **demo ROI proposal only** and does not change classification.
- With a trained U-Net checkpoint: the predicted ROI is cropped and passed into the existing Swin-Tiny classifier.
- Grad-CAM is generated for the predicted Swin class.

## 7. Research comparison

Run and report at least:

1. Raw image -> Swin-Tiny baseline.
2. U-Net ROI -> Swin-Tiny.
3. U-Net ROI -> Swin-Tiny + Grad-CAM.

Use patient/exam-level splits where possible and report per-class metrics plus macro-F1.
