# Research gap and evaluation plan

## What the upgraded system adds

1. **Anatomical ROI segmentation**: a U-Net can localize the fetal brain/head region before classification.
2. **Segmentation-guided classification**: when a trained U-Net checkpoint is present, the ROI is passed to the existing Swin-Tiny classifier.
3. **Transformer explainability**: Grad-CAM is generated from the final Swin block and overlaid on the classification image.
4. **Quantitative segmentation evaluation**: Dice and IoU are provided by `training/evaluate_segmentation.py`.
5. **Ablation-ready design**: the original raw-image Swin path remains available as the baseline; compare it against the trained-segmentation ROI path.

## Do not claim novelty incorrectly

Segmentation + Grad-CAM/Grad-CAM++ has already appeared in fetal ultrasound literature. The contribution should therefore be framed around the **specific fetal-brain fine-grained classification task, anatomy-guided ROI pipeline, reproducible ablation, and quantitative explanation/localization evaluation**, not merely the existence of a heatmap.

## Recommended experiments

- Baseline: raw ultrasound -> Swin-Tiny.
- ROI model: ultrasound -> U-Net -> ROI -> Swin-Tiny.
- Full XAI: ROI model + Grad-CAM.
- Report per-class precision, recall, F1, macro-F1, balanced accuracy and confusion matrix.
- For segmentation report Dice and IoU.
- If repeated frames/patients exist, split by patient/exam rather than by image to avoid leakage.
- If an independent dataset is available, use it as external validation and report it separately from the internal test set.
- For explainability, do not treat Grad-CAM as ground-truth segmentation. If expert regions are available, evaluate localization overlap or pointing/faithfulness metrics.

## Dataset requirement

The current classification labels do not create segmentation labels. A real U-Net requires paired image/mask data. Until a trained segmentation checkpoint is supplied, the web app's fallback ROI is explicitly marked as a demo visualization and is **not** used to change the classifier prediction.
