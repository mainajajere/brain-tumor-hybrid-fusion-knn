# Brain Tumor Classification: Hybrid Deep Learning with Interpretability

A robust pipeline for MRI-based brain tumor classification using dual-backbone feature fusion and KNN classification with comprehensive explainable AI (XAI).

## 🧠 Overview

**Hybrid deep-learning pipeline** for MRI-based brain tumor classification with interpretability:
- **Dual backbones**: MobileNetV2 + EfficientNetV2B0
- **Late fusion**: Global average pooling + concatenation (+ dropout)
- **Classifier**: KNN (k=5, Euclidean, distance weighting)  
- **XAI**: Grad-CAM and SHAP (including waterfall plots in the paper)

**One-click Colab run** uses an embedded 40-image mini dataset (10 images per class) in this repo, so anyone can validate the pipeline without Google Drive or Kaggle.

## 🚀 Quick Start

### Colab (One-Click, No Setup)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mainajajere/brain-tumor-hybrid-fusion-knn/blob/main/notebooks/BrainTumor_FusionKNN_Validation.ipynb)

1. **Click the Colab badge above**
2. **Run all cells** - The notebook will:
   - Clone this repo in Colab
   - Use the embedded dataset at `data/images` (no Drive/Kaggle required)
   - Write `configs/config.yaml` with a 64/16/20 split
   - Run the full pipeline via `scripts/run_full_pipeline.py`
   - Display confusion matrix and class-wise metrics

### Local Installation
```bash
git clone https://github.com/mainajajere/brain-tumor-hybrid-fusion-knn
cd brain-tumor-hybrid-fusion-knn
pip install -r requirements.txt
📊 Key Outputs
Performance figures: outputs/figures/confusion_matrix.png, outputs/figures/class_metrics.png

Summary: outputs/results/summary.txt

Optional XAI: outputs/xai/gradcam/, outputs/xai/shap/ (run extra notebook cells)

🗂️ Repository Structure

brain-tumor-hybrid-fusion-knn/
├── data/images/                 # Embedded mini dataset (10 images/class)
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
├── configs/config.yaml          # Configuration file
├── scripts/run_full_pipeline.py # End-to-end runner
├── src/                         # Python modules
│   ├── data/                    # Data loading
│   ├── models/                  # Dual-backbone architecture
│   ├── train/                   # Training utilities
│   └── eval/                    # Evaluation metrics
└── notebooks/BrainTumor_FusionKNN_Validation.ipynb
⚙️ Pipeline Flow
Split: Stratified 64/16/20 (holdout test=20%; from remaining 80%, val=20% → final 64/16/20)

Feature Extraction: MobileNetV2 + EfficientNetV2B0 (ImageNet pretrained), GAP each

Fusion: Concatenate pooled features + dropout (0.5)

Classifier: KNN (k=5, Euclidean, distance weighting)

Evaluation: Confusion matrix, class-wise precision/recall/F1, 5-fold CV with normality tests

Optional XAI: Grad-CAM (class-wise overlays) and SHAP (summary + waterfall)

🎯 Usage
Run Full Pipeline

python scripts/run_full_pipeline.py --config configs/config.yaml

Using Your Own Dataset

Folder layout must match lowercase class names:


your_dataset/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/

Update data.root_dir and data.classes in configs/config.yaml accordingly.

📋 Sample Configuration

data:
  root_dir: data/images
  classes: [glioma, meningioma, pituitary, notumor]
  image_size: [224, 224]
  seed: 42
  split: {test: 0.20, val_from_train: 0.20}

augment:
  rotation: 0.055
  zoom: 0.10
  translate: 0.10
  hflip: true
  contrast: 0.15

train:
  batch_size: 32
  epochs: 50
  optimizer: adam
  lr: 0.001
  dropout: 0.5

fusion:
  type: late
  pooling: gap
  concat: true

knn:
  n_neighbors: 5
  metric: euclidean
  weights: distance

cv:
  n_folds: 5
  stratify: true

xai:
  shap_background_per_class: 25

🔍 Reproduce Figures
Confusion matrix: outputs/figures/confusion_matrix.png

Class metrics: outputs/figures/class_metrics.png

Optional XAI: Grad-CAM and SHAP figures produced by extra notebook cells (outputs/xai/...)

🛠️ Troubleshooting
Colab link error "malformed GitHub path": Ensure the link uses:


https://colab.research.google.com/github/mainajajere/brain-tumor-hybrid-fusion-knn/blob/main/notebooks/BrainTumor_FusionKNN_Validation.ipynb
Missing dataset folders: Embedded demo is in data/images. For custom datasets, set data.root_dir and ensure folder names match classes exactly (lowercase).

No outputs: Ensure scripts/run_full_pipeline.py ran without errors; check configs/config.yaml paths.

SHAP/Grad-CAM not generated: Run the optional XAI cells in the notebook. KNN is non-differentiable, so a small auxiliary softmax head is trained in-notebook solely for explanations.

CPU-only: Training the auxiliary head is small; the rest is feature extraction + KNN, which is light.
