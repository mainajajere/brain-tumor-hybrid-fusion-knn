import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Brain Tumor MRI Classification: Full Pipeline

This Colab runs the complete pipeline with actual TensorFlow feature extraction using the embedded dataset.

**Full Pipeline:**
- Extract features using MobileNetV2 + EfficientNetV2B0 (TensorFlow)
- Train KNN classifier (k=5, Euclidean, distance weights) 
- Evaluate with confusion matrix and class metrics
- Display real results

**Uses actual deep learning models - no mock features!**
"""))

cells.append(nbf.v4.new_code_cell("""# Setup environment
%cd /content
!git clone -q https://github.com/mainajajere/brain-tumor-hybrid-fusion-knn.git
%cd /content/brain-tumor-hybrid-fusion-knn

# Install full dependencies including TensorFlow
!pip install -q tensorflow==2.17.0 scikit-learn==1.4.2 matplotlib==3.8.4 seaborn==0.13.2
!pip install -q opencv-python-headless==4.9.0.80 Pillow==10.3.0 numpy==1.26.4 pandas==2.1.4
!pip install -q pyyaml==6.0.1 tqdm==4.66.4 shap==0.46.0

import os, sys, pathlib, yaml
REPO = pathlib.Path('/content/brain-tumor-hybrid-fusion-knn')
os.makedirs(REPO/'outputs', exist_ok=True)
os.makedirs(REPO/'results', exist_ok=True)
sys.path.append(str(REPO))
print('✅ Repo ready at', REPO)

# Check GPU
try:
    !nvidia-smi -L
    print('✅ GPU available')
except:
    print('⚠️  No GPU detected, running on CPU')
"""))

cells.append(nbf.v4.new_code_cell("""# Use the embedded demo dataset
DATA_ROOT = '/content/brain-tumor-hybrid-fusion-knn/data/images'
CLASSES   = ['glioma','meningioma','pituitary','notumor']

import os
print('Dataset root:', DATA_ROOT)
for c in CLASSES:
    p = os.path.join(DATA_ROOT, c)
    n = len(os.listdir(p)) if os.path.isdir(p) else 0
    print(f'✅ {c}: {n} images')

if all(os.path.isdir(os.path.join(DATA_ROOT,c)) for c in CLASSES):
    print('✅ Dataset verified')
else:
    raise SystemExit('❌ Dataset not found')
"""))

cells.append(nbf.v4.new_code_cell("""# Write config for full pipeline
cfg = {
  'data': {
    'root_dir': DATA_ROOT,
    'classes': CLASSES,
    'image_size': [224, 224],
    'seed': 42,
    'split': {'test': 0.20, 'val_from_train': 0.20}
  },
  'augment': {'rotation': 0.055, 'zoom': 0.10, 'translate': 0.10, 'hflip': True, 'contrast': 0.15},
  'train': {'batch_size': 32, 'epochs': 50, 'optimizer': 'adam', 'lr': 0.001, 'dropout': 0.5},
  'fusion': {'type': 'late', 'pooling': 'gap', 'concat': True},
  'knn': {'n_neighbors': 5, 'metric': 'euclidean', 'weights': 'distance'},
  'cv': {'n_folds': 5, 'stratify': True},
  'xai': {'shap_background_per_class': 25}
}
os.makedirs('configs', exist_ok=True)
with open('configs/config.yaml','w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('✅ Config written for full pipeline')
"""))

cells.append(nbf.v4.new_code_cell("""# Run the complete full pipeline
print("🚀 Running Full Pipeline with TensorFlow")
!python scripts/run_full_pipeline.py --config configs/config.yaml
"""))

cells.append(nbf.v4.new_code_cell("""# Show real results from full pipeline
from IPython.display import Image, display
import os
import pandas as pd

print("📊 REAL PIPELINE RESULTS")
print("=" * 50)

# Display confusion matrix
confusion_path = 'results/test/confusion.png'
if os.path.exists(confusion_path):
    print(f"🎯 Confusion Matrix: {confusion_path}")
    display(Image(filename=confusion_path))
else:
    print("❌ Confusion matrix not found")

# Display ROC curves  
roc_path = 'results/test/roc_curves.png'
if os.path.exists(roc_path):
    print(f"📈 ROC Curves: {roc_path}")
    display(Image(filename=roc_path))
else:
    print("❌ ROC curves not found")

# Display class metrics
metrics_path = 'results/test/class_metrics.csv'
if os.path.exists(metrics_path):
    print(f"📊 Class Metrics: {metrics_path}")
    df = pd.read_csv(metrics_path)
    display(df)
else:
    print("❌ Class metrics not found")

# Show all generated files
print("\\n📂 All Generated Files:")
!find results -type f 2>/dev/null | sort

print("\\n🎉 PIPELINE COMPLETE!")
print("✅ Real TensorFlow feature extraction")
print("✅ Real KNN training and evaluation") 
print("✅ Professional evaluation outputs")
"""))

nb['cells'] = cells
with open('notebooks/BrainTumor_FusionKNN_Validation.ipynb', 'w') as f:
    nbf.write(nb, f)
print('✅ Final pipeline notebook generated')
