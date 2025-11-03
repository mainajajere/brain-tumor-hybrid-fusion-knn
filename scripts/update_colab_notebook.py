import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Brain Tumor MRI Classification: Validation + XAI (Notebook)

This Colab runs the released pipeline end-to-end using the embedded demo dataset (10 images per class) stored in this repo under `data/images`.

**Pipeline:**
- Create stratified 64/16/20 splits
- Extract features (MobileNetV2 + EfficientNetV2B0, GAP+concat)
- Train KNN (k=5, Euclidean, distance weights)
- Evaluate (confusion matrix, class metrics)
- Display results

**No Google Drive or Kaggle required by default.**
"""))

cells.append(nbf.v4.new_code_cell("""# Setup environment
%cd /content
!git clone -q https://github.com/mainajajere/brain-tumor-hybrid-fusion-knn.git
%cd /content/brain-tumor-hybrid-fusion-knn

# Install dependencies
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

cells.append(nbf.v4.new_code_cell("""# Use the embedded demo dataset under data/images
DATA_ROOT = '/content/brain-tumor-hybrid-fusion-knn/data/images'
CLASSES   = ['glioma','meningioma','pituitary','notumor']

import os
print('Dataset root:', DATA_ROOT)
for c in CLASSES:
    p = os.path.join(DATA_ROOT, c)
    n = len(os.listdir(p)) if os.path.isdir(p) else 0
    print(f'✅ {c}: {n} images' if n > 0 else f'❌ {c}: MISSING')

if all(os.path.isdir(os.path.join(DATA_ROOT,c)) and len(os.listdir(os.path.join(DATA_ROOT,c)))>0 for c in CLASSES):
    print('✅ Dataset verified successfully')
else:
    raise SystemExit('❌ Embedded demo dataset not found')
"""))

cells.append(nbf.v4.new_code_cell("""# Write config
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
print('✅ Config written: configs/config.yaml')
"""))

cells.append(nbf.v4.new_code_cell("""# Verify scripts are available
import os
print("=== Checking scripts ===")
scripts = ['check_split_counts.py', 'run_full_pipeline.py']
for script in scripts:
    path = f'scripts/{script}'
    if os.path.exists(path):
        print(f'✅ {path} - exists ({os.path.getsize(path)} bytes)')
    else:
        print(f'❌ {path} - MISSING')

print("\\n=== Running data split check ===")
!python scripts/check_split_counts.py --config configs/config.yaml
"""))

cells.append(nbf.v4.new_code_cell("""# Run the complete pipeline
print("=== Running full pipeline ===")
!python scripts/run_full_pipeline.py --config configs/config.yaml
"""))

cells.append(nbf.v4.new_code_cell("""# Show results
from IPython.display import Image, display
import os

print("=== Pipeline Outputs ===")
outputs = [
    'outputs/figures/confusion_matrix.png',
    'outputs/figures/class_metrics.png', 
    'outputs/results/summary.txt'
]

for p in outputs:
    print(f'\\n📁 {p}')
    if p.endswith('.png') and os.path.exists(p):
        display(Image(filename=p))
    elif os.path.exists(p):
        print(open(p).read())
    else:
        print('❌ File not generated')

if os.path.exists('outputs'):
    print(f'\\n📂 All outputs in outputs/:')
    !find outputs -type f 2>/dev/null | head -15
"""))

nb['cells'] = cells
with open('notebooks/BrainTumor_FusionKNN_Validation.ipynb', 'w') as f:
    nbf.write(nb, f)
print('✅ Notebook generated')
