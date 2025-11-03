import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Brain Tumor MRI Classification: Validation + XAI (Notebook)

This Colab runs the released pipeline end-to-end using the embedded demo dataset (10 images per class) stored in this repo under `data/images`.

Pipeline:
- Create stratified 64/16/20 splits
- Extract features (MobileNetV2 + EfficientNetV2B0, GAP+concat)
- Train KNN (k=5, Euclidean, distance weights)
- Evaluate (confusion, class metrics)
- Optional: Grad-CAM + SHAP explanations

No Google Drive or Kaggle required by default.
"""))

cells.append(nbf.v4.new_code_cell("""!nvidia-smi -L || true
%cd /content
!git clone https://github.com/mainajajere/brain-tumor-hybrid-fusion-knn.git
%cd /content/brain-tumor-hybrid-fusion-knn
!pip install -q -r requirements.txt

import os, sys, pathlib, yaml
REPO = pathlib.Path('/content/brain-tumor-hybrid-fusion-knn')
os.makedirs(REPO/'outputs', exist_ok=True)
os.makedirs(REPO/'results', exist_ok=True)
sys.path.append(str(REPO))
print('Repo ready at', REPO)
"""))

cells.append(nbf.v4.new_code_cell("""# Use the embedded demo dataset under data/images (lowercase class names)
DATA_ROOT = '/content/brain-tumor-hybrid-fusion-knn/data/images'
CLASSES   = ['glioma','meningioma','pituitary','notumor']

import os
def have_dataset(root, classes):
    return all(os.path.isdir(os.path.join(root,c)) and len(os.listdir(os.path.join(root,c)))>0 for c in classes)

print('Dataset root:', DATA_ROOT)
for c in CLASSES:
    p = os.path.join(DATA_ROOT, c)
    print(c, 'OK' if os.path.isdir(p) else 'MISSING', p)

if not have_dataset(DATA_ROOT, CLASSES):
    raise SystemExit('Embedded demo dataset not found. Expected data/images/<class> folders.')
"""))

cells.append(nbf.v4.new_code_cell("""# Write config (64/16/20 split; KNN k=5 Euclidean distance weighting)
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
!sed -n '1,160p' configs/config.yaml
"""))

cells.append(nbf.v4.new_code_cell("""# Run the pipeline
!python scripts/run_full_pipeline.py --config configs/config.yaml
"""))

cells.append(nbf.v4.new_code_cell("""# Show key outputs
from IPython.display import Image, display
for p in ['outputs/figures/confusion_matrix.png',
          'outputs/figures/class_metrics.png',
          'outputs/results/summary.txt']:
    print('\\n', p)
    if p.endswith('.png') and os.path.exists(p): display(Image(filename=p))
    elif os.path.exists(p): print(open(p).read())
    else: print('MISSING')
"""))

nb['cells'] = cells
with open('notebooks/BrainTumor_FusionKNN_Validation.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Notebook overwritten with self-contained demo run.')
