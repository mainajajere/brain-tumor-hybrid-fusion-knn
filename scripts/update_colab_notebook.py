import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Brain Tumor MRI Classification: Demo Pipeline

This Colab runs a **demo version** of the pipeline using the embedded dataset. For the full version with actual deep learning feature extraction, run the code locally.

**Demo Pipeline:**
- Create stratified 64/16/20 splits  
- Mock feature extraction (random features for demo)
- Train KNN (k=5, Euclidean, distance weights)
- Evaluate and display results

**Note:** Uses mock features for Colab demo. Real version uses MobileNetV2+EfficientNetV2B0.
"""))

cells.append(nbf.v4.new_code_cell("""# Setup environment
%cd /content
!git clone -q https://github.com/mainajajere/brain-tumor-hybrid-fusion-knn.git
%cd /content/brain-tumor-hybrid-fusion-knn

# Install minimal dependencies for demo
!pip install -q scikit-learn==1.4.2 matplotlib==3.8.4 seaborn==0.13.2
!pip install -q numpy==1.26.4 pandas==2.1.4 pyyaml==6.0.1

import os, sys, pathlib, yaml
REPO = pathlib.Path('/content/brain-tumor-hybrid-fusion-knn')
os.makedirs(REPO/'outputs', exist_ok=True)
sys.path.append(str(REPO))
print('✅ Repo ready at', REPO)
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

cells.append(nbf.v4.new_code_cell("""# Write config
cfg = {
  'data': {
    'root_dir': DATA_ROOT,
    'classes': CLASSES,
    'image_size': [224, 224],
    'seed': 42,
    'split': {'test': 0.20, 'val_from_train': 0.20}
  },
  'knn': {'n_neighbors': 5, 'metric': 'euclidean', 'weights': 'distance'}
}
os.makedirs('configs', exist_ok=True)
with open('configs/config.yaml','w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('✅ Config written')
"""))

cells.append(nbf.v4.new_code_cell("""# Run the simple demo pipeline
print("🚀 Running Demo Pipeline (mock features)")
!python scripts/run_simple_pipeline.py --config configs/config.yaml
"""))

cells.append(nbf.v4.new_code_cell("""# Show results
from IPython.display import Image, display
import os

print("📊 Demo Results:")
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

print("\\n💡 Note: This is a demo with mock features.")
print("   For real feature extraction, run the full pipeline locally.")
"""))

nb['cells'] = cells
with open('notebooks/BrainTumor_FusionKNN_Validation.ipynb', 'w') as f:
    nbf.write(nb, f)
print('✅ Demo notebook generated')
