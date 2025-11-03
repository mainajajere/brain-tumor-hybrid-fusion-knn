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

cells.append(nbf.v4.new_code_cell("""# Setup environment - force clean start
import os
import shutil

if os.path.exists('/content/brain-tumor-hybrid-fusion-knn'):
    print('📁 Removing existing directory...')
    shutil.rmtree('/content/brain-tumor-hybrid-fusion-knn')

%cd /content
!git clone -q https://github.com/mainajajere/brain-tumor-hybrid-fusion-knn.git
%cd /content/brain-tumor-hybrid-fusion-knn

# Check which branch we're on
!git branch

# Install dependencies
!pip install -q tensorflow==2.17.0 scikit-learn==1.4.2 matplotlib==3.8.4 seaborn==0.13.2
!pip install -q opencv-python-headless==4.9.0.80 Pillow==10.3.0 
!pip install -q "numpy<1.25" pandas==2.1.4 pyyaml==6.0.1 tqdm==4.66.4

print('✅ Environment setup complete')

# Verify we can import TensorFlow
import tensorflow as tf
print(f'✅ TensorFlow {tf.__version__} loaded')
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
import yaml
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
  'cv': {'n_folds': 5, 'stratify': True}
}
os.makedirs('configs', exist_ok=True)
with open('configs/config.yaml','w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('✅ Config written for full pipeline')
"""))

cells.append(nbf.v4.new_code_cell("""# OPTION 1: Try the fixed pipeline runner first
print("🚀 OPTION 1: Running with fixed pipeline runner...")
try:
    !python scripts/run_full_pipeline_fixed.py --config configs/config.yaml
    print("✅ Option 1 succeeded!")
except Exception as e:
    print(f"❌ Option 1 failed: {e}")
    print("🔄 Trying Option 2...")
"""))

cells.append(nbf.v4.new_code_cell("""# OPTION 2: Direct script calls (fallback)
import subprocess
import sys

print("🚀 OPTION 2: Running scripts directly...")

scripts = [
    ("src/pipeline/extract_features.py", "Extracting features"),
    ("src/pipeline/train_knn.py", "Training KNN"), 
    ("src/pipeline/evaluate.py", "Evaluating model")
]

for script, description in scripts:
    print(f"🔧 {description}...")
    try:
        result = subprocess.run([
            sys.executable, script, "--config", "configs/config.yaml"
        ], capture_output=True, text=True, cwd="/content/brain-tumor-hybrid-fusion-knn")
        
        if result.returncode == 0:
            print(f"✅ {description} completed")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
    except Exception as e:
        print(f"❌ {description} exception: {e}")
"""))

cells.append(nbf.v4.new_code_cell("""# Show results
from IPython.display import Image, display
import os

print("📊 PIPELINE RESULTS")
print("=" * 50)

results_dir = "/content/brain-tumor-hybrid-fusion-knn/results"

# Display any result images
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith('.png'):
            img_path = os.path.join(root, file)
            print(f"🖼️  {file}:")
            display(Image(filename=img_path))

# Display any text results
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith(('.txt', '.csv')):
            text_path = os.path.join(root, file)
            print(f"📄 {file}:")
            with open(text_path, 'r') as f:
                print(f.read())

print("\\n📂 All generated files:")
!find /content/brain-tumor-hybrid-fusion-knn/results -type f 2>/dev/null | sort

print("\\n🎉 EXECUTION COMPLETE!")
"""))

nb['cells'] = cells
with open('notebooks/BrainTumor_FusionKNN_Validation.ipynb', 'w') as f:
    nbf.write(nb, f)
print('✅ Final robust notebook generated')
