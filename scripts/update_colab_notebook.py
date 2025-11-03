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

cells.append(nbf.v4.new_code_cell("""# Setup environment - use pre-installed packages to avoid compatibility issues
import os
import shutil

# Clean start
if os.path.exists('/content/brain-tumor-hybrid-fusion-knn'):
    print('📁 Removing existing directory...')
    shutil.rmtree('/content/brain-tumor-hybrid-fusion-knn')

%cd /content
!git clone -q https://github.com/mainajajere/brain-tumor-hybrid-fusion-knn.git
%cd /content/brain-tumor-hybrid-fusion-knn

print('✅ Repository cloned')

# Use Colab's pre-installed packages to avoid compatibility issues
# Just install missing ones that we know work
!pip install -q pyyaml==6.0.1 tqdm==4.66.4 opencv-python-headless==4.9.0.80

print('✅ Minimal dependencies installed')

# Force restart to use pre-installed packages
print('🔄 Restarting runtime to use pre-installed packages...')
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
"""))

cells.append(nbf.v4.new_code_cell("""# After restart - use pre-installed packages
import os
print('✅ Runtime restarted - using pre-installed packages')

# Verify we can import everything
try:
    import tensorflow as tf
    print(f'✅ TensorFlow {tf.__version__} loaded')
except ImportError as e:
    print(f'❌ TensorFlow import failed: {e}')

try:
    import numpy as np
    print(f'✅ NumPy {np.__version__} loaded')
except ImportError as e:
    print(f'❌ NumPy import failed: {e}')

try:
    import sklearn
    print(f'✅ scikit-learn {sklearn.__version__} loaded')
except ImportError as e:
    print(f'❌ scikit-learn import failed: {e}')

try:
    import matplotlib
    print(f'✅ matplotlib {matplotlib.__version__} loaded')
except ImportError as e:
    print(f'❌ matplotlib import failed: {e}')

print('✅ Environment verification complete')
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

cells.append(nbf.v4.new_code_cell("""# Run pipeline using direct script calls with pre-installed packages
import subprocess
import sys

print("�� Running Pipeline with Pre-installed Packages")

scripts = [
    ("src/pipeline/extract_features.py", "Extracting features"),
    ("src/pipeline/train_knn.py", "Training KNN"), 
    ("src/pipeline/evaluate.py", "Evaluating model")
]

for script, description in scripts:
    print(f"\\n🔧 {description}...")
    try:
        result = subprocess.run([
            sys.executable, script, "--config", "configs/config.yaml"
        ], capture_output=True, text=True, cwd="/content/brain-tumor-hybrid-fusion-knn")
        
        if result.returncode == 0:
            print(f"✅ {description} completed")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
    except Exception as e:
        print(f"❌ {description} exception: {e}")

print("\\n🎉 Pipeline execution attempted!")
"""))

cells.append(nbf.v4.new_code_cell("""# Show any results that were generated
from IPython.display import Image, display
import os
import glob

print("📊 CHECKING FOR RESULTS")
print("=" * 50)

results_dir = "/content/brain-tumor-hybrid-fusion-knn/results"

if os.path.exists(results_dir):
    print(f"✅ Results directory exists: {results_dir}")
    
    # Find all result files
    png_files = glob.glob(f"{results_dir}/**/*.png", recursive=True)
    csv_files = glob.glob(f"{results_dir}/**/*.csv", recursive=True)
    txt_files = glob.glob(f"{results_dir}/**/*.txt", recursive=True)
    
    # Display images
    if png_files:
        print("\\n🖼️  Result Images:")
        for img_file in png_files:
            print(f"   - {os.path.basename(img_file)}")
            display(Image(filename=img_file))
    else:
        print("\\n❌ No PNG result images found")
    
    # Display text files
    all_text_files = csv_files + txt_files
    if all_text_files:
        print("\\n📄 Text Results:")
        for text_file in all_text_files:
            print(f"\\n--- {os.path.basename(text_file)} ---")
            try:
                with open(text_file, 'r') as f:
                    print(f.read())
            except Exception as e:
                print(f"   Error reading file: {e}")
    else:
        print("\\n❌ No text result files found")
        
    # Show directory structure
    print("\\n📂 Results directory structure:")
    !find /content/brain-tumor-hybrid-fusion-knn/results -type f 2>/dev/null | sort
    
else:
    print(f"❌ Results directory not found: {results_dir}")
    print("\\n📁 Checking what WAS created:")
    !find /content/brain-tumor-hybrid-fusion-knn -name "*.png" -o -name "*.csv" -o -name "*.txt" 2>/dev/null | head -20

print("\\n" + "=" * 50)
print("🎯 PIPELINE EXECUTION COMPLETE")
"""))

nb['cells'] = cells
with open('notebooks/BrainTumor_FusionKNN_Validation.ipynb', 'w') as f:
    nbf.write(nb, f)
print('✅ Compatibility-fixed notebook generated')
