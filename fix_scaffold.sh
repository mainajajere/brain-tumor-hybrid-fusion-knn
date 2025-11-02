#!/usr/bin/env bash
set -euo pipefail
mkdir -p src/{data,models,pipeline,utils}

Make src a package
for d in src src/data src/models src/pipeline src/utils; do touch "$d/init.py"; done

configs/default.yaml (overwrite to be safe)
cat > configs/default.yaml << 'YAML'
data:
root: data/images
image_size: 224
classes: [glioma, meningioma, pituitary, notumor]
split: {train: 0.64, val: 0.16, test: 0.20}
stratify: true
seed: 1337
extensions: [".png", ".jpg", ".jpeg"]

model:
freeze: true

knn:
k: 5
metric: euclidean
weights: distance

output:
dir: results
YAML

src/utils/io.py
cat > src/utils/io.py << 'PY'
import os, yaml
def load_config(path):
with open(path, "r") as f:
return yaml.safe_load(f)
def ensure_dir(path):
os.makedirs(path, exist_ok=True)
PY

src/utils/metrics.py
cat > src/utils/metrics.py << 'PY'
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
def compute_metrics(y_true, y_pred, y_prob, class_names):
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
auc = None
try:
auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
except Exception:
pass
return cm, report, auc
PY

src/models/backbones.py
cat > src/models/backbones.py << 'PY'
import tensorflow as tf

def mobilenet_v2_backbone(image_size=224, freeze=True):
base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
input_shape=(image_size, image_size, 3))
base.trainable = not freeze
return base

def efficientnet_v2_b0_backbone(image_size=224, freeze=True):
base = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet",
input_shape=(image_size, image_size, 3))
base.trainable = not freeze
return base

def build_dual_backbones(image_size=224, freeze=True):
mb = mobilenet_v2_backbone(image_size, freeze)
ef = efficientnet_v2_b0_backbone(image_size, freeze)
return mb, ef
PY

src/data/prepare_splits.py
cat > src/data/prepare_splits.py << 'PY'
import argparse, os, glob, pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.io import load_config, ensure_dir

def find_images(root, class_names, exts):
rows=[]
for ci, cname in enumerate(class_names):
cdir=os.path.join(root, cname)
files=[]
for e in exts:
files += glob.glob(os.path.join(cdir, f"*{e}"))
rows += [{"path": fp, "label": ci, "class": cname} for fp in files]
return pd.DataFrame(rows)

def main(cfg):
df = find_images(cfg["data"]["root"], cfg["data"]["classes"], cfg["data"]["extensions"])
assert len(df) > 0, f"No images found under {cfg['data']['root']}."
trr, vr, ter = cfg["data"]["split"]["train"], cfg["data"]["split"]["val"], cfg["data"]["split"]["test"]
strat = df["label"] if cfg["data"]["stratify"] else None
trainval, test = train_test_split(df, test_size=ter, stratify=strat, random_state=cfg["data"]["seed"])
val_rel = vr / (trr + vr)
strat_tv = trainval["label"] if cfg["data"]["stratify"] else None
train, val = train_test_split(trainval, test_size=val_rel, stratify=strat_tv, random_state=cfg["data"]["seed"])outdir = os.path.join(cfg["output"]["dir"], "splits"); ensure_dir(outdir)
train.to_csv(os.path.join(outdir, "train.csv"), index=False)
val.to_csv(os.path.join(outdir, "val.csv"), index=False)
test.to_csv(os.path.join(outdir, "test.csv"), index=False)

def counts(x): return x.groupby("class").size().reindex(cfg["data"]["classes"]).fillna(0).astype(int)
pd.DataFrame({"train": counts(train), "val": counts(val), "test": counts(test)}).to_csv(os.path.join(outdir, "counts.csv"))
print("Saved splits to", outdir)if name == "main":
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = load_config(args.config)
main(cfg)
PY

src/pipeline/extract_features.py
cat > src/pipeline/extract_features.py << 'PY'
import argparse, os, numpy as np, pandas as pd, tensorflow as tf
from tqdm import tqdm
from src.utils.io import load_config, ensure_dir
from src.models.backbones import build_dual_backbones

def load_image(path, image_size):
b = tf.io.read_file(path)
img = tf.image.decode_image(b, channels=3, expand_animations=False)
img = tf.image.resize(img, (image_size, image_size))
img = tf.cast(img, tf.float32)
mb = tf.keras.applications.mobilenet_v2.preprocess_input(img)
ef = tf.keras.applications.efficientnet_v2.preprocess_input(img)
return mb, ef

def main(cfg):
image_size = cfg["data"]["image_size"]
outdir = os.path.join(cfg["output"]["dir"], "features"); ensure_dir(outdir)
mb, ef = build_dual_backbones(image_size, freeze=True)for split in ["train","val","test"]:
    df = pd.read_csv(os.path.join(cfg["output"]["dir"], "splits", f"{split}.csv"))
    feats, labels = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Extract {split}"):
        mb_img, ef_img = load_image(r["path"], image_size)
        mb_feat = mb(tf.expand_dims(mb_img, 0), training=False)
        ef_feat = ef(tf.expand_dims(ef_img, 0), training=False)
        gap_mb = tf.reduce_mean(mb_feat, axis=[1,2])
        gap_ef = tf.reduce_mean(ef_feat, axis=[1,2])
        fused = tf.concat([gap_mb, gap_ef], axis=-1).numpy().squeeze()
        feats.append(fused); labels.append(int(r["label"]))
    np.savez(os.path.join(outdir, f"{split}.npz"), X=np.stack(feats), y=np.array(labels))
print("Saved features to", outdir)if name == "main":
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = load_config(args.config)
main(cfg)
PY

src/pipeline/train_knn.py
cat > src/pipeline/train_knn.py << 'PY'
import argparse, os, joblib, numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.utils.io import load_config, ensure_dir

def main(cfg):
fdir = os.path.join(cfg["output"]["dir"], "features")
tr = np.load(os.path.join(fdir, "train.npz")); va = np.load(os.path.join(fdir, "val.npz"))
X = np.vstack([tr["X"], va["X"]]); y = np.hstack([tr["y"], va["y"]])
kc = cfg["knn"]
clf = KNeighborsClassifier(n_neighbors=kc["k"], metric=kc["metric"], weights=kc["weights"])
clf.fit(X, y)
mdir = os.path.join(cfg["output"]["dir"], "models"); ensure_dir(mdir)
joblib.dump(clf, os.path.join(mdir, "knn.joblib"))
print("Saved KNN to", mdir)

if name == "main":
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = load_config(args.config)
main(cfg)
PY

src/pipeline/evaluate.py
cat > src/pipeline/evaluate.py << 'PY'
import argparse, os, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from src.utils.io import load_config, ensure_dir
from src.utils.metrics import compute_metrics

def plot_confusion(cm, class_names, out_path):
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_roc(y_true, y_prob, class_names, out_path):
n = len(class_names)
y_true_bin = label_binarize(y_true, classes=list(range(n)))
plt.figure(figsize=(6,5))
for i in range(n):
fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
plt.savefig(out_path); plt.close()

def main(cfg):
class_names = cfg["data"]["classes"]
fdir = os.path.join(cfg["output"]["dir"], "features")
t = np.load(os.path.join(fdir, "test.npz"))
X_test, y_test = t["X"], t["y"]clf = joblib.load(os.path.join(cfg["output"]["dir"], "models", "knn.joblib"))
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

cm, report, auc_val = compute_metrics(y_test, y_pred, y_prob, class_names)

outdir = os.path.join(cfg["output"]["dir"], "test"); ensure_dir(outdir)
plot_confusion(cm, class_names, os.path.join(outdir, "confusion.png"))
plot_roc(y_test, y_prob, class_names, os.path.join(outdir, "roc_curves.png"))
pd.DataFrame(report).transpose().to_csv(os.path.join(outdir, "class_metrics.csv"))
print("Saved evaluation to", outdir, "AUC(ovr)=", auc_val)if name == "main":
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = load_config(args.config)
main(cfg)
PY
