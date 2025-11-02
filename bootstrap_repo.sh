bootstrap_repo.sh
#!/usr/bin/env bash
set -euo pipefail

Always work at repo root
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

Create folders
mkdir -p configs src/{data,models,pipeline,xai,utils} scripts results .github/workflows

.gitignore
cat > .gitignore << 'EOF'
pycache/
.ipynb_checkpoints/
.env
.venv
env/
venv/
results/
data/*/
!.gitkeep
EOF

README
cat > README.md << 'EOF'
Explainable Hybrid Deep Learning Framework Integrating MobileNetV2, EfficientNetV2B0, and KNN for MRI-Based Brain Tumor Classification

Dual-backbone late fusion (MobileNetV2 + EfficientNetV2B0 with GAP+concat) and a KNN head (k=5, Euclidean, distance weights). 64/16/20 stratified split. XAI via Grad-CAM and SHAP (N=25/class). No LIME or ViT baseline.

Quickstart

pip install -r requirements-tf212.txt
Edit configs/default.yaml (data.root)
python -m src.data.prepare_splits --config configs/default.yaml
python -m src.pipeline.extract_features --config configs/default.yaml
python -m src.pipeline.train_knn --config configs/default.yaml
python -m src.pipeline.evaluate --config configs/default.yaml EOF
Requirements (TF 2.12)
cat > requirements-tf212.txt << 'EOF'
tensorflow==2.12.0
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
opencv-python==4.8.0.76
Pillow==9.5.0
pyyaml==6.0
shap==0.43.0
scipy==1.10.1
statsmodels==0.14.0
tqdm==4.66.1
joblib==1.3.2
EOF

Config
cat > configs/default.yaml << 'EOF'
data:
root: data/images
image_size: 224
classes: [glioma, meningioma, pituitary, notumor]
split: {train: 0.64, val: 0.16, test: 0.20}
stratify: true
seed: 1337
extensions: [".png", ".jpg", ".jpeg"]

model:
backbones: [mobilenet_v2, efficientnet_v2_b0]
pretrained: imagenet
freeze: true
pooling: gap
fusion: concat
aux_dropout: 0.5

knn:
k: 5
metric: euclidean
weights: distance

training_aux_head:
optimizer: adam
lr: 0.001
batch_size: 32
epochs: 50
early_stopping_patience: 5

xai:
shap_background_per_class: 25
shap_clip_percentile: 99
gradcam_combine: mean

cv:
folds: 5
seed: 1337

output:
dir: results
EOF

Utils
cat > src/utils/seed.py << 'EOF'
import os, random, numpy as np, tensorflow as tf
def set_seed(seed: int = 1337):
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
EOF

cat > src/utils/io.py << 'EOF'
import os, yaml
def load_config(path):
with open(path, "r") as f:
return yaml.safe_load(f)
def ensure_dir(path):
os.makedirs(path, exist_ok=True)
EOF

cat > src/utils/metrics.py << 'EOF'
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
def compute_metrics(y_true, y_pred, y_prob, class_names):
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
auc = None
try: auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
except Exception: pass
return cm, report, auc
EOF

Data split
cat > src/data/prepare_splits.py << 'EOF'
import argparse, os, glob, pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.io import load_config, ensure_dir
def find_images(root, class_names, exts):
rows=[]
for ci,cname in enumerate(class_names):
cdir=os.path.join(root,cname); files=[]
for e in exts: files+=glob.glob(os.path.join(cdir,f"*{e}"))
rows += [{"path":fp,"label":ci,"class":cname} for fp in files]
return pd.DataFrame(rows)
def main(cfg):
df=find_images(cfg["data"]["root"], cfg["data"]["classes"], cfg["data"]["extensions"])
assert len(df)>0, "No images found."
trr, vr, ter = cfg["data"]["split"]["train"], cfg["data"]["split"]["val"], cfg["data"]["split"]["test"]
strat=df["label"] if cfg["data"]["stratify"] else None
trainval, test = train_test_split(df, test_size=ter, stratify=strat, random_state=cfg["data"]["seed"])
val_rel = vr/(trr+vr)
strat_tv=trainval["label"] if cfg["data"]["stratify"] else None
train, val = train_test_split(trainval, test_size=val_rel, stratify=strat_tv, random_state=cfg["data"]["seed"])
outdir=os.path.join(cfg["output"]["dir"],"splits"); ensure_dir(outdir)
train.to_csv(os.path.join(outdir,"train.csv"), index=False)
val.to_csv(os.path.join(outdir,"val.csv"), index=False)
test.to_csv(os.path.join(outdir,"test.csv"), index=False)
def counts(x): return x.groupby("class").size().reindex(cfg["data"]["classes"]).fillna(0).astype(int)
counts_df = {"train":counts(train), "val":counts(val), "test":counts(test)}
pd.DataFrame(counts_df).to_csv(os.path.join(outdir,"counts.csv"))
print("Saved splits to", outdir)
if name=="main":
p=argparse.ArgumentParser(); p.add_argument("--config", required=True); a=p.parse_args()
cfg=load_config(a.config); main(cfg)
EOF

Models
cat > src/models/backbones.py << 'EOF'
import tensorflow as tf
def mobilenet_v2_backbone(image_size=224):
return tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(image_size,image_size,3))
def efficientnet_v2_b0_backbone(image_size=224):
try:
return tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(image_size,image_size,3))
except Exception:
from keras_efficientnet_v2 import EfficientNetV2B0
return EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(image_size,image_size,3))
def build_dual_backbones(image_size=224, freeze=True):
mb=mobilenet_v2_backbone(image_size); ef=efficientnet_v2_b0_backbone(image_size)
if freeze:
mb.trainable=False; ef.trainable=False
return mb, ef
EOF

cat > src/models/aux_head.py << 'EOF'
import tensorflow as tf
from src.models.backbones import build_dual_backbones
def build_aux_model(image_size, n_classes, dropout=0.5, freeze_backbones=True):
mb, ef = build_dual_backbones(image_size, freeze=freeze_backbones)
inputs = tf.keras.Input(shape=(image_size,image_size,3))
x_mb = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x_ef = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
feat_mb = mb(x_mb, training=False)
feat_ef = ef(x_ef, training=False)
gap_mb = tf.keras.layers.GlobalAveragePooling2D()(feat_mb)
gap_ef = tf.keras.layers.GlobalAveragePooling2D()(feat_ef)
fused = tf.keras.layers.Concatenate()([gap_mb, gap_ef])
fused = tf.keras.layers.Dropout(dropout)(fused)
logits = tf.keras.layers.Dense(n_classes, activation="softmax", name="aux_softmax")(fused)
return tf.keras.Model(inputs=inputs, outputs=[logits, feat_mb, feat_ef], name="aux_dual_backbone")
EOF

Feature extraction
cat > src/pipeline/extract_features.py << 'EOF'
import argparse, os, numpy as np, pandas as pd, tensorflow as tf
from tqdm import tqdm
from src.utils.io import load_config, ensure_dir
from src.models.backbones import build_dual_backbones
def load_image(path, image_size):
img=tf.io.read_file(path)
img=tf.image.decode_image(img, channels=3, expand_animations=False)
img=tf.image.resize(img,(image_size,image_size))
img=tf.cast(img, tf.float32)
mb=tf.keras.applications.mobilenet_v2.preprocess_input(img)
ef=tf.keras.applications.efficientnet_v2.preprocess_input(img)
return mb, ef
def main(cfg):
image_size=cfg["data"]["image_size"]
outdir=os.path.join(cfg["output"]["dir"],"features"); ensure_dir(outdir)
mb, ef = build_dual_backbones(image_size, freeze=True)
for split in ["train","val","test"]:
df=pd.read_csv(os.path.join(cfg["output"]["dir"],"splits",f"{split}.csv"))
feats, labels = [], []
for _,r in tqdm(df.iterrows(), total=len(df), desc=f"Extract {split}"):
mb_img, ef_img = load_image(r["path"], image_size)
mb_feat=mb(tf.expand_dims(mb_img,0), training=False)
ef_feat=ef(tf.expand_dims(ef_img,0), training=False)
gap_mb=tf.reduce_mean(mb_feat,axis=[1,2]); gap_ef=tf.reduce_mean(ef_feat,axis=[1,2])
fused=tf.concat([gap_mb,gap_ef],axis=-1).numpy().squeeze()
feats.append(fused); labels.append(int(r["label"]))
np.savez(os.path.join(outdir,f"{split}.npz"), X=np.stack(feats), y=np.array(labels))
print("Saved features to", outdir)
if name=="main":
p=argparse.ArgumentParser(); p.add_argument("--config", required=True); a=p.parse_args()
cfg=load_config(a.config); main(cfg)
EOF

Train KNN
cat > src/pipeline/train_knn.py << 'EOF'
import argparse, os, joblib, numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.utils.io import load_config, ensure_dir
def main(cfg):
fdir=os.path.join(cfg["output"]["dir"],"features")
tr=np.load(os.path.join(fdir,"train.npz")); va=np.load(os.path.join(fdir,"val.npz"))
X=np.vstack([tr["X"], va["X"]]); y=np.hstack([tr["y"], va["y"]])
kc=cfg["knn"]; clf=KNeighborsClassifier(n_neighbors=kc["k"], metric=kc["metric"], weights=kc["weights"])
clf.fit(X,y)
mdir=os.path.join(cfg["output"]["dir"],"models"); ensure_dir(mdir)
joblib.dump(clf, os.path.join(mdir,"knn.joblib")); print("Saved KNN to", mdir)
if name=="main":
p=argparse.ArgumentParser(); p.add_argument("--config", required=True); a=p.parse_args()
cfg=load_config(a.config); main(cfg)
EOF

Evaluate
cat > src/pipeline/evaluate.py << 'EOF'
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
n=len(class_names); y_true_bin=label_binarize(y_true, classes=list(range(n)))
plt.figure(figsize=(6,5))
for i in range(n):
fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_prob[:,i])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
plt.savefig(out_path); plt.close()
def main(cfg):
class_names=cfg["data"]["classes"]
fdir=os.path.join(cfg["output"]["dir"],"features"); t=np.load(os.path.join(fdir,"test.npz"))
X_test, y_test = t["X"], t["y"]
clf=joblib.load(os.path.join(cfg["output"]["dir"],"models","knn.joblib"))
y_pred=clf.predict(X_test); y_prob=clf.predict_proba(X_test)
cm, report, auc_val = compute_metrics(y_test, y_pred, y_prob, class_names)
outdir=os.path.join(cfg["output"]["dir"],"test"); ensure_dir(outdir)
plot_confusion(cm, class_names, os.path.join(outdir,"confusion.png"))
plot_roc(y_test, y_prob, class_names, os.path.join(outdir,"roc_curves.png"))
pd.DataFrame(report).transpose().to_csv(os.path.join(outdir,"class_metrics.csv"))
print("Saved evaluation to", outdir, "AUC(ovr)=", auc_val)
if name=="main":
p=argparse.ArgumentParser(); p.add_argument("--config", required=True); a=p.parse_args()
cfg=load_config(a.config); main(cfg)
EOF

Cross-val, stats
cat > src/pipeline/crossval.py << 'EOF'
import argparse, os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.utils.io import load_config, ensure_dir
def main(cfg):
fdir=os.path.join(cfg["output"]["dir"],"features")
tr=np.load(os.path.join(fdir,"train.npz")); va=np.load(os.path.join(fdir,"val.npz"))
X=np.vstack([tr["X"], va["X"]]); y=np.hstack([tr["y"], va["y"]])
skf=StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=cfg["cv"]["seed"])
rows=[]
for i,(tr_idx,te_idx) in enumerate(skf.split(X,y),1):
clf=KNeighborsClassifier(n_neighbors=cfg["knn"]["k"], metric=cfg["knn"]["metric"], weights=cfg["knn"]["weights"])
clf.fit(X[tr_idx], y[tr_idx]); pred=clf.predict(X[te_idx])
rows.append({"fold":i,"accuracy":accuracy_score(y[te_idx],pred),"f1_macro":f1_score(y[te_idx],pred,average="macro")})
df=pd.DataFrame(rows); outdir=os.path.join(cfg["output"]["dir"],"cv"); ensure_dir(outdir)
df.to_csv(os.path.join(outdir,"fold_metrics.csv"), index=False); print(df.describe())
if name=="main":
p=argparse.ArgumentParser(); p.add_argument("--config", required=True); a=p.parse_args()
cfg=load_config(a.config); main(cfg)
EOF

cat > src/pipeline/stats.py << 'EOF'
import argparse, os, pandas as pd
from scipy.stats import shapiro
from src.utils.io import load_config, ensure_dir
def main(cfg):
cv_path=os.path.join(cfg["output"]["dir"],"cv","fold_metrics.csv")
df=pd.read_csv(cv_path); W,p=shapiro(df["accuracy"]); normal=p>=0.05
outdir=os.path.join(cfg["output"]["dir"],"stats"); ensure_dir(outdir)
pd.DataFrame([{"metric":"accuracy","shapiro_W":W,"shapiro_p":p,"normal":normal}]).to_csv(os.path.join(outdir,"tests.csv"), index=False)
print("Saved stats to", outdir)
if name=="main":
p=argparse.ArgumentParser(); p.add_argument("--config", required=True); a=p.parse_args()
cfg=load_config(a.config); main(cfg)
EOF

XAI (Grad-CAM, SHAP)
cat > src/xai/train_aux_head.py << 'EOF'
import argparse, os, pandas as pd, tensorflow as tf
from src.utils.io import load_config, ensure_dir
from src.utils.seed import set_seed
from src.models.aux_head import build_aux_model
def load_ds(df, image_size, batch, shuffle=False):
def _load(path, label):
img=tf.io.read_file(path); img=tf.image.decode_image(img,3,expand_animations=False)
img=tf.image.resize(img,(image_size,image_size)); img=tf.cast(img, tf.float32)
return img, tf.cast(label, tf.int32)
ds=tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values)).map(_load, num_parallel_calls=tf.data.AUTOTUNE)
if shuffle: ds=ds.shuffle(4096, reshuffle_each_iteration=True)
return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
def main(cfg):
set_seed(cfg["data"]["seed"])
image_size=cfg["data"]["image_size"]; n_classes=len(cfg["data"]["classes"])
outdir=os.path.join(cfg["output"]["dir"],"models"); ensure_dir(outdir)
tr=pd.read_csv(os.path.join(cfg["output"]["dir"],"splits","train.csv"))
va=pd.read_csv(os.path.join(cfg["output"]["dir"],"splits","val.csv"))
model=build_aux_model(image_size, n_classes, dropout=cfg["model"]["aux_dropout"], freeze_backbones=True)
opt=tf.keras.optimizers.Adam(learning_rate=cfg["training_aux_head"]["lr"])
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es=tf.keras.callbacks.EarlyStopping(patience=cfg["training_aux_head"]["early_stopping_patience"], restore_best_weights=True, monitor="val_accuracy")
ds_tr=load_ds(tr, image_size, cfg["training_aux_head"]["batch_size"], shuffle=True)
ds_va=load_ds(va, image_size, cfg["training_aux_head"]["batch_size"], shuffle=False)
model.fit(ds_tr, validation_data=ds_va, epochs=cfg["training_aux_head"]["epochs"], callbacks=[es], verbose=1)
model.save(os.path.join(outdir,"aux_head")); print("Saved aux head.")
if name=="main":
p=argparse.ArgumentParser(); p.add_argument("--config", required=True); a=p.parse_args()
cfg=load_config(a.config); main(cfg)
EOF

cat > src/xai/gradcam.py << 'EOF'
import argparse, os, numpy as np, pandas as pd, cv2, tensorflow as tf
from src.utils.io import load_config, ensure_dir
def gradcam(model, img, class_idx, which="mb"):
with tf.GradientTape() as tape:
logits, conv_mb, conv_ef = model(tf.expand_dims(img,0), training=False)
score = logits[:, class_idx]
conv_out = conv_mb if which=="mb" else conv_ef
grads = tape.gradient(score, conv_out); weights = tf.reduce_mean(grads, axis=(1,2))
cam = tf.reduce_sum(weights[:,None,None,:] * conv_out, axis=-1); cam = tf.nn.relu(cam)[0].numpy()
cam = (cam - cam.min()) / (cam.max() + 1e-8); return cam
def load_raw(path, image_size):
b=tf.io.read_file(path); im=tf.image.decode_image(b,3,expand_animations=False); im=tf.image.resize(im,(image_size,image_size))
raw=tf.cast(tf.clip_by_value(im,0,255), tf.uint8).numpy(); return tf.cast(im, tf.float32), raw
def overlay(raw, cam, alpha=0.35):
heat=cv2.applyColorMap(np.uint8(255cam), cv2.COLORMAP_JET); heat=cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
return (heatalpha + raw*(1-alpha)).astype(np.uint8)
def main(cfg):
img_sz=cfg["data"]["image_size"]; classes=cfg["data"]["classes"]
outdir=os.path.join(cfg["output"]["dir"],"xai","gradcam"); ensure_dir(outdir)
model=tf.keras.models.load_model(os.path.join(cfg["output"]["dir"],"models","aux_head"), compile=False)
df=pd.read_csv(os.path.join(cfg["output"]["dir"],"splits","test.csv"))
ex=[]
for c in classes: ex += df[df["class"]==c].head(5).to_dict("records")
for r in ex:
img, raw = load_raw(r["path"], img_sz)
logits,,=model(tf.expand_dims(img,0), training=False); pred=int(tf.argmax(logits[0]).numpy())
cam_mb=gradcam(model, img, pred, "mb"); cam_ef=gradcam(model, img, pred, "ef")
cam=(cam_mb+cam_ef)/2.0 if cfg["xai"]["gradcam_combine"]=="mean" else np.maximum(cam_mb,cam_ef)
over=overlay(raw, cam, 0.35)
out=os.path.join(outdir, f"{r['class']}pred{classes[pred]}{os.path.basename(r['path'])}.png")
cv2.imwrite(out, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
print("Saved Grad-CAM to", outdir)
if name=="main":
p=argparse.ArgumentParser(); p.add_argument("--config", required=True); a=p.parse_args()
cfg=load_config(a.config); main(cfg)
EOF

cat > src/xai/shap_explain.py << 'EOF'
import argparse, os, numpy as np, pandas as pd, shap, tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.io import load_config, ensure_dir
def load_image(path, image_size):
b=tf.io.read_file(path); im=tf.image.decode_image(b,3,expand_animations=False); im=tf.image.resize(im,(image_size,image_size))
return tf.cast(im, tf.float32)
def main(cfg):
img_sz=cfg["data"]["image_size"]; classes=cfg["data"]["classes"]
outdir=os.path.join(cfg["output"]["dir"],"xai","shap"); ensure_dir(outdir); ensure_dir(os.path.join(outdir,"waterfalls"))
model=tf.keras.models.load_model(os.path.join(cfg["output"]["dir"],"models","aux_head"), compile=False)
tr=pd.read_csv(os.path.join(cfg["output"]["dir"],"splits","train.csv"))
te=pd.read_csv(os.path.join(cfg["output"]["dir"],"splits","test.csv"))
bgN=cfg["xai"]["shap_background_per_class"]; bg=[]
for ci,_ in enumerate(classes):
sub=tr[tr["label"]==ci].head(bgN)["path"].tolist()
for p in sub: bg.append(load_image(p, img_sz))
bg=tf.stack(bg,0)
explainer=shap.GradientExplainer((model, model.layers[-1].output), bg)
for ci,cname in enumerate(classes):
sample=te[te["label"]==ci].head(2)
for ,row in sample.iterrows():
x=tf.expand_dims(load_image(row["path"], img_sz),0)
shap_values,=explainer.shap_values(x)
pred=int(np.argmax(model(x)[0].numpy()))
sv=shap_values[pred][0]
shap.image_plot([sv], show=False)
plt.savefig(os.path.join(outdir, f"summary_{cname}.png")); plt.close()
flat_sv = sv.mean(axis=(0,1))
shap.plots.waterfall.waterfall_legacy(base_value=0, shap_values=flat_sv, max_display=20, show=False)
plt.savefig(os.path.join(outdir,"waterfalls", f"{cname}{os.path.basename(row['path'])}.png")); plt.close()
print("Saved SHAP to", outdir)
if name=="main":
p=argparse.ArgumentParser(); p.add_argument("--config", required=True); a=p.parse_args()
cfg=load_config(a.config); main(cfg)
EOF

Make all
cat > scripts/make_all.sh << 'EOF'
#!/usr/bin/env bash
set -e
CFG=${1:-configs/default.yaml}
python -m src.data.prepare_splits --config $CFG
python -m src.pipeline.extract_features --config $CFG
python -m src.pipeline.train_knn --config $CFG
python -m src.pipeline.evaluate --config $CFG
python -m src.pipeline.crossval --config $CFG
python -m src.pipeline.stats --config $CFG
python -m src.xai.train_aux_head --config $CFG
python -m src.xai.gradcam --config $CFG
python -m src.xai.shap_explain --config $CFG
EOF
chmod +x scripts/make_all.sh

Commit
git add .
git commit -m "scaffold: dual-backbone GAP+concat, KNN head, splits, CV, Grad-CAM, SHAP"

echo "Done. Now: git push -u origin scaffold/v1"