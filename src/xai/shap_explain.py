import os, numpy as np, pandas as pd, tensorflow as tf, shap, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.layers import TFSMLayer

with open("configs/default.yaml","r") as f: cfg = yaml.safe_load(f)
image_size = cfg["data"]["image_size"]; classes = cfg["data"]["classes"]
logits_path = os.path.join(cfg["output"]["dir"], "models", "aux_logits")

print(f"Loading logits model from {logits_path}...")
try:
    model = TFSMLayer(logits_path, call_endpoint='serving_default')
    print("Model loaded with TFSMLayer")
except Exception as e:
    print(f"Failed to load: {e}")
    exit(1)

def load_img(path):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size))
    return tf.cast(img, tf.float32)

train_csv = os.path.join(cfg["output"]["dir"], "splits", "train.csv")
test_csv = os.path.join(cfg["output"]["dir"], "splits", "test.csv")
if not os.path.exists(test_csv):
    print(f"TEST CSV MISSING: {test_csv}")
    exit(1)
tr = pd.read_csv(train_csv); te = pd.read_csv(test_csv)

bg = []
for ci in range(len(classes)):
    sample = tr[tr["label"]==ci].head(5)
    for _, r in sample.iterrows():
        if os.path.exists(r["path"]):
            bg.append(load_img(r["path"]))
if len(bg) == 0:
    print("No background images!")
    exit(1)
bg = tf.stack(bg)

explainer = shap.GradientExplainer(model, bg)
outdir = os.path.join(cfg["output"]["dir"], "xai", "shap")
os.makedirs(outdir, exist_ok=True)

saved = 0
for ci, cname in enumerate(classes):
    sample = te[te["label"]==ci].head(2)
    if sample.empty: continue
    xs = []
    for p in sample["path"]:
        if os.path.exists(p):
            xs.append(load_img(p))
    if not xs: continue
    xs = tf.stack(xs)
    shap_values = explainer.shap_values(xs)
    preds = tf.argmax(model(xs), axis=1).numpy()
    sv_list = [shap_values[p][i] for i,p in enumerate(preds)]
    shap.image_plot(sv_list, xs.numpy(), show=False)
    plt.savefig(os.path.join(outdir, f"summary_{cname}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    saved += 1
print(f"Saved {saved} SHAP summaries to {outdir}")
