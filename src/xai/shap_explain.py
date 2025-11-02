import os, numpy as np, pandas as pd, tensorflow as tf, shap, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
with open("configs/default.yaml","r") as f: cfg = yaml.safe_load(f)
image_size = cfg["data"]["image_size"]; classes = cfg["data"]["classes"]
logits_path = os.path.join(cfg["output"]["dir"], "models", "aux_logits")
model = tf.keras.models.load_model(logits_path, compile=False)

def load_img(path):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size))
    return tf.cast(img, tf.float32)

train_csv = os.path.join(cfg["output"]["dir"], "splits", "train.csv")
test_csv = os.path.join(cfg["output"]["dir"], "splits", "test.csv")
tr = pd.read_csv(train_csv); te = pd.read_csv(test_csv)

bg = []
for ci, cname in enumerate(classes):
    sample = tr[tr["label"]==ci].head(5)
    for _, r in sample.iterrows():
        bg.append(load_img(r["path"]))
bg = tf.stack(bg, axis=0)

explainer = shap.GradientExplainer(model, bg)
outdir = os.path.join(cfg["output"]["dir"], "xai", "shap")
os.makedirs(outdir, exist_ok=True)

for ci, cname in enumerate(classes):
    sample = te[te["label"]==ci].head(2)
    if sample.empty: continue
    xs = tf.stack([load_img(p) for p in sample["path"].values], axis=0)
    shap_values = explainer.shap_values(xs)
    preds = tf.argmax(model(xs), axis=1).numpy()
    sv_list = [shap_values[p][i] for i,p in enumerate(preds)]
    shap.image_plot(sv_list, xs.numpy(), show=False)
    plt.savefig(os.path.join(outdir, f"summary_{cname}.png"), dpi=200, bbox_inches="tight")
    plt.close()
print("Saved SHAP to", outdir)
