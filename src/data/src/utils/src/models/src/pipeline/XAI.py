# ------------------------------------------------------------------
# 0. Go to repo root
# ------------------------------------------------------------------
cd /workspaces/brain-tumor-hybrid-fusion-knn
pwd

# ------------------------------------------------------------------
# 1. Install missing deps
# ------------------------------------------------------------------
python -m pip install --upgrade pip
pip install --no-cache-dir \
    shap==0.46.0 \
    tensorflow-cpu==2.17.0 \
    numpy==1.26.4 \
    matplotlib==3.8.4 \
    opencv-python==4.9.0.80

# ------------------------------------------------------------------
# 2. Create XAI scripts directory
# ------------------------------------------------------------------
mkdir -p src/xai
mkdir -p results/xai/gradcam results/xai/shap
mkdir -p docs/figures/xai

# ------------------------------------------------------------------
# 3. Step 1: train_aux_head.py
# ------------------------------------------------------------------
cat > src/xai/train_aux_head.py << 'PY'
import os, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import yaml

CFG_PATH = "configs/default.yaml"
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

image_size = cfg["data"]["image_size"]
classes = cfg["data"]["classes"]
n_classes = len(classes)

def load_df(csv_path):
    return pd.read_csv(csv_path)

def load_ds(df, batch=16, shuffle=False):
    def _load(path, label):
        b = tf.io.read_file(path)
        img = tf.image.decode_image(b, channels=3, expand_animations=False)
        img = tf.image.resize(img, (image_size, image_size))
        img = tf.cast(img, tf.float32)
        return img, tf.cast(label, tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

# Build dual-backbone
inputs = layers.Input(shape=(image_size, image_size, 3))
mb_in = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
ef_in = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)

mb = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                       input_shape=(image_size, image_size, 3))
ef = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet",
                                            input_shape=(image_size, image_size, 3))
mb.trainable = False
ef.trainable = False

mb_conv = mb(mb_in, training=False)
ef_conv = ef(ef_in, training=False)
gap_mb = layers.GlobalAveragePooling2D()(mb_conv)
gap_ef = layers.GlobalAveragePooling2D()(ef_conv)
fused = layers.Concatenate()([gap_mb, gap_ef])
fused = layers.Dropout(0.5)(fused)
logits = layers.Dense(n_classes, activation="softmax", name="aux_softmax")(fused)

logits_model = models.Model(inputs=inputs, outputs=logits, name="aux_logits")
cam_model = models.Model(inputs=inputs, outputs=[logits, mb_conv, ef_conv], name="aux_cam")

# Data
train_df = load_df(os.path.join(cfg["output"]["dir"], "splits", "train.csv"))
val_df = load_df(os.path.join(cfg["output"]["dir"], "splits", "val.csv"))
ds_tr = load_ds(train_df, batch=16, shuffle=True)
ds_va = load_ds(val_df, batch=16, shuffle=False)

# Train
opt = optimizers.Adam(learning_rate=1e-3)
logits_model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es = callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
logits_model.fit(ds_tr, validation_data=ds_va, epochs=30, callbacks=[es], verbose=1)

# Save
out_dir = os.path.join(cfg["output"]["dir"], "models")
os.makedirs(out_dir, exist_ok=True)
logits_model.save(os.path.join(out_dir, "aux_logits"))
cam_model.save(os.path.join(out_dir, "aux_cam"))
print("Saved aux models to", out_dir)
PY

# ------------------------------------------------------------------
# 4. Step 2: gradcam.py
# ------------------------------------------------------------------
cat > src/xai/gradcam.py << 'PY'
import os, cv2, numpy as np, pandas as pd, tensorflow as tf, yaml
with open("configs/default.yaml","r") as f: cfg = yaml.safe_load(f)
image_size = cfg["data"]["image_size"]; classes = cfg["data"]["classes"]
cam_path = os.path.join(cfg["output"]["dir"], "models", "aux_cam")
model = tf.keras.models.load_model(cam_path, compile=False)

def load_img(path):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size))
    raw = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8).numpy()
    return tf.cast(img, tf.float32), raw

def gradcam_for_class(img_tensor, class_idx):
    with tf.GradientTape() as tape:
        logits, conv_mb, conv_ef = model(tf.expand_dims(img_tensor,0), training=False)
        score = logits[:, class_idx]
    grads_mb = tape.gradient(score, conv_mb)
    grads_ef = tape.gradient(score, conv_ef)
    w_mb = tf.reduce_mean(grads_mb, axis=(1,2)); w_ef = tf.reduce_mean(grads_ef, axis=(1,2))
    cam_mb = tf.reduce_sum(w_mb[:,None,None,:] * conv_mb, axis=-1)[0].numpy()
    cam_ef = tf.reduce_sum(w_ef[:,None,None,:] * conv_ef, axis=-1)[0].numpy()
    cam_mb = np.maximum(cam_mb, 0); cam_ef = np.maximum(cam_ef, 0)
    cam_mb = (cam_mb - cam_mb.min()) / (cam_mb.max() + 1e-8)
    cam_ef = (cam_ef - cam_ef.min()) / (cam_ef.max() + 1e-8)
    cam = (cam_mb + cam_ef)/2.0
    cam = cv2.resize(cam, (image_size, image_size))
    return cam

def overlay(raw_rgb, cam, alpha=0.35):
    heat = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (heat*alpha + raw_rgb*(1-alpha)).astype(np.uint8)
    return out

test_csv = os.path.join(cfg["output"]["dir"], "splits", "test.csv")
df = pd.read_csv(test_csv)
outdir = os.path.join(cfg["output"]["dir"], "xai", "gradcam")
os.makedirs(outdir, exist_ok=True)

for ci, cname in enumerate(classes):
    subset = df[df["label"] == ci].head(3)
    for _, row in subset.iterrows():
        img_t, raw = load_img(row["path"])
        logits, _, _ = model(tf.expand_dims(img_t,0), training=False)
        pred = int(tf.argmax(logits[0]).numpy())
        cam = gradcam_for_class(img_t, pred)
        over = overlay(raw, cam, alpha=0.35)
        fn = f"{cname}_pred{classes[pred]}_{os.path.basename(row['path'])}.png"
        cv2.imwrite(os.path.join(outdir, fn), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
print("Saved Grad-CAM overlays to", outdir)
PY

# ------------------------------------------------------------------
# 5. Step 3: shap_explain.py
# ------------------------------------------------------------------
cat > src/xai/shap_explain.py << 'PY'
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
if not bg: raise SystemExit("No background images found.")
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
print("Saved SHAP images to", outdir)
PY

# ------------------------------------------------------------------
# 6. Run the XAI pipeline
# ------------------------------------------------------------------
python src/xai/train_aux_head.py
python src/xai/gradcam.py
python src/xai/shap_explain.py

# ------------------------------------------------------------------
# 7. Copy example figures to docs
# ------------------------------------------------------------------
cp results/xai/gradcam/*.png docs/figures/xai/ 2>/dev/null || true
cp results/xai/shap/*.png docs/figures/xai/ 2>/dev/null || true

# ------------------------------------------------------------------
# 8. Update README
# ------------------------------------------------------------------
if ! grep -q "XAI" README.md; then
    cat >> README.md << 'EOF'


## XAI: Grad-CAM & SHAP
We provide interpretability:
- **Grad-CAM overlays**: `results/xai/gradcam/`
- **SHAP explanations**: `results/xai/shap/`
- Example figures: `docs/figures/xai/`

See paper for discussion.
EOF
    echo "XAI section added to README"
fi

# ------------------------------------------------------------------
# 9. Commit & push
# ------------------------------------------------------------------
git add -A
git commit -m "feat(xai): add Grad-CAM + SHAP with aux head; example figures"
git push

echo "XAI pipeline complete! Check docs/figures/xai/"