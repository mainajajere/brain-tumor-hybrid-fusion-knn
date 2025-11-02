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
        over = overlay(raw, cam)
        fn = f"{cname}_pred{classes[pred]}_{os.path.basename(row['path'])}.png"
        cv2.imwrite(os.path.join(outdir, fn), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
print("Saved Grad-CAM to", outdir)
