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

inputs = layers.Input(shape=(image_size, image_size, 3))
mb_in = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
ef_in = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)

mb = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
ef = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
mb.trainable = False; ef.trainable = False

mb_conv = mb(mb_in, training=False)
ef_conv = ef(ef_in, training=False)
gap_mb = layers.GlobalAveragePooling2D()(mb_conv)
gap_ef = layers.GlobalAveragePooling2D()(ef_conv)
fused = layers.Concatenate()([gap_mb, gap_ef])
fused = layers.Dropout(0.5)(fused)
logits = layers.Dense(n_classes, activation="softmax", name="aux_softmax")(fused)

logits_model = models.Model(inputs=inputs, outputs=logits, name="aux_logits")
cam_model = models.Model(inputs=inputs, outputs=[logits, mb_conv, ef_conv], name="aux_cam")

train_df = load_df(os.path.join(cfg["output"]["dir"], "splits", "train.csv"))
val_df = load_df(os.path.join(cfg["output"]["dir"], "splits", "val.csv"))
ds_tr = load_ds(train_df, batch=16, shuffle=True)
ds_va = load_ds(val_df, batch=16, shuffle=False)

opt = optim  optimizers.Adam(learning_rate=1e-3)
logits_model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es = callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
logits_model.fit(ds_tr, validation_data=ds_va, epochs=30, callbacks=[es], verbose=1)

out_dir = os.path.join(cfg["output"]["dir"], "models")
os.makedirs(out_dir, exist_ok=True)
logits_model.save(os.path.join(out_dir, "aux_logits"))
cam_model.save(os.path.join(out_dir, "aux_cam"))
print("Saved aux models to", out_dir)
