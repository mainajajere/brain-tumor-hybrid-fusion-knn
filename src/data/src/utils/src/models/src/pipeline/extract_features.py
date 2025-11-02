import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
cat > src/pipeline/extract_features.py << 'EOF'
import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils.io import load_config, ensure_dir
from src.models.backbones import build_dual_backbones

def _load_image(path: str, image_size: int):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size))
    img = tf.cast(img, tf.float32)
    mb = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    ef = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return mb, ef

def main(cfg: dict):
    image_size = cfg["data"]["image_size"]
    outdir = os.path.join(cfg["output"]["dir"], "features")
    ensure_dir(outdir)

    mb, ef = build_dual_backbones(image_size, freeze=True)

    for split in ["train", "val", "test"]:
        csv_path = os.path.join(cfg["output"]["dir"], "splits", f"{split}.csv")
        df = pd.read_csv(csv_path)

        feats, labels = [], []
        for _, row in df.iterrows():
            mb_img, ef_img = _load_image(row["path"], image_size)

            mb_feat = mb(tf.expand_dims(mb_img, 0), training=False)
            ef_feat = ef(tf.expand_dims(ef_img, 0), training=False)

            gap_mb = tf.reduce_mean(mb_feat, axis=[1, 2])
            gap_ef = tf.reduce_mean(ef_feat, axis=[1, 2])

            fused = tf.concat([gap_mb, gap_ef], axis=-1).numpy().squeeze()
            feats.append(fused)
            labels.append(int(row["label"]))

        np.savez(os.path.join(outdir, f"{split}.npz"),
                 X=np.stack(feats), y=np.array(labels))

    print("Saved features to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
EOF