import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
cat > src/pipeline/train_knn.py << 'EOF'
import argparse
import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.utils.io import load_config, ensure_dir

def main(cfg: dict):
    fdir = os.path.join(cfg["output"]["dir"], "features")
    tr = np.load(os.path.join(fdir, "train.npz"))
    va = np.load(os.path.join(fdir, "val.npz"))

    X = np.vstack([tr["X"], va["X"]])
    y = np.hstack([tr["y"], va["y"]])

    clf = KNeighborsClassifier(n_neighbors=5,
                               metric="euclidean",
                               weights="distance")
    clf.fit(X, y)

    mdir = os.path.join(cfg["output"]["dir"], "models")
    ensure_dir(mdir)
    joblib.dump(clf, os.path.join(mdir, "knn.joblib"))

    print("Saved KNN to", mdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
EOF