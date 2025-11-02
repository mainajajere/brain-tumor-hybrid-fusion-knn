import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
cat > src/pipeline/evaluate.py << 'EOF'
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.utils.io import load_config, ensure_dir

def _plot_confusion(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def _plot_roc(y_true, y_prob, class_names, out_path):
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{cname} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main(cfg: dict):
    classes = cfg["data"]["classes"]
    fdir = os.path.join(cfg["output"]["dir"], "features")
    t = np.load(os.path.join(fdir, "test.npz"))
    X_test, y_test = t["X"], t["y"]

    clf = joblib.load(os.path.join(cfg["output"]["dir"], "models", "knn.joblib"))
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
    report = classification_report(y_test,38 y_pred,
                                   target_names=classes,
                                   output_dict=True,
                                   zero_division=0)

    outdir = os.path.join(cfg["output"]["dir"], "test")
    ensure_dir(outdir)

    _plot_confusion(cm, classes, os.path.join(outdir, "confusion.png"))
    _plot_roc(y_test, y_prob, classes, os.path.join(outdir, "roc_curves.png"))
    pd.DataFrame(report).transpose().to_csv(os.path.join(outdir, "class_metrics.csv"))

    print("Saved evaluation to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
EOF