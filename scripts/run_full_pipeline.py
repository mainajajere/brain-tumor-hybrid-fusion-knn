import argparse
import yaml
import os
import numpy as np
from collections import Counter

# Import your custom modules
from src.data.dataset import list_images, make_splits, build_ds
from src.data.augment import build_augment
from src.models.backbones import fused_extractor
from src.train.extract_features import extract_features
from src.train.train_knn import fit_knn
from src.eval.evaluate import eval_knn
from src.eval.crossval import run_cv
from viz.plots import plot_confusion, bar_class_metrics

def main(cfg_path):
    # Load configuration
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg["data"]
    split_cfg = data_cfg["split"]
    seed = data_cfg["seed"]
    classes = data_cfg["classes"]
    
    print("=== Brain Tumor Classification Pipeline ===")
    print("Listing images...")
    paths, labels = list_images(data_cfg["root_dir"], classes)
    print(f"Found {len(paths)} total images")
    
    print("Creating stratified splits...")
    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = make_splits(paths, labels, split_cfg, seed)
    
    # Calculate and display split counts
    def get_counts(y):
        c = Counter(y)
        return [c[i] for i in range(len(classes))]
    
    train_counts = get_counts(tr_y)
    val_counts = get_counts(va_y)
    test_counts = get_counts(te_y)
    
    print(f"Train split: {train_counts} (Total: {len(tr_y)})")
    print(f"Val split:   {val_counts} (Total: {len(va_y)})")
    print(f"Test split:  {test_counts} (Total: {len(te_y)})")
    
    print("Building datasets with augmentation...")
    augment_fn = build_augment(cfg["augment"])
    tr_ds = build_ds(tr_p, tr_y, augment=augment_fn, 
                     image_size=data_cfg["image_size"], 
                     batch_size=cfg["train"]["batch_size"])
    va_ds = build_ds(va_p, va_y, augment=None,
                     image_size=data_cfg["image_size"], 
                     batch_size=cfg["train"]["batch_size"])
    te_ds = build_ds(te_p, te_y, augment=None,
                     image_size=data_cfg["image_size"], 
                     batch_size=cfg["train"]["batch_size"])
    
    print("Initializing dual-backbone feature extractor...")
    extractor = fused_extractor(cfg["fusion"])
    
    print("Extracting features from all splits...")
    X_tr, y_tr = extract_features(extractor, tr_ds)
    X_va, y_va = extract_features(extractor, va_ds)
    X_te, y_te = extract_features(extractor, te_ds)
    
    # Combine for cross-validation
    X_all = np.concatenate([X_tr, X_va, X_te])
    y_all = np.concatenate([y_tr, y_va, y_te])
    
    print(f"Feature shapes - Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    
    print("Training KNN classifier...")
    knn_cfg = cfg["knn"]
    clf = fit_knn(X_tr, y_tr, k=knn_cfg["n_neighbors"], 
                  metric=knn_cfg["metric"], 
                  weights=knn_cfg["weights"])
    
    print("Evaluating on test set...")
    acc, pr, rc, f1, cm, yhat = eval_knn(clf, X_te, y_te, n_classes=len(classes))
    
    # Create output directories
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    
    print("Generating evaluation plots...")
    plot_confusion(cm, classes, "outputs/figures/confusion_matrix.png")
    bar_class_metrics(pr, rc, f1, classes, "outputs/figures/class_metrics.png")
    
    print("Running cross-validation...")
    cv = run_cv(X_all, y_all, n_folds=cfg["cv"]["n_folds"], seed=seed)
    
    # Write comprehensive results summary
    with open("outputs/results/summary.txt", "w") as f:
        f.write("=== BRAIN TUMOR CLASSIFICATION RESULTS ===\\n\\n")
        f.write(f"Test Accuracy: {acc:.4f}\\n\\n")
        
        f.write("Class-wise Metrics:\\n")
        for i, c in enumerate(classes):
            f.write(f"  {c:12} Precision={pr[i]:.3f}, Recall={rc[i]:.3f}, F1={f1[i]:.3f}\\n")
        
        f.write("\\nDataset Split Counts:\\n")
        f.write(f"  Train: {train_counts} (Total={len(tr_y)})\\n")
        f.write(f"  Val:   {val_counts} (Total={len(va_y)})\\n")
        f.write(f"  Test:  {test_counts} (Total={len(te_y)})\\n")
        
        f.write(f"\\n5-Fold Cross-Validation:\\n")
        f.write(f"  Accuracies: {[f'{a:.4f}' for a in cv['accs']]}\\n")
        f.write(f"  F1 Scores:  {[f'{f:.4f}' for f in cv['f1s']]}\\n")
        f.write(f"  Shapiro-Wilk Normality Test:\\n")
        f.write(f"    Accuracy p-value: {cv['p_norm_acc']:.4f}\\n")
        f.write(f"    F1 Score p-value: {cv['p_norm_f1']:.4f}\\n")
    
    print("\\n=== PIPELINE COMPLETE ===")
    print(f"Test Accuracy: {acc:.4f}")
    print("Outputs saved to:")
    print("  - outputs/figures/confusion_matrix.png")
    print("  - outputs/figures/class_metrics.png") 
    print("  - outputs/results/summary.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full brain tumor classification pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found!")
        exit(1)
        
    main(args.config)
