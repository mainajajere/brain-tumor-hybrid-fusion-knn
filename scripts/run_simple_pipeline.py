import argparse
import yaml
import os
import numpy as np
from collections import Counter
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def list_images(root_dir, classes):
    """Simple image lister without complex dependencies"""
    paths, labels = [], []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.exists(class_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                paths.extend(glob.glob(os.path.join(class_dir, ext)))
                labels.extend([label] * len(glob.glob(os.path.join(class_dir, ext))))
    return paths, labels

def make_splits(paths, labels, split_cfg, seed=42):
    """Simple train/val/test split"""
    test_size = split_cfg['test']
    val_from_train = split_cfg['val_from_train']
    
    # First split: test vs train+val
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    
    # Second split: train vs val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_from_train, 
        stratify=train_val_labels, random_state=seed
    )
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def extract_simple_features(paths, labels, image_size=224):
    """
    Mock feature extraction for demo purposes
    In a real scenario, this would use TensorFlow models
    """
    print("🔧 Using mock feature extraction for demo")
    print("📝 In full version, this would use MobileNetV2 + EfficientNetV2B0")
    
    # Generate random features for demo (replace with actual model extraction)
    n_samples = len(paths)
    n_features = 1280 + 1280  # Mock MobileNetV2 + EfficientNetV2B0 features
    features = np.random.randn(n_samples, n_features)
    
    return features, np.array(labels)

def main(cfg_path):
    print("🚀 Starting Simple Brain Tumor Classification Pipeline")
    
    # Load config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg["data"]
    classes = data_cfg["classes"]
    
    print("📁 Listing images...")
    paths, labels = list_images(data_cfg["root_dir"], classes)
    print(f"📊 Found {len(paths)} total images")
    
    print("🎯 Creating stratified splits...")
    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = make_splits(paths, labels, data_cfg["split"], data_cfg["seed"])
    
    # Display split counts
    def get_counts(y):
        c = Counter(y)
        return [c[i] for i in range(len(classes))]
    
    train_counts = get_counts(tr_y)
    val_counts = get_counts(va_y) 
    test_counts = get_counts(te_y)
    
    print(f"📈 Train: {train_counts} (Total: {len(tr_y)})")
    print(f"📈 Val:   {val_counts} (Total: {len(va_y)})")
    print(f"📈 Test:  {test_counts} (Total: {len(te_y)})")
    
    print("🔍 Extracting features (mock for demo)...")
    X_tr, y_tr = extract_simple_features(tr_p, tr_y)
    X_te, y_te = extract_simple_features(te_p, te_y)
    
    print(f"📐 Feature shapes - Train: {X_tr.shape}, Test: {X_te.shape}")
    
    print("🤖 Training KNN classifier...")
    knn_cfg = cfg["knn"]
    clf = KNeighborsClassifier(
        n_neighbors=knn_cfg["n_neighbors"],
        metric=knn_cfg["metric"], 
        weights=knn_cfg["weights"]
    )
    clf.fit(X_tr, y_tr)
    
    print("📊 Evaluating on test set...")
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    
    # Create outputs
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/figures/confusion_matrix.png', dpi=100)
    plt.close()
    
    # Class metrics
    report = classification_report(y_te, y_pred, target_names=classes, output_dict=True)
    metrics = []
    for cls in classes:
        if cls in report:
            metrics.append({
                'class': cls,
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'], 
                'f1': report[cls]['f1-score']
            })
    
    # Plot metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x - width, [m['precision'] for m in metrics], width, label='Precision')
    ax.bar(x, [m['recall'] for m in metrics], width, label='Recall')
    ax.bar(x + width, [m['f1'] for m in metrics], width, label='F1-Score')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Class-wise Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/class_metrics.png', dpi=100)
    plt.close()
    
    # Write summary
    with open("outputs/results/summary.txt", "w") as f:
        f.write("=== BRAIN TUMOR CLASSIFICATION RESULTS ===\\n\\n")
        f.write(f"Test Accuracy: {acc:.4f}\\n\\n")
        f.write("Class-wise Metrics:\\n")
        for m in metrics:
            f.write(f"  {m['class']:12} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}\\n")
        f.write("\\nDataset Split Counts:\\n")
        f.write(f"  Train: {train_counts} (Total={len(tr_y)})\\n")
        f.write(f"  Val:   {val_counts} (Total={len(va_y)})\\n") 
        f.write(f"  Test:  {test_counts} (Total={len(te_y)})\\n")
        f.write("\\n⚠️  NOTE: Using mock feature extraction for demo\\n")
        f.write("    Real version uses MobileNetV2 + EfficientNetV2B0 feature fusion\\n")
    
    print("🎉 PIPELINE COMPLETE!")
    print(f"📈 Test Accuracy: {acc:.4f}")
    print("📁 Outputs saved to outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
