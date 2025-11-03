import argparse
import os
import subprocess
import sys

def main(cfg_path):
    print("🧠 Brain Tumor Classification - Full Pipeline")
    print("=" * 50)
    
    # Verify config exists
    if not os.path.exists(cfg_path):
        print(f"❌ Config file not found: {cfg_path}")
        sys.exit(1)
    
    # Get absolute path to ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Step 1: Extract features
    print("�� Step 1: Extracting features with dual backbones")
    result1 = subprocess.run([
        sys.executable, "src/pipeline/extract_features.py", 
        "--config", cfg_path
    ], capture_output=True, text=True)
    
    if result1.returncode == 0:
        print("✅ Feature extraction completed")
        print(f"   Output: {result1.stdout.strip()}")
    else:
        print(f"❌ Feature extraction failed: {result1.stderr.strip()}")
        sys.exit(1)
    
    # Step 2: Train KNN
    print("🚀 Step 2: Training KNN classifier")
    result2 = subprocess.run([
        sys.executable, "src/pipeline/train_knn.py", 
        "--config", cfg_path
    ], capture_output=True, text=True)
    
    if result2.returncode == 0:
        print("✅ KNN training completed")
        print(f"   Output: {result2.stdout.strip()}")
    else:
        print(f"❌ KNN training failed: {result2.stderr.strip()}")
        sys.exit(1)
    
    # Step 3: Evaluate
    print("🚀 Step 3: Evaluating model performance")
    result3 = subprocess.run([
        sys.executable, "src/pipeline/evaluate.py", 
        "--config", cfg_path
    ], capture_output=True, text=True)
    
    if result3.returncode == 0:
        print("✅ Evaluation completed")
        print(f"   Output: {result3.stdout.strip()}")
    else:
        print(f"❌ Evaluation failed: {result3.stderr.strip()}")
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 Pipeline completed successfully!")
    print("📁 Outputs saved to results/ directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full brain tumor classification pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args.config)
