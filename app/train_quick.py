"""
Quick Training Script - Sample-based Training
==============================================

This script trains the Random Forest classifier on a SUBSET of images
for faster testing and development. Use this when you want to quickly
test the full pipeline without waiting for hours.

Usage:
------
    python -m app.train_quick

Options (environment variables):
--------------------------------
    SAMPLE_SIZE: Number of images per class (default: 500)
    
    Example: SAMPLE_SIZE=1000 python -m app.train_quick
"""

import os
import sys
import random
import numpy as np
from datetime import datetime
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATASET_DIR, MODELS_DIR, CLASS_NAMES
from app.feature_extraction import DETRFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import cv2


def load_sample_data(
    dataset_path: str,
    split: str = "train",
    samples_per_class: int = 500
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load a random sample of images from each class.
    
    Args:
        dataset_path: Path to the dataset
        split: 'train' or 'val'
        samples_per_class: Number of images to sample from each class
        
    Returns:
        Tuple of (images, labels)
    """
    images = []
    labels = []
    
    split_path = os.path.join(dataset_path, split)
    
    # Get class folders
    if os.path.exists(split_path):
        class_folders = sorted([d for d in os.listdir(split_path) 
                               if os.path.isdir(os.path.join(split_path, d))])
    else:
        # Fallback: classes directly in dataset_path
        split_path = dataset_path
        class_folders = sorted([d for d in os.listdir(split_path) 
                               if os.path.isdir(os.path.join(split_path, d)) 
                               and d not in ['train', 'val', 'test']])
    
    print(f"\n  Loading {split} data from: {split_path}")
    print(f"  Classes found: {class_folders}")
    print(f"  Samples per class: {samples_per_class}")
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(split_path, class_name)
        
        # Get all image files
        all_files = []
        for f in os.listdir(class_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                all_files.append(os.path.join(class_path, f))
        
        # Randomly sample
        sample_size = min(samples_per_class, len(all_files))
        sampled_files = random.sample(all_files, sample_size)
        
        print(f"    {class_name}: {sample_size} images (from {len(all_files)} total)")
        
        for img_path in sampled_files:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                labels.append(class_idx)
    
    return images, labels


def quick_train(
    samples_per_class: int = 500,
    n_estimators: int = 100,
    max_depth: int = 20,
    verbose: bool = True
):
    """
    Quick training with sampled data.
    
    Args:
        samples_per_class: Number of images to sample from each class
        n_estimators: Number of trees in Random Forest
        max_depth: Maximum depth of trees
        verbose: Whether to print progress
    """
    print("=" * 60)
    print("  RF-DETR Quick Training (Sampled Data)")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Samples per class: {samples_per_class}")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # ================================================================
    # Step 1: Load Sampled Data
    # ================================================================
    print("\n" + "=" * 60)
    print("  Step 1: Loading Sampled Data")
    print("=" * 60)
    
    train_images, train_labels = load_sample_data(
        DATASET_DIR, "train", samples_per_class
    )
    
    # Use smaller validation set
    val_samples = max(100, samples_per_class // 5)
    val_images, val_labels = load_sample_data(
        DATASET_DIR, "val", val_samples
    )
    
    print(f"\n  Total training samples: {len(train_images)}")
    print(f"  Total validation samples: {len(val_images)}")
    
    # ================================================================
    # Step 2: Extract Features
    # ================================================================
    print("\n" + "=" * 60)
    print("  Step 2: Extracting DETR Features")
    print("=" * 60)
    
    extractor = DETRFeatureExtractor()
    
    print(f"\n  Extracting features for {len(train_images)} training images...")
    X_train = extractor.extract_batch_features(train_images, verbose=verbose)
    y_train = np.array(train_labels)
    
    print(f"\n  Extracting features for {len(val_images)} validation images...")
    X_val = extractor.extract_batch_features(val_images, verbose=verbose)
    y_val = np.array(val_labels)
    
    print(f"\n  Feature matrix shape: {X_train.shape}")
    
    # ================================================================
    # Step 3: Train Random Forest
    # ================================================================
    print("\n" + "=" * 60)
    print("  Step 3: Training Random Forest Classifier")
    print("=" * 60)
    
    print(f"\n  Hyperparameters:")
    print(f"    - n_estimators: {n_estimators}")
    print(f"    - max_depth: {max_depth}")
    print(f"    - min_samples_split: 2")
    print(f"    - n_jobs: -1 (all cores)")
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\n  Training...")
    rf.fit(X_train, y_train)
    print("  Training complete!")
    
    # ================================================================
    # Step 4: Evaluate
    # ================================================================
    print("\n" + "=" * 60)
    print("  Step 4: Evaluation")
    print("=" * 60)
    
    # Training accuracy
    train_pred = rf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\n  Training Accuracy: {train_acc:.4f}")
    
    # Validation accuracy
    val_pred = rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"  Validation Accuracy: {val_acc:.4f}")
    
    # Classification report
    print(f"\n  Classification Report (Validation):")
    print("-" * 50)
    print(classification_report(y_val, val_pred, target_names=CLASS_NAMES))
    
    # Confusion matrix
    print(f"\n  Confusion Matrix (Validation):")
    print("-" * 50)
    cm = confusion_matrix(y_val, val_pred)
    print(f"  {cm}")
    
    # ================================================================
    # Step 5: Save Model
    # ================================================================
    print("\n" + "=" * 60)
    print("  Step 5: Saving Model")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "rf_model.joblib")
    joblib.dump(rf, model_path)
    print(f"\n  Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        "training_samples": len(train_images),
        "validation_samples": len(val_images),
        "samples_per_class": samples_per_class,
        "training_accuracy": float(train_acc),
        "validation_accuracy": float(val_acc),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "classes": CLASS_NAMES,
        "feature_dim": X_train.shape[1],
        "timestamp": datetime.now().isoformat(),
        "mode": "quick_training"
    }
    
    import json
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {metadata_path}")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"""
  Summary:
  --------
  • Training samples: {len(train_images)}
  • Validation samples: {len(val_images)}
  • Training accuracy: {train_acc:.4f}
  • Validation accuracy: {val_acc:.4f}
  • Model saved: {model_path}
  
  Next Steps:
  -----------
  1. Start the API:
     python -m app.main
     
  2. Open the web interface:
     http://localhost:8000
     
  3. Test with the camera or upload an image!
  
  Note: This model was trained on a SAMPLE of the data.
  For production, run the full training:
     python -m app.train
""")
    
    return rf


if __name__ == "__main__":
    # Get sample size from environment or use default
    sample_size = int(os.environ.get("SAMPLE_SIZE", "500"))
    
    quick_train(
        samples_per_class=sample_size,
        n_estimators=100,
        max_depth=20
    )
