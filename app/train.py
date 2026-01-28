"""
Model Training Script
======================

This script trains the Random Forest classifier using DETR features
for manufacturing defect detection.

Training Pipeline:
------------------
1. Load training and validation images
2. Extract features using DETR (256-dim embeddings)
3. Train Random Forest classifier
4. Evaluate on validation set
5. Save trained model

Random Forest Mathematics:
--------------------------

Random Forest is an ensemble of Decision Trees using bagging (bootstrap aggregating):

1. Bootstrap Sampling:
   - For each tree t, sample n data points with replacement
   - This creates diverse training sets for each tree
   
2. Feature Subspace Sampling:
   - At each node split, consider only m = √(total_features) features
   - This decorrelates the trees, reducing variance
   
3. Decision Tree Split Criterion (Gini Impurity):
   Gini(S) = 1 - Σ p_i²
   
   Where p_i is the proportion of class i in set S.
   
   Information Gain for split:
   IG(S, A) = Gini(S) - Σ (|S_v|/|S|) * Gini(S_v)
   
4. Ensemble Voting:
   - Classification: Majority vote across all trees
   - Probability: Average of individual tree probabilities
   
   P(y=k|x) = (1/T) * Σ P_t(y=k|x) for t=1 to T

Why Random Forest for Defect Detection:
---------------------------------------
1. Robust to overfitting (ensemble reduces variance)
2. Handles high-dimensional features well (256-dim from DETR)
3. Provides probability estimates for confidence analysis
4. Fast inference time (important for manufacturing lines)
5. No need for GPU during inference

Usage:
------
    python -m app.train
    
    # Or with custom parameters:
    from app.train import train_model
    train_model(n_estimators=200, max_depth=20)

Author: RF-DETR Defect Detection System
"""

import os
import time
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)

from app import config
from app.data_loader import load_data
from app.feature_extraction import DETRFeatureExtractor


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def train_model(
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    verbose: bool = True
) -> dict:
    """
    Train the Random Forest classifier.
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of each tree (None = unlimited)
        min_samples_split: Minimum samples to split a node
        min_samples_leaf: Minimum samples in a leaf node
        random_state: Random seed for reproducibility
        verbose: Print progress updates
        
    Returns:
        Dictionary with training results and metrics
    """
    results = {
        "success": False,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state
        }
    }
    
    if verbose:
        print_separator("RF-DETR Training Pipeline")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Dataset: {config.DATASET_DIR}")
        print(f"Classes: {config.CLASS_NAMES}")
        print()
    
    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    if verbose:
        print_separator("Step 1: Loading Data")
    
    start_time = time.time()
    (train_images, train_labels), (val_images, val_labels) = load_data()
    load_time = time.time() - start_time
    
    if len(train_images) == 0:
        print("ERROR: No training data found. Aborting.")
        results["error"] = "No training data found"
        return results
    
    if verbose:
        print(f"  Training samples: {len(train_images)}")
        print(f"  Validation samples: {len(val_images)}")
        print(f"  Load time: {load_time:.2f}s")
        
        # Class distribution
        unique, counts = np.unique(train_labels, return_counts=True)
        print(f"  Class distribution (train):")
        for idx, count in zip(unique, counts):
            print(f"    - {config.CLASS_NAMES[idx]}: {count}")
    
    results["data"] = {
        "train_samples": len(train_images),
        "val_samples": len(val_images),
        "class_names": config.CLASS_NAMES,
        "load_time_seconds": load_time
    }
    
    # ============================================================
    # STEP 2: Feature Extraction (DETR)
    # ============================================================
    if verbose:
        print()
        print_separator("Step 2: Feature Extraction (DETR)")
    
    extractor = DETRFeatureExtractor()
    
    # Extract training features
    if verbose:
        print(f"  Extracting features for {len(train_images)} training images...")
    
    start_time = time.time()
    X_train = extractor.extract_batch_features(train_images, verbose=verbose)
    train_extract_time = time.time() - start_time
    
    if verbose:
        print(f"  Training features shape: {X_train.shape}")
        print(f"  Extraction time: {train_extract_time:.2f}s")
    
    # Extract validation features
    X_val = None
    val_extract_time = 0
    if len(val_images) > 0:
        if verbose:
            print(f"  Extracting features for {len(val_images)} validation images...")
        
        start_time = time.time()
        X_val = extractor.extract_batch_features(val_images, verbose=verbose)
        val_extract_time = time.time() - start_time
        
        if verbose:
            print(f"  Validation features shape: {X_val.shape}")
            print(f"  Extraction time: {val_extract_time:.2f}s")
    
    results["feature_extraction"] = {
        "feature_dimension": X_train.shape[1],
        "train_extraction_time_seconds": train_extract_time,
        "val_extraction_time_seconds": val_extract_time
    }
    
    # ============================================================
    # STEP 3: Train Random Forest
    # ============================================================
    if verbose:
        print()
        print_separator("Step 3: Training Random Forest Classifier")
        print(f"  Configuration:")
        print(f"    - n_estimators: {n_estimators}")
        print(f"    - max_depth: {max_depth}")
        print(f"    - min_samples_split: {min_samples_split}")
        print(f"    - min_samples_leaf: {min_samples_leaf}")
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        verbose=1 if verbose else 0
    )
    
    start_time = time.time()
    rf.fit(X_train, train_labels)
    train_time = time.time() - start_time
    
    if verbose:
        print(f"  Training time: {train_time:.2f}s")
    
    results["training"] = {
        "training_time_seconds": train_time
    }
    
    # ============================================================
    # STEP 4: Evaluate Model
    # ============================================================
    if verbose:
        print()
        print_separator("Step 4: Model Evaluation")
    
    # Training accuracy
    train_preds = rf.predict(X_train)
    train_accuracy = accuracy_score(train_labels, train_preds)
    
    if verbose:
        print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    results["metrics"] = {
        "train_accuracy": train_accuracy
    }
    
    # Validation metrics
    if X_val is not None and len(val_labels) > 0:
        val_preds = rf.predict(X_val)
        val_accuracy = accuracy_score(val_labels, val_preds)
        
        # Detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='weighted'
        )
        
        conf_matrix = confusion_matrix(val_labels, val_preds)
        
        if verbose:
            print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            print(f"  Precision (weighted): {precision:.4f}")
            print(f"  Recall (weighted): {recall:.4f}")
            print(f"  F1-Score (weighted): {f1:.4f}")
            print()
            print("  Classification Report:")
            print(classification_report(
                val_labels, val_preds,
                target_names=config.CLASS_NAMES,
                digits=4
            ))
            print("  Confusion Matrix:")
            print(f"    {conf_matrix}")
        
        results["metrics"].update({
            "val_accuracy": val_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix.tolist()
        })
    
    # Feature importance (top 10)
    feature_importance = rf.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    
    if verbose:
        print()
        print("  Top 10 Most Important Features (by dimension index):")
        for i, feat_idx in enumerate(top_features, 1):
            print(f"    {i}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
    
    # ============================================================
    # STEP 5: Save Model
    # ============================================================
    if verbose:
        print()
        print_separator("Step 5: Save Model")
    
    model_path = os.path.join(config.MODELS_DIR, "rf_model.joblib")
    joblib.dump(rf, model_path)
    
    # Get model file size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    if verbose:
        print(f"  Model saved to: {model_path}")
        print(f"  Model size: {model_size_mb:.2f} MB")
    
    results["model"] = {
        "path": model_path,
        "size_mb": model_size_mb
    }
    
    # ============================================================
    # Summary
    # ============================================================
    total_time = load_time + train_extract_time + val_extract_time + train_time
    
    if verbose:
        print()
        print_separator("Training Complete!")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Final validation accuracy: {results['metrics'].get('val_accuracy', 'N/A')}")
        print()
        print("  Next steps:")
        print("    1. Run the API: python -m app.main")
        print("    2. Test prediction: POST /predict with an image")
        print("=" * 60)
    
    results["success"] = True
    results["total_time_seconds"] = total_time
    
    return results


if __name__ == "__main__":
    train_model()
