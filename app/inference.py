"""
Inference Service
=================

This module orchestrates the complete defect detection pipeline, integrating:
1. Image Preprocessing (adversarial defense + environment normalization)
2. Feature Extraction (DETR-based deep features)
3. Classification (Random Forest)
4. Result Interpretation (confidence analysis + anomaly detection)

Pipeline Flow:
--------------
    Input Image
         │
         ▼
    ┌─────────────────────────────────────┐
    │     Image Preprocessing             │
    │  - JPEG compression defense         │
    │  - Gaussian smoothing               │
    │  - Bilateral filtering              │
    │  - CLAHE normalization              │
    │  - Color normalization              │
    └────────────────┬────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────┐
    │     Feature Extraction (DETR)       │
    │  - ResNet-50 backbone               │
    │  - Transformer encoder/decoder      │
    │  - 256-dim feature embedding        │
    │  - Object identification            │
    └────────────────┬────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────┐
    │     Classification (Random Forest)  │
    │  - 100 decision trees               │
    │  - Ensemble voting                  │
    │  - Probability estimation           │
    └────────────────┬────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────┐
    │     Result Interpretation           │
    │  - Confidence analysis              │
    │  - Anomaly detection                │
    │  - Defect severity assessment       │
    │  - Human-readable explanations      │
    └────────────────┬────────────────────┘
                     │
                     ▼
              Output (JSON)

Author: RF-DETR Defect Detection System
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, Optional

from app import config
from app.feature_extraction import DETRFeatureExtractor
from app.preprocessing import ImagePreprocessor, get_preprocessor
from app.result_interpretation import ResultInterpreter, get_interpreter, InterpretationResult


class InferenceService:
    """
    Main inference service for defect detection.
    
    This service integrates all microservices into a unified pipeline:
    1. Preprocessing: Robustness against adversarial attacks and lighting changes
    2. Feature Extraction: DETR-based deep features
    3. Classification: Random Forest classifier
    4. Interpretation: Comprehensive result analysis
    
    Attributes:
        preprocessor: Image preprocessing service
        extractor: DETR feature extractor
        interpreter: Result interpretation service
        model: Random Forest classifier (loaded from disk)
    """
    
    def __init__(
        self,
        enable_preprocessing: bool = True,
        enable_adversarial_defense: bool = True,
        enable_lighting_normalization: bool = True,
        confidence_threshold: float = 0.6,
        expected_objects: Optional[list] = None
    ):
        """
        Initialize the inference service.
        
        Args:
            enable_preprocessing: Enable the preprocessing pipeline
            enable_adversarial_defense: Enable adversarial attack defenses
            enable_lighting_normalization: Enable lighting/environment normalization
            confidence_threshold: Minimum confidence for reliable predictions
            expected_objects: List of expected object types for anomaly detection
        """
        print("Initializing Inference Service...")
        
        # Initialize preprocessor
        self.enable_preprocessing = enable_preprocessing
        if enable_preprocessing:
            self.preprocessor = ImagePreprocessor(
                enable_adversarial_defense=enable_adversarial_defense,
                enable_lighting_normalization=enable_lighting_normalization
            )
            print("  ✓ Preprocessor initialized")
        else:
            self.preprocessor = None
        
        # Initialize feature extractor
        self.extractor = DETRFeatureExtractor()
        print("  ✓ Feature extractor initialized")
        
        # Initialize result interpreter
        self.interpreter = ResultInterpreter(
            class_names=config.CLASS_NAMES,
            confidence_threshold=confidence_threshold,
            expected_objects=expected_objects or []
        )
        print("  ✓ Result interpreter initialized")
        
        # Load RF model
        self.model = None
        self.load_model()
        
        print("Inference Service ready!")
    
    def load_model(self) -> bool:
        """
        Load the trained Random Forest model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        model_path = os.path.join(config.MODELS_DIR, "rf_model.joblib")
        
        if os.path.exists(model_path):
            print(f"  Loading RF Model from {model_path}")
            self.model = joblib.load(model_path)
            print(f"  ✓ RF Model loaded ({self.model.n_estimators} trees)")
            return True
        else:
            print("Warning: RF Model not found. Please run train.py first.")
            return False
    
    def predict(
        self,
        image_input: np.ndarray,
        return_detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete defect detection pipeline.
        
        Pipeline Steps:
        1. Preprocess image (defensive + normalization)
        2. Extract features using DETR
        3. Classify using Random Forest
        4. Interpret results
        
        Args:
            image_input: Input image as numpy array (BGR format from OpenCV)
            return_detailed: If True, return full interpretation details
            
        Returns:
            Dictionary containing prediction results
        """
        result = {}
        quality_metrics = None
        
        # ============================================================
        # STEP 1: Image Preprocessing
        # ============================================================
        if self.enable_preprocessing and self.preprocessor is not None:
            processed_image, quality_metrics = self.preprocessor.preprocess(
                image_input,
                return_quality_metrics=True
            )
        else:
            processed_image = image_input
        
        # ============================================================
        # STEP 2: Feature Extraction (DETR)
        # ============================================================
        embedding, object_name, object_confidence = self.extractor.extract(processed_image)
        
        # ============================================================
        # STEP 3: Classification (Random Forest)
        # ============================================================
        if self.model is None:
            return {
                "error": "Model not loaded. Please train the model first.",
                "prediction": "Unknown",
                "object_name": object_name,
                "status": "Error"
            }
        
        # Reshape for single sample prediction
        features = embedding.reshape(1, -1)
        
        # Get class probabilities from Random Forest
        # Each tree votes, and probabilities are the fraction of trees voting for each class
        probabilities = self.model.predict_proba(features)[0]
        prediction_idx = np.argmax(probabilities)
        
        # ============================================================
        # STEP 4: Result Interpretation
        # ============================================================
        interpretation = self.interpreter.interpret(
            prediction_idx=prediction_idx,
            probabilities=probabilities,
            object_name=object_name,
            object_confidence=object_confidence,
            image_quality_metrics=quality_metrics
        )
        
        # ============================================================
        # Format Output
        # ============================================================
        if return_detailed:
            result = interpretation.to_dict()
            result["status"] = "Success"
            
            # Add display message for backward compatibility
            result["display_message"] = (
                f"Prediction: {interpretation.prediction_label} | "
                f"Object: {object_name.replace('_', ' ')} | "
                f"Confidence: {interpretation.confidence:.1%}"
            )
        else:
            # Simplified output
            result = {
                "prediction": interpretation.prediction,
                "prediction_label": interpretation.prediction_label,
                "object_name": object_name.replace("_", " "),
                "confidence": round(interpretation.confidence, 4),
                "is_anomaly": interpretation.is_anomaly,
                "status": "Success"
            }
        
        return result
    
    def predict_batch(
        self,
        images: list,
        return_detailed: bool = False
    ) -> list:
        """
        Run predictions on multiple images.
        
        Args:
            images: List of images (numpy arrays)
            return_detailed: Return detailed interpretation for each
            
        Returns:
            List of prediction results
        """
        results = []
        total = len(images)
        
        for idx, img in enumerate(images):
            if (idx + 1) % 10 == 0:
                print(f"  Processing {idx + 1}/{total}...")
            
            try:
                result = self.predict(img, return_detailed=return_detailed)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "prediction": "Error",
                    "status": "Error"
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "preprocessor_enabled": self.enable_preprocessing,
            "feature_extractor": "DETR (facebook/detr-resnet-50)",
            "feature_dimension": 256,
            "class_names": config.CLASS_NAMES,
            "model_loaded": self.model is not None
        }
        
        if self.model is not None:
            info["classifier"] = {
                "type": "Random Forest",
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "n_features": self.model.n_features_in_
            }
        
        return info


# ============================================================
# Global Instance Management
# ============================================================

_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """
    Get or create the global inference service instance.
    
    This ensures only one instance is created, saving memory
    and initialization time.
    
    Returns:
        InferenceService instance
    """
    global _service
    if _service is None:
        _service = InferenceService()
    return _service


def reset_inference_service():
    """Reset the global inference service (useful for testing)."""
    global _service
    _service = None
