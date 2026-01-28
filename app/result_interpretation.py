"""
Result Interpretation Microservice
===================================

This module provides comprehensive interpretation of classification results,
including confidence analysis, anomaly detection, and actionable insights.

Key Features:
-------------
1. Confidence Score Analysis
2. Decision Boundary Interpretation  
3. Anomaly/Constraint Checking
4. Human-Readable Explanations
5. Manufacturing Quality Metrics

Mathematical Foundations:
-------------------------

1. Confidence Calibration (Platt Scaling)
   - Raw probabilities from RF may not be well-calibrated
   - Platt scaling: P(y=1|f) = 1 / (1 + exp(A*f + B))
   - Parameters A, B learned via logistic regression on held-out data

2. Uncertainty Quantification
   - Entropy: H = -Σ p_i * log(p_i)
   - High entropy → high uncertainty → potential anomaly
   - Normalized entropy: H_norm = H / log(K) where K = number of classes

3. Out-of-Distribution Detection
   - Distance to nearest training example in feature space
   - Mahalanobis distance: D_M = √((x - μ)ᵀ Σ⁻¹ (x - μ))
   - If D_M > threshold → likely OOD sample

4. Decision Confidence Metrics
   - Margin: |P(class_1) - P(class_2)| 
   - Higher margin = more confident decision
   - Low margin indicates ambiguous sample

Author: RF-DETR Defect Detection System
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import json


class DefectSeverity(Enum):
    """Severity levels for detected defects."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    NORMAL = "normal"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_UNCERTAINTY = "high_uncertainty"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    UNEXPECTED_OBJECT = "unexpected_object"
    IMAGE_QUALITY_ISSUE = "image_quality_issue"


@dataclass
class InterpretationResult:
    """Complete interpretation of a classification result."""
    # Core prediction
    prediction: str
    prediction_label: str  # "Defected" or "Non-Defected"
    confidence: float
    
    # Object detection
    object_name: str
    object_confidence: float
    
    # Uncertainty metrics
    entropy: float
    normalized_entropy: float
    prediction_margin: float
    
    # Anomaly detection
    is_anomaly: bool
    anomaly_type: AnomalyType
    anomaly_reason: str
    
    # Defect analysis
    defect_severity: DefectSeverity
    
    # Quality metrics
    image_quality_score: float
    is_reliable_prediction: bool
    
    # Human-readable explanation
    explanation: str
    recommendations: List[str]
    
    # Raw data for debugging
    raw_probabilities: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prediction": self.prediction,
            "prediction_label": self.prediction_label,
            "confidence": round(self.confidence, 4),
            "object_name": self.object_name,
            "object_confidence": round(self.object_confidence, 4),
            "entropy": round(self.entropy, 4),
            "normalized_entropy": round(self.normalized_entropy, 4),
            "prediction_margin": round(self.prediction_margin, 4),
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type.value,
            "anomaly_reason": self.anomaly_reason,
            "defect_severity": self.defect_severity.value,
            "image_quality_score": round(self.image_quality_score, 4),
            "is_reliable_prediction": self.is_reliable_prediction,
            "explanation": self.explanation,
            "recommendations": self.recommendations,
            "raw_probabilities": {k: round(v, 4) for k, v in self.raw_probabilities.items()}
        }


class ResultInterpreter:
    """
    Interprets classification results with comprehensive analysis.
    
    Handles:
    - Confidence calibration and thresholding
    - Uncertainty quantification
    - Anomaly detection
    - Human-readable explanations
    """
    
    def __init__(
        self,
        class_names: List[str] = None,
        confidence_threshold: float = 0.6,
        entropy_threshold: float = 0.7,
        margin_threshold: float = 0.3,
        expected_objects: List[str] = None
    ):
        """
        Initialize the interpreter.
        
        Args:
            class_names: List of class names (e.g., ["defect", "good"])
            confidence_threshold: Minimum confidence for reliable prediction
            entropy_threshold: Maximum normalized entropy for reliable prediction
            margin_threshold: Minimum margin for reliable prediction
            expected_objects: List of expected object types (for anomaly detection)
        """
        self.class_names = class_names or ["defect", "good"]
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.expected_objects = expected_objects or []
    
    def compute_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute Shannon entropy of probability distribution.
        
        Formula: H = -Σ p_i * log(p_i)
        
        Interpretation:
        - H = 0: Maximum certainty (one class has probability 1)
        - H = log(K): Maximum uncertainty (uniform distribution)
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Entropy value
        """
        # Avoid log(0) by adding small epsilon
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def compute_normalized_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute normalized entropy (0 to 1 scale).
        
        Formula: H_norm = H / log(K)
        
        Interpretation:
        - H_norm ≈ 0: Very confident
        - H_norm ≈ 1: Very uncertain (uniform distribution)
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Normalized entropy (0-1)
        """
        entropy = self.compute_entropy(probabilities)
        max_entropy = np.log(len(probabilities))
        
        if max_entropy == 0:
            return 0.0
        
        return float(entropy / max_entropy)
    
    def compute_prediction_margin(self, probabilities: np.ndarray) -> float:
        """
        Compute the margin between top two predictions.
        
        Formula: margin = P(class_1) - P(class_2)
        
        Interpretation:
        - margin ≈ 0: Ambiguous decision
        - margin ≈ 1: Clear decision
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Prediction margin (0-1)
        """
        sorted_probs = np.sort(probabilities)[::-1]
        
        if len(sorted_probs) < 2:
            return 1.0
        
        return float(sorted_probs[0] - sorted_probs[1])
    
    def determine_defect_severity(
        self,
        prediction: str,
        confidence: float
    ) -> DefectSeverity:
        """
        Determine the severity of a detected defect.
        
        Heuristic based on confidence and class:
        - Non-defect → NONE
        - Defect with low confidence → MINOR
        - Defect with medium confidence → MODERATE
        - Defect with high confidence → SEVERE/CRITICAL
        
        Args:
            prediction: Predicted class name
            confidence: Prediction confidence
            
        Returns:
            DefectSeverity enum value
        """
        if prediction.lower() in ["good", "non-defect", "non_defect"]:
            return DefectSeverity.NONE
        
        # Defect detected
        if confidence < 0.5:
            return DefectSeverity.MINOR
        elif confidence < 0.7:
            return DefectSeverity.MODERATE
        elif confidence < 0.9:
            return DefectSeverity.SEVERE
        else:
            return DefectSeverity.CRITICAL
    
    def check_anomalies(
        self,
        confidence: float,
        normalized_entropy: float,
        prediction_margin: float,
        object_name: str,
        image_quality_metrics: Dict = None
    ) -> Tuple[bool, AnomalyType, str]:
        """
        Check for various types of anomalies.
        
        Anomaly Types:
        1. Low Confidence: Model is unsure about prediction
        2. High Uncertainty: Probability distribution is too uniform
        3. Unexpected Object: Detected object not in expected list
        4. Image Quality Issue: Poor image quality affects prediction
        
        Args:
            confidence: Prediction confidence
            normalized_entropy: Normalized entropy of probabilities
            prediction_margin: Margin between top predictions
            object_name: Detected object name
            image_quality_metrics: Optional quality metrics dict
            
        Returns:
            Tuple of (is_anomaly, anomaly_type, reason)
        """
        anomalies = []
        
        # Check confidence
        if confidence < self.confidence_threshold:
            anomalies.append((
                AnomalyType.LOW_CONFIDENCE,
                f"Low confidence ({confidence:.2%}). Model is uncertain about this prediction."
            ))
        
        # Check entropy
        if normalized_entropy > self.entropy_threshold:
            anomalies.append((
                AnomalyType.HIGH_UNCERTAINTY,
                f"High uncertainty (entropy: {normalized_entropy:.2f}). Prediction may be unreliable."
            ))
        
        # Check margin
        if prediction_margin < self.margin_threshold:
            anomalies.append((
                AnomalyType.HIGH_UNCERTAINTY,
                f"Low decision margin ({prediction_margin:.2%}). Prediction is ambiguous."
            ))
        
        # Check expected objects
        if self.expected_objects and object_name.lower() not in [o.lower() for o in self.expected_objects]:
            anomalies.append((
                AnomalyType.UNEXPECTED_OBJECT,
                f"Unexpected object type: '{object_name}'. Expected: {self.expected_objects}"
            ))
        
        # Check image quality
        if image_quality_metrics:
            quality_issues = []
            if image_quality_metrics.get("is_blurry", False):
                quality_issues.append("blurry")
            if image_quality_metrics.get("is_underexposed", False):
                quality_issues.append("underexposed")
            if image_quality_metrics.get("is_overexposed", False):
                quality_issues.append("overexposed")
            if image_quality_metrics.get("is_low_contrast", False):
                quality_issues.append("low contrast")
            
            if quality_issues:
                anomalies.append((
                    AnomalyType.IMAGE_QUALITY_ISSUE,
                    f"Image quality issues detected: {', '.join(quality_issues)}"
                ))
        
        if anomalies:
            # Return the most severe anomaly
            return True, anomalies[0][0], anomalies[0][1]
        
        return False, AnomalyType.NORMAL, "No anomalies detected"
    
    def generate_explanation(
        self,
        prediction: str,
        confidence: float,
        object_name: str,
        defect_severity: DefectSeverity,
        is_anomaly: bool,
        anomaly_reason: str
    ) -> str:
        """
        Generate a human-readable explanation of the prediction.
        
        Args:
            prediction: Predicted class
            confidence: Prediction confidence
            object_name: Detected object
            defect_severity: Severity of defect
            is_anomaly: Whether an anomaly was detected
            anomaly_reason: Reason for anomaly
            
        Returns:
            Human-readable explanation string
        """
        # Base explanation
        if prediction.lower() in ["defect", "defected"]:
            base = f"The system detected a DEFECT in the {object_name}."
            severity_text = {
                DefectSeverity.MINOR: "This appears to be a minor defect that may not significantly affect product quality.",
                DefectSeverity.MODERATE: "This is a moderate defect that should be reviewed by quality control.",
                DefectSeverity.SEVERE: "This is a severe defect. The product should be flagged for immediate inspection.",
                DefectSeverity.CRITICAL: "CRITICAL DEFECT detected with high confidence. Product should be rejected."
            }
            explanation = f"{base} {severity_text.get(defect_severity, '')}"
        else:
            explanation = f"The {object_name} passed quality inspection. No defects detected."
        
        # Add confidence context
        if confidence >= 0.9:
            confidence_text = f"The model is highly confident ({confidence:.1%}) in this prediction."
        elif confidence >= 0.7:
            confidence_text = f"The model is reasonably confident ({confidence:.1%}) in this prediction."
        elif confidence >= 0.5:
            confidence_text = f"The model has moderate confidence ({confidence:.1%}). Consider manual verification."
        else:
            confidence_text = f"The model has low confidence ({confidence:.1%}). Manual inspection is recommended."
        
        explanation = f"{explanation} {confidence_text}"
        
        # Add anomaly warning
        if is_anomaly:
            explanation = f"{explanation} ⚠️ ANOMALY: {anomaly_reason}"
        
        return explanation
    
    def generate_recommendations(
        self,
        prediction: str,
        defect_severity: DefectSeverity,
        is_anomaly: bool,
        anomaly_type: AnomalyType,
        is_reliable: bool
    ) -> List[str]:
        """
        Generate actionable recommendations based on the analysis.
        
        Args:
            prediction: Predicted class
            defect_severity: Severity of defect
            is_anomaly: Whether an anomaly was detected
            anomaly_type: Type of anomaly
            is_reliable: Whether prediction is reliable
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Defect-based recommendations
        if defect_severity == DefectSeverity.CRITICAL:
            recommendations.append("IMMEDIATE ACTION: Remove product from production line")
            recommendations.append("Notify quality control supervisor")
            recommendations.append("Log incident for root cause analysis")
        elif defect_severity == DefectSeverity.SEVERE:
            recommendations.append("Flag product for manual inspection")
            recommendations.append("Consider production line check")
        elif defect_severity == DefectSeverity.MODERATE:
            recommendations.append("Add to review queue for quality control")
        elif defect_severity == DefectSeverity.MINOR:
            recommendations.append("Monitor for recurring patterns")
        
        # Anomaly-based recommendations
        if anomaly_type == AnomalyType.LOW_CONFIDENCE:
            recommendations.append("Retake image with better lighting/angle")
            recommendations.append("Consider manual classification")
        elif anomaly_type == AnomalyType.UNEXPECTED_OBJECT:
            recommendations.append("Verify product is on correct production line")
            recommendations.append("Check camera alignment")
        elif anomaly_type == AnomalyType.IMAGE_QUALITY_ISSUE:
            recommendations.append("Clean camera lens")
            recommendations.append("Adjust lighting conditions")
            recommendations.append("Check camera focus settings")
        
        # Reliability-based recommendations
        if not is_reliable and not recommendations:
            recommendations.append("Consider additional verification due to prediction uncertainty")
        
        if not recommendations:
            recommendations.append("No action required - product passed inspection")
        
        return recommendations
    
    def compute_image_quality_score(self, quality_metrics: Dict = None) -> float:
        """
        Compute overall image quality score (0-1).
        
        Args:
            quality_metrics: Quality metrics from preprocessing
            
        Returns:
            Quality score (0-1, higher is better)
        """
        if not quality_metrics:
            return 1.0
        
        score = 1.0
        
        # Blur affects quality
        blur_score = quality_metrics.get("blur_score", 100)
        if blur_score < 100:
            score *= 0.8  # Significant penalty for blur
        elif blur_score < 500:
            score *= 0.95
        
        # Brightness issues
        brightness = quality_metrics.get("brightness", 128)
        if brightness < 50 or brightness > 200:
            score *= 0.85
        elif brightness < 80 or brightness > 180:
            score *= 0.95
        
        # Contrast issues
        contrast = quality_metrics.get("contrast", 50)
        if contrast < 30:
            score *= 0.9
        
        return max(0.0, min(1.0, score))
    
    def interpret(
        self,
        prediction_idx: int,
        probabilities: np.ndarray,
        object_name: str,
        object_confidence: float = 1.0,
        image_quality_metrics: Dict = None
    ) -> InterpretationResult:
        """
        Fully interpret a classification result.
        
        Args:
            prediction_idx: Index of predicted class
            probabilities: Array of class probabilities
            object_name: Detected object name from DETR
            object_confidence: Confidence of object detection
            image_quality_metrics: Optional quality metrics from preprocessing
            
        Returns:
            InterpretationResult with comprehensive analysis
        """
        # Get prediction info
        prediction = self.class_names[prediction_idx]
        confidence = float(probabilities[prediction_idx])
        
        # Compute uncertainty metrics
        entropy = self.compute_entropy(probabilities)
        normalized_entropy = self.compute_normalized_entropy(probabilities)
        prediction_margin = self.compute_prediction_margin(probabilities)
        
        # Check anomalies
        is_anomaly, anomaly_type, anomaly_reason = self.check_anomalies(
            confidence,
            normalized_entropy,
            prediction_margin,
            object_name,
            image_quality_metrics
        )
        
        # Determine defect severity
        defect_severity = self.determine_defect_severity(prediction, confidence)
        
        # Compute image quality score
        image_quality_score = self.compute_image_quality_score(image_quality_metrics)
        
        # Determine if prediction is reliable
        is_reliable = (
            confidence >= self.confidence_threshold and
            normalized_entropy <= self.entropy_threshold and
            prediction_margin >= self.margin_threshold and
            not is_anomaly
        )
        
        # Generate explanation
        explanation = self.generate_explanation(
            prediction,
            confidence,
            object_name,
            defect_severity,
            is_anomaly,
            anomaly_reason
        )
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            prediction,
            defect_severity,
            is_anomaly,
            anomaly_type,
            is_reliable
        )
        
        # Create prediction label
        if prediction.lower() in ["defect", "defected", "bad"]:
            prediction_label = "Defected"
        else:
            prediction_label = "Non-Defected"
        
        # Build raw probabilities dict
        raw_probabilities = {
            name: float(prob) for name, prob in zip(self.class_names, probabilities)
        }
        
        return InterpretationResult(
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            object_name=object_name,
            object_confidence=object_confidence,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            prediction_margin=prediction_margin,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            anomaly_reason=anomaly_reason,
            defect_severity=defect_severity,
            image_quality_score=image_quality_score,
            is_reliable_prediction=is_reliable,
            explanation=explanation,
            recommendations=recommendations,
            raw_probabilities=raw_probabilities
        )


# Global interpreter instance
_interpreter = None


def get_interpreter(class_names: List[str] = None) -> ResultInterpreter:
    """Get or create the global interpreter instance."""
    global _interpreter
    if _interpreter is None or class_names is not None:
        _interpreter = ResultInterpreter(class_names=class_names)
    return _interpreter
