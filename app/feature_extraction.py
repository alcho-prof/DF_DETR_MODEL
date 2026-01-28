"""
Feature Extraction Microservice
================================

This module extracts deep features using DETR (DEtection TRansformer) for 
image classification in manufacturing defect detection.

Architecture Overview:
----------------------

DETR (DEtection TRansformer) Architecture:
    Input Image → CNN Backbone (ResNet-50) → Transformer Encoder → Transformer Decoder → Features

1. CNN Backbone (ResNet-50)
   - Extracts spatial features from input image
   - Output: Feature map of shape (H/32, W/32, 2048)
   - Uses standard residual blocks: F(x) + x

2. Positional Encoding
   - Adds spatial position information to features
   - Sine/Cosine encoding: 
     PE(pos, 2i) = sin(pos / 10000^(2i/d))
     PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

3. Transformer Encoder
   - Multi-Head Self-Attention: Attention(Q, K, V) = softmax(QK^T / √d_k) * V
   - Feed-Forward Network: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
   - Layer Normalization: LN(x) = γ * (x - μ) / σ + β
   - Output: Encoded features (HW, d_model)

4. Transformer Decoder
   - Object Queries: Learned embeddings for 100 potential objects
   - Cross-Attention: Attends to encoder output
   - Self-Attention: Queries attend to each other
   - Output: Object embeddings (100, d_model)

Feature Extraction Strategy:
----------------------------
For classification (not detection), we use the decoder hidden states:
1. Run forward pass through DETR
2. Extract decoder hidden states from the last layer
3. Find the query with highest detection confidence
4. Use this embedding as the feature vector

Why This Works:
- The most confident query represents the "main" object in the image
- Its embedding encodes both object identity and appearance features
- These features are discriminative for defect detection

Mathematical Foundation:
------------------------

Self-Attention Mechanism:
    Q = XW_Q, K = XW_K, V = XW_V
    Attention(Q, K, V) = softmax(QK^T / √d_k) * V

Multi-Head Attention:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W_O
    where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)

The attention mechanism allows the model to:
1. Focus on relevant parts of the image
2. Capture long-range dependencies
3. Be robust to spatial transformations

Author: RF-DETR Defect Detection System
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
from transformers import DetrImageProcessor, DetrForObjectDetection


class DETRFeatureExtractor:
    """
    Feature extractor using DETR (DEtection TRansformer).
    
    This class uses a pre-trained DETR model to:
    1. Detect and identify objects in images
    2. Extract deep feature embeddings for classification
    
    The extracted features are used by a Random Forest classifier
    for defect/non-defect classification.
    
    Attributes:
        model_name: HuggingFace model identifier
        processor: DETR image processor
        model: DETR object detection model
        device: CPU or CUDA device
        feature_dim: Dimension of extracted features (256 for DETR-ResNet-50)
    """
    
    def __init__(self, model_name: str = "facebook/detr-resnet-50"):
        """
        Initialize the DETR feature extractor.
        
        Args:
            model_name: HuggingFace model identifier. Options:
                - "facebook/detr-resnet-50" (default, 41M params)
                - "facebook/detr-resnet-101" (60M params, more accurate)
        
        Model Architecture:
            - Backbone: ResNet-50 or ResNet-101
            - Encoder: 6 transformer layers
            - Decoder: 6 transformer layers
            - Hidden dimension: 256
            - Number of object queries: 100
        """
        print(f"Loading DETR model: {model_name}...")
        
        # Load processor (handles image resizing and normalization)
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        
        # Load model
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        
        # Set to evaluation mode (disable dropout, etc.)
        self.model.eval()
        
        # Select device (CUDA if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Feature dimension (256 for DETR)
        self.feature_dim = 256
        
        print(f"DETR Model loaded on {self.device}.")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Number of classes: {len(self.model.config.id2label)}")
    
    def _convert_to_pil(self, image_input) -> Image.Image:
        """
        Convert various image formats to PIL Image.
        
        Handles:
        - PIL Image: Return as-is
        - NumPy array (BGR from OpenCV): Convert to RGB
        - NumPy array (RGB): Convert directly
        - NumPy array (Grayscale): Convert to RGB
        
        Args:
            image_input: Image in various formats
            
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image_input, Image.Image):
            # Already PIL, ensure RGB
            return image_input.convert('RGB')
        
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                # Grayscale
                return Image.fromarray(image_input).convert('RGB')
            elif image_input.shape[-1] == 3:
                # Assume BGR (OpenCV format) -> RGB
                rgb = image_input[..., ::-1]
                return Image.fromarray(rgb)
            elif image_input.shape[-1] == 4:
                # RGBA or BGRA -> RGB
                rgb = image_input[..., :3]
                rgb = rgb[..., ::-1]  # BGR -> RGB
                return Image.fromarray(rgb)
        
        raise ValueError(f"Unsupported image type: {type(image_input)}")
    
    def extract(
        self, 
        image_input,
        return_all_queries: bool = False
    ) -> Tuple[np.ndarray, str, float]:
        """
        Extract features and identify object from an image.
        
        Process:
        1. Preprocess image using DETR processor
        2. Run forward pass through the model
        3. Extract decoder hidden states (feature embeddings)
        4. Find the most confident object detection
        5. Return the embedding for that query
        
        Mathematical Details:
        - Decoder output shape: (batch_size, num_queries, hidden_dim) = (1, 100, 256)
        - Logits shape: (batch_size, num_queries, num_classes + 1) = (1, 100, 92)
        - We find: argmax over all queries of max class probability
        - The last class (index 91) is "no-object" class
        
        Args:
            image_input: PIL Image or numpy array (BGR/RGB)
            return_all_queries: If True, return embeddings for all 100 queries
            
        Returns:
            Tuple of:
            - embedding: Feature vector (256,) or (100, 256) if return_all_queries
            - object_name: Detected object class name (e.g., "bottle", "car")
            - object_confidence: Confidence score of the detection
        """
        # Convert to PIL
        image = self._convert_to_pil(image_input)
        
        # Preprocess
        # The processor:
        # 1. Resizes image to model's expected size (800 x 1333 max)
        # 2. Normalizes with ImageNet mean/std
        # 3. Converts to tensor
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass (no gradient computation needed)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract hidden states from decoder
        # decoder_hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor has shape: (batch_size, num_queries, hidden_dim)
        # We use the last layer's output
        hidden_states = outputs.decoder_hidden_states[-1]  # (1, 100, 256)
        
        # Get classification logits
        logits = outputs.logits  # (1, 100, num_classes + 1)
        
        # Compute probabilities (exclude "no-object" class)
        # Softmax over classes: P(class | query) = exp(logit) / sum(exp(logits))
        probs = logits.softmax(-1)[0, :, :-1]  # (100, 91)
        
        # Find best detection:
        # For each query, find its max probability and class
        max_scores, class_indices = probs.max(dim=-1)  # Both (100,)
        
        # Find the query with highest confidence
        best_query_idx = max_scores.argmax().item()
        best_confidence = max_scores[best_query_idx].item()
        
        # Get the class name
        detected_class_idx = class_indices[best_query_idx].item()
        object_name = self.model.config.id2label[detected_class_idx]
        
        if return_all_queries:
            # Return all 100 query embeddings
            embeddings = hidden_states[0].cpu().numpy()  # (100, 256)
            return embeddings, object_name, best_confidence
        else:
            # Return only the best query's embedding
            embedding = hidden_states[0, best_query_idx, :].cpu().numpy()  # (256,)
            return embedding, object_name, best_confidence
    
    def extract_batch_features(
        self, 
        images: List,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Extract features for a batch of images (for training).
        
        This method is optimized for extracting features from many images,
        as needed for training the Random Forest classifier.
        
        Args:
            images: List of images (PIL Images or numpy arrays)
            verbose: Print progress updates
            
        Returns:
            Feature matrix of shape (num_images, feature_dim)
        """
        features = []
        total = len(images)
        
        for idx, img in enumerate(images):
            if verbose and (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{total} images...")
            
            try:
                embedding, _, _ = self.extract(img)
                features.append(embedding)
            except Exception as e:
                print(f"  Warning: Error processing image {idx}: {e}")
                # Use zero vector as fallback
                features.append(np.zeros(self.feature_dim))
        
        return np.array(features)
    
    def get_attention_weights(self, image_input) -> dict:
        """
        Extract attention weights for visualization/interpretation.
        
        This can be used to understand which parts of the image
        the model focuses on for its prediction.
        
        Args:
            image_input: PIL Image or numpy array
            
        Returns:
            Dictionary containing:
            - encoder_attention: Self-attention in encoder
            - decoder_self_attention: Self-attention in decoder
            - decoder_cross_attention: Cross-attention (decoder attending to encoder)
        """
        # Convert to PIL
        image = self._convert_to_pil(image_input)
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass with attention outputs
        with torch.no_grad():
            outputs = self.model(
                **inputs, 
                output_attentions=True,
                output_hidden_states=True
            )
        
        # Extract attention weights
        # Shape info:
        # - encoder_attentions: list of (batch, heads, H*W, H*W)
        # - decoder_attentions: list of (batch, heads, 100, 100)
        # - cross_attentions: list of (batch, heads, 100, H*W)
        
        result = {}
        
        if hasattr(outputs, 'encoder_attentions') and outputs.encoder_attentions:
            result['encoder_attention'] = [
                attn.cpu().numpy() for attn in outputs.encoder_attentions
            ]
        
        if hasattr(outputs, 'decoder_attentions') and outputs.decoder_attentions:
            result['decoder_self_attention'] = [
                attn.cpu().numpy() for attn in outputs.decoder_attentions
            ]
        
        if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
            result['decoder_cross_attention'] = [
                attn.cpu().numpy() for attn in outputs.cross_attentions
            ]
        
        return result


# Global extractor instance (lazy loading)
_extractor: Optional[DETRFeatureExtractor] = None


def get_feature_extractor() -> DETRFeatureExtractor:
    """Get or create the global feature extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = DETRFeatureExtractor()
    return _extractor
