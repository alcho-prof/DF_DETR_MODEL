"""
Image Preprocessing Microservice
================================

This module provides robust image preprocessing with defenses against:
1. Adversarial Attacks - Subtle perturbations designed to cause misclassification
2. Dynamic Environments - Changes in lighting, weather, and object configurations

Mathematical Foundations:
-------------------------

1. JPEG Compression Defense (Against Adversarial Perturbations)
   - Adversarial perturbations are typically high-frequency noise
   - JPEG compression acts as a low-pass filter: F_compressed = DCT^(-1)(Q(DCT(F)))
   - Where DCT = Discrete Cosine Transform, Q = Quantization operator
   - Quality factor q ∈ [1, 100] controls trade-off between defense and image quality

2. Gaussian Blur Defense
   - Smoothing kernel: G(x,y) = (1 / 2πσ²) * exp(-(x² + y²) / 2σ²)
   - Convolution: I_smoothed = I * G
   - Small σ values preserve features while removing high-frequency adversarial noise

3. Adaptive Histogram Equalization (CLAHE) for Lighting Normalization
   - Standard histogram: h(k) = Σ δ(I(x,y) - k) for all pixels
   - CDF: C(k) = Σ h(i) for i ∈ [0, k]
   - Transform: I_eq(x,y) = round((C(I(x,y)) - C_min) / (N - C_min) * (L-1))
   - CLAHE adds clip limit λ to prevent over-amplification

4. Color Normalization (Lab Color Space)
   - Convert RGB → Lab: L* (lightness), a* (green-red), b* (blue-yellow)
   - Normalize L* channel: L_norm = (L - μ_L) / σ_L * σ_target + μ_target
   - Robust to lighting changes as chromaticity is separated from luminance

5. Noise Reduction via Bilateral Filtering
   - f(x) = (1/W_p) Σ G_s(||p-q||) * G_r(|I_p - I_q|) * I_q
   - G_s = spatial Gaussian, G_r = range (intensity) Gaussian
   - Preserves edges while smoothing noise

Author: RF-DETR Defect Detection System
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional, Dict


class ImagePreprocessor:
    """
    Robust Image Preprocessor with defenses against adversarial attacks
    and handling for dynamic environmental conditions.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        jpeg_quality: int = 85,
        gaussian_sigma: float = 0.5,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75,
        bilateral_sigma_space: float = 75,
        enable_adversarial_defense: bool = True,
        enable_lighting_normalization: bool = True
    ):
        """
        Initialize the preprocessor with configurable parameters.
        
        Args:
            target_size: Output image dimensions (width, height)
            jpeg_quality: JPEG compression quality [1-100], lower = more defense
            gaussian_sigma: Gaussian blur sigma, higher = more smoothing
            clahe_clip_limit: Contrast limiting for adaptive histogram equalization
            clahe_tile_grid_size: Grid size for CLAHE
            bilateral_d: Diameter of bilateral filter neighborhood
            bilateral_sigma_color: Filter sigma in color space
            bilateral_sigma_space: Filter sigma in coordinate space
            enable_adversarial_defense: Enable adversarial attack defenses
            enable_lighting_normalization: Enable lighting/environment normalization
        """
        self.target_size = target_size
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.enable_adversarial_defense = enable_adversarial_defense
        self.enable_lighting_normalization = enable_lighting_normalization
        
        # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size
        )
    
    def jpeg_compression_defense(self, image: np.ndarray) -> np.ndarray:
        """
        Apply JPEG compression as a defense against adversarial perturbations.
        
        Theory:
        - Adversarial perturbations are typically small, high-frequency noise
        - JPEG compression uses DCT which attenuates high-frequency components
        - The quantization step removes small perturbations
        
        Mathematical Formulation:
        1. Divide image into 8x8 blocks
        2. Apply 2D DCT: F(u,v) = (1/4) * C(u)C(v) * Σ Σ f(x,y) * cos((2x+1)uπ/16) * cos((2y+1)vπ/16)
        3. Quantize: F_q(u,v) = round(F(u,v) / Q(u,v)) * Q(u,v)
        4. Inverse DCT to reconstruct
        
        Args:
            image: Input BGR image
            
        Returns:
            JPEG-compressed image
        """
        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        
        # Decode back
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        return decoded
    
    def gaussian_smoothing_defense(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to diffuse adversarial perturbations.
        
        Mathematical Formulation:
        - Gaussian kernel: G(x,y) = (1 / 2πσ²) * exp(-(x² + y²) / 2σ²)
        - Filtered image: I'(x,y) = Σ Σ I(x-i, y-j) * G(i, j)
        
        The kernel size is determined by 6σ to capture 99.7% of the distribution.
        
        Args:
            image: Input BGR image
            
        Returns:
            Smoothed image
        """
        # Kernel size must be odd
        ksize = int(np.ceil(self.gaussian_sigma * 6)) | 1
        
        return cv2.GaussianBlur(image, (ksize, ksize), self.gaussian_sigma)
    
    def bilateral_filter_defense(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering for edge-preserving noise reduction.
        
        Mathematical Formulation:
        I'(x) = (1/W_p) * Σ I(x_i) * f_r(||I(x_i) - I(x)||) * g_s(||x_i - x||)
        
        Where:
        - f_r: Range kernel for intensity differences (preserves edges)
        - g_s: Spatial kernel for distance weighting
        - W_p: Normalization factor
        
        This is particularly effective against adversarial noise while
        preserving important edge features for defect detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            Filtered image
        """
        return cv2.bilateralFilter(
            image,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space
        )
    
    def clahe_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE for adaptive contrast enhancement.
        
        Addresses: Variable lighting conditions in manufacturing environments.
        
        Mathematical Formulation:
        1. Divide image into tiles (8x8 default)
        2. For each tile, compute histogram h(k)
        3. Clip histogram: h_clipped(k) = min(h(k), clip_limit)
        4. Redistribute excess: h_redistributed = h_clipped + excess / num_bins
        5. Compute CDF and apply transformation
        
        The clip limit prevents over-amplification of noise in homogeneous regions.
        
        Args:
            image: Input BGR image
            
        Returns:
            Contrast-normalized image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel only (luminance)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self.clahe.apply(l_channel)
        
        # Merge channels back
        lab = cv2.merge([l_channel, a_channel, b_channel])
        
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def color_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image colors to handle varying lighting conditions.
        
        Method: Reinhard Color Transfer adapted for normalization
        
        Mathematical Formulation:
        1. Convert RGB → Lab (perceptually uniform)
        2. For each channel c ∈ {L, a, b}:
           - μ_c = mean(channel_c)
           - σ_c = std(channel_c)
           - channel_c_norm = (channel_c - μ_c) / σ_c * σ_target + μ_target
        3. Convert Lab → RGB
        
        This normalizes image statistics, making the model robust to
        changes in ambient lighting, shadows, and color temperature.
        
        Args:
            image: Input BGR image
            
        Returns:
            Color-normalized image
        """
        # Convert to float32 for precision
        image_float = image.astype(np.float32) / 255.0
        
        # Convert to LAB
        lab = cv2.cvtColor(image_float, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Normalize L channel (most affected by lighting)
        # Target: mean=50 (mid-gray), std=20 (moderate contrast)
        l_mean, l_std = l.mean(), l.std() + 1e-6
        l_norm = (l - l_mean) / l_std * 20 + 50
        l_norm = np.clip(l_norm, 0, 100)
        
        # Normalize a and b channels (chrominance)
        # Target: mean=0 (neutral), std=10 (moderate saturation)
        a_mean, a_std = a.mean(), a.std() + 1e-6
        a_norm = (a - a_mean) / a_std * 10
        a_norm = np.clip(a_norm, -128, 127)
        
        b_mean, b_std = b.mean(), b.std() + 1e-6
        b_norm = (b - b_mean) / b_std * 10
        b_norm = np.clip(b_norm, -128, 127)
        
        # Merge and convert back
        lab_norm = cv2.merge([l_norm.astype(np.float32), 
                              a_norm.astype(np.float32), 
                              b_norm.astype(np.float32)])
        
        bgr_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
        bgr_norm = np.clip(bgr_norm * 255, 0, 255).astype(np.uint8)
        
        return bgr_norm
    
    def resize_and_pad(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio with padding.
        
        This prevents distortion which could affect defect detection accuracy.
        
        Args:
            image: Input BGR image
            
        Returns:
            Resized and padded image
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (black padding)
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def estimate_image_quality(self, image: np.ndarray) -> Dict:
        """
        Estimate image quality metrics for anomaly detection.
        
        Metrics:
        1. Blur Score: Variance of Laplacian (lower = more blur)
        2. Noise Level: Estimated from high-frequency components
        3. Brightness: Mean intensity
        4. Contrast: Standard deviation of intensity
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary of quality metrics
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness (mean luminance)
        brightness = gray.mean()
        
        # Contrast (standard deviation)
        contrast = gray.std()
        
        # Noise estimation using median absolute deviation of gradient
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        noise_estimate = np.median(np.abs(gradient_magnitude - np.median(gradient_magnitude)))
        
        return {
            "blur_score": float(laplacian_var),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "noise_level": float(noise_estimate),
            "is_blurry": laplacian_var < 100,
            "is_underexposed": brightness < 50,
            "is_overexposed": brightness > 200,
            "is_low_contrast": contrast < 30
        }
    
    def preprocess(
        self,
        image: np.ndarray,
        return_quality_metrics: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Full preprocessing pipeline.
        
        Pipeline Order (optimized for defense and normalization):
        1. Quality Assessment
        2. JPEG Compression (adversarial defense)
        3. Gaussian Smoothing (adversarial defense)
        4. Bilateral Filtering (noise reduction)
        5. CLAHE (lighting normalization)
        6. Color Normalization (environment handling)
        7. Resize and Pad
        
        Args:
            image: Input BGR image (numpy array)
            return_quality_metrics: Whether to return quality metrics
            
        Returns:
            Tuple of (processed_image, quality_metrics or None)
        """
        processed = image.copy()
        quality_metrics = None
        
        # 1. Quality Assessment
        if return_quality_metrics:
            quality_metrics = self.estimate_image_quality(processed)
        
        # 2-4. Adversarial Defense Pipeline
        if self.enable_adversarial_defense:
            # JPEG compression removes high-frequency adversarial noise
            processed = self.jpeg_compression_defense(processed)
            
            # Gaussian smoothing diffuses remaining perturbations
            processed = self.gaussian_smoothing_defense(processed)
            
            # Bilateral filtering for edge-preserving denoising
            processed = self.bilateral_filter_defense(processed)
        
        # 5-6. Environment Normalization Pipeline
        if self.enable_lighting_normalization:
            # CLAHE for local contrast enhancement
            processed = self.clahe_normalization(processed)
            
            # Color normalization for consistent appearance
            processed = self.color_normalization(processed)
        
        # 7. Resize to target size
        processed = self.resize_and_pad(processed)
        
        return processed, quality_metrics


# Global preprocessor instance
_preprocessor = None


def get_preprocessor() -> ImagePreprocessor:
    """Get or create the global preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ImagePreprocessor()
    return _preprocessor


def preprocess_image(
    image: np.ndarray,
    return_quality_metrics: bool = False
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Convenience function to preprocess an image.
    
    Args:
        image: Input BGR image
        return_quality_metrics: Whether to return quality metrics
        
    Returns:
        Tuple of (processed_image, quality_metrics or None)
    """
    preprocessor = get_preprocessor()
    return preprocessor.preprocess(image, return_quality_metrics)
