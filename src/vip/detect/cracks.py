"""Crack detection using ridge/edge emphasis and morphological analysis."""


import cv2
import numpy as np

from .base import BaseDetector, Detection
from ..config_loader import get_config_loader


class CrackDetector(BaseDetector):
    """Detects cracks using ridge/edge emphasis and morphological analysis."""

    def __init__(self, name: str = "cracks"):
        """
        Initialize crack detector.

        Args:
            name: Detector name
        """
        super().__init__(name)
        
        # Load parameters from YAML configuration
        config = get_config_loader()
        detector_params = config.get_detector_params('cracks')
        global_params = config.get_global_params()
        morphology_params = config.get_morphology_params()
        
        # Set detector-specific parameters
        self.sobel_kernel = detector_params.get('sobel_kernel', 3)
        self.threshold = detector_params.get('threshold', 50)
        self.min_length = detector_params.get('min_length', 15)
        self.min_width = detector_params.get('min_width', 1)
        self.min_area = detector_params.get('min_area', 20)
        self.min_aspect_ratio = detector_params.get('min_aspect_ratio', 1.5)
        self.elongation_normalization = detector_params.get('elongation_normalization', 8.0)
        self.gradient_normalization = detector_params.get('gradient_normalization', 100.0)
        
        # Set global parameters
        self.gaussian_blur_size = global_params.get('gaussian_blur_size', 5)
        
        # Set morphological parameters
        self.horizontal_kernel = morphology_params.get('crack_horizontal_kernel', [1, 3])
        self.vertical_kernel = morphology_params.get('crack_vertical_kernel', [3, 1])
        self.close_kernel_size = morphology_params.get('crack_close_kernel_size', 3)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect cracks in the input image.

        Args:
            image: BGR input image

        Returns:
            List of crack detections
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_size, self.gaussian_blur_size), 0)

        # Sobel edge detection in X and Y directions
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Avoid division by zero
        max_gradient = gradient_magnitude.max()
        if max_gradient > 0:
            gradient_magnitude = np.uint8(gradient_magnitude * 255 / max_gradient)
        else:
            gradient_magnitude = np.zeros_like(gradient_magnitude, dtype=np.uint8)

        # Apply threshold to find potential crack regions
        _, thresh = cv2.threshold(gradient_magnitude, self.threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to enhance crack-like structures
        # Use thin kernels to preserve thin lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(self.horizontal_kernel))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(self.vertical_kernel))

        # Apply horizontal and vertical morphological operations
        thresh_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        thresh_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

        # Combine horizontal and vertical detections
        thresh_combined = cv2.bitwise_or(thresh_h, thresh_v)

        # Clean up with closing operation
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel_size, self.close_kernel_size))
        thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel_close)

        # Find contours
        contours, _ = cv2.findContours(thresh_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by minimum area
            if area < self.min_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by dimensions (cracks should be thin and elongated)
            if max(w, h) < self.min_length or min(w, h) < self.min_width:
                continue

            # Calculate aspect ratio (cracks should be elongated)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio < self.min_aspect_ratio:  # Must be somewhat elongated
                continue

            # Calculate score based on elongation and gradient strength
            elongation_score = min(1.0, aspect_ratio / self.elongation_normalization)

            # Calculate gradient strength score
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_gradient = np.mean(gradient_magnitude[mask > 0])
            gradient_score = min(1.0, mean_gradient / self.gradient_normalization)

            # Combined score
            score = (elongation_score + gradient_score) / 2.0

            # Create detection
            detection = Detection(label="crack", score=score, bbox=(x, y, w, h))
            detections.append(detection)

        return detections
