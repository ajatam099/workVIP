"""Scratch detection using edge detection and morphology."""


import cv2
import numpy as np

from .base import BaseDetector, Detection
from ..config_loader import get_config_loader


class ScratchDetector(BaseDetector):
    """Detects scratches using Canny edge detection and morphology."""

    def __init__(self, name: str = "scratches"):
        """
        Initialize scratch detector.

        Args:
            name: Detector name
        """
        super().__init__(name)
        
        # Load parameters from YAML configuration
        config = get_config_loader()
        detector_params = config.get_detector_params('scratches')
        global_params = config.get_global_params()
        morphology_params = config.get_morphology_params()
        
        # Set detector-specific parameters
        self.canny_low = detector_params.get('canny_low', 50)
        self.canny_high = detector_params.get('canny_high', 150)
        self.min_length = detector_params.get('min_length', 20)
        self.min_width = detector_params.get('min_width', 2)
        self.min_area = detector_params.get('min_area', 50)
        self.min_aspect_ratio = detector_params.get('min_aspect_ratio', 2.0)
        self.score_normalization = detector_params.get('score_normalization', 10.0)
        
        # Set global parameters
        self.gaussian_blur_size = global_params.get('gaussian_blur_size', 5)
        
        # Set morphological parameters
        self.kernel_size = morphology_params.get('scratch_kernel_size', 3)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect scratches in the input image.

        Args:
            image: BGR input image

        Returns:
            List of scratch detections
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_size, self.gaussian_blur_size), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            edges, 8, cv2.CV_32S
        )

        detections = []

        # Process each connected component
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]

            # Filter by size and aspect ratio
            if area < self.min_area:  # Minimum area
                continue

            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio < self.min_aspect_ratio:  # Must be elongated
                continue

            # Check minimum dimensions
            if max(w, h) < self.min_length or min(w, h) < self.min_width:
                continue

            # Calculate score based on elongation and contrast
            score = min(1.0, aspect_ratio / self.score_normalization)  # Normalize score

            # Create detection
            detection = Detection(label="scratch", score=score, bbox=(x, y, w, h))
            detections.append(detection)

        return detections
