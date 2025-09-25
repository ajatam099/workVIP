"""Contamination detection using high-pass filtering and blob analysis."""


import cv2
import numpy as np

from .base import BaseDetector, Detection
from ..config_loader import get_config_loader


class ContaminationDetector(BaseDetector):
    """Detects contamination using high-pass filtering and blob detection."""

    def __init__(self, name: str = "contamination"):
        """
        Initialize contamination detector.

        Args:
            name: Detector name
        """
        super().__init__(name)
        
        # Load parameters from YAML configuration
        config = get_config_loader()
        detector_params = config.get_detector_params('contamination')
        morphology_params = config.get_morphology_params()
        
        # Set detector-specific parameters
        self.blur_size = detector_params.get('blur_size', 15)
        self.threshold = detector_params.get('threshold', 30)
        self.min_area = detector_params.get('min_area', 100)
        self.max_area = detector_params.get('max_area', 10000)
        self.area_normalization = detector_params.get('area_normalization', 5000.0)
        self.contrast_normalization = detector_params.get('contrast_normalization', 100.0)
        
        # Set morphological parameters
        self.kernel_size = morphology_params.get('contamination_kernel_size', 5)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect contamination in the input image.

        Args:
            image: BGR input image

        Returns:
            List of contamination detections
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to create low-pass version
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        # High-pass filter: subtract blurred from original
        high_pass = cv2.subtract(gray, blurred)

        # Apply threshold to find contamination regions
        _, thresh = cv2.threshold(high_pass, self.threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate score based on area and contrast
            # Normalize area score (larger contamination = higher score)
            area_score = min(1.0, area / self.area_normalization)

            # Calculate contrast score from high-pass response
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_response = np.mean(high_pass[mask > 0])
            contrast_score = min(1.0, mean_response / self.contrast_normalization)

            # Combined score
            score = (area_score + contrast_score) / 2.0

            # Create detection
            detection = Detection(label="contamination", score=score, bbox=(x, y, w, h))
            detections.append(detection)

        return detections
