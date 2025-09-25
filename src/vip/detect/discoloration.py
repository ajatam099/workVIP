"""Discoloration detection using LAB color space analysis."""


import cv2
import numpy as np

from .base import BaseDetector, Detection
from ..config_loader import get_config_loader


class DiscolorationDetector(BaseDetector):
    """Detects discoloration using LAB color space analysis."""

    def __init__(self, name: str = "discoloration"):
        """
        Initialize discoloration detector.

        Args:
            name: Detector name
        """
        super().__init__(name)
        
        # Load parameters from YAML configuration
        config = get_config_loader()
        detector_params = config.get_detector_params('discoloration')
        morphology_params = config.get_morphology_params()
        
        # Set detector-specific parameters
        self.window_size = detector_params.get('window_size', 21)
        self.threshold = detector_params.get('threshold', 15.0)
        self.min_area = detector_params.get('min_area', 200)
        self.area_normalization = detector_params.get('area_normalization', 10000.0)
        self.color_normalization = detector_params.get('color_normalization', 50.0)
        
        # Set morphological parameters
        self.kernel_size = morphology_params.get('discoloration_kernel_size', 7)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect discoloration in the input image.

        Args:
            image: BGR input image

        Returns:
            List of discoloration detections
        """
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split LAB channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Calculate local means using Gaussian blur
        l_mean = cv2.GaussianBlur(l_channel, (self.window_size, self.window_size), 0)
        a_mean = cv2.GaussianBlur(a_channel, (self.window_size, self.window_size), 0)
        b_mean = cv2.GaussianBlur(b_channel, (self.window_size, self.window_size), 0)

        # Calculate color differences (ΔE approximation)
        l_diff = np.abs(l_channel.astype(np.float32) - l_mean.astype(np.float32))
        a_diff = np.abs(a_channel.astype(np.float32) - a_mean.astype(np.float32))
        b_diff = np.abs(b_channel.astype(np.float32) - b_mean.astype(np.float32))

        # Combined color difference (simplified ΔE)
        color_diff = np.sqrt(l_diff**2 + a_diff**2 + b_diff**2)

        # Apply threshold to find discolored regions
        discolored = (color_diff > self.threshold).astype(np.uint8) * 255

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        discolored = cv2.morphologyEx(discolored, cv2.MORPH_OPEN, kernel)
        discolored = cv2.morphologyEx(discolored, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(discolored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate score based on area and color difference
            area_score = min(1.0, area / self.area_normalization)

            # Calculate color difference score
            mask = np.zeros_like(l_channel)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color_diff = np.mean(color_diff[mask > 0])
            color_score = min(1.0, mean_color_diff / self.color_normalization)

            # Combined score
            score = (area_score + color_score) / 2.0

            # Create detection
            detection = Detection(label="discoloration", score=score, bbox=(x, y, w, h))
            detections.append(detection)

        return detections
