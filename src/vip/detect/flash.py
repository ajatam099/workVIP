"""Flash detection using gradient analysis and boundary proximity."""


import cv2
import numpy as np

from .base import BaseDetector, Detection
from ..config_loader import get_config_loader


class FlashDetector(BaseDetector):
    """Detects flash (excess material) using gradient and boundary analysis."""

    def __init__(self, name: str = "flash"):
        """
        Initialize flash detector.

        Args:
            name: Detector name
        """
        super().__init__(name)
        
        # Load parameters from YAML configuration
        config = get_config_loader()
        detector_params = config.get_detector_params('flash')
        morphology_params = config.get_morphology_params()
        
        # Set detector-specific parameters
        self.brightness_threshold = detector_params.get('brightness_threshold', 200)
        self.gradient_threshold = detector_params.get('gradient_threshold', 30)
        self.min_area = detector_params.get('min_area', 150)
        self.sobel_kernel = detector_params.get('sobel_kernel', 3)
        self.area_normalization = detector_params.get('area_normalization', 5000.0)
        self.brightness_normalization = detector_params.get('brightness_normalization', 255.0)
        
        # Set morphological parameters
        self.kernel_size = morphology_params.get('flash_kernel_size', 5)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect flash in the input image.

        Args:
            image: BGR input image

        Returns:
            List of flash detections
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find bright regions (potential flash)
        _, bright_mask = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)

        # Calculate gradient magnitude
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Avoid division by zero
        max_gradient = gradient_magnitude.max()
        if max_gradient > 0:
            gradient_magnitude = np.uint8(gradient_magnitude * 255 / max_gradient)
        else:
            gradient_magnitude = np.zeros_like(gradient_magnitude, dtype=np.uint8)

        # Find high gradient regions (edges)
        _, gradient_mask = cv2.threshold(
            gradient_magnitude, self.gradient_threshold, 255, cv2.THRESH_BINARY
        )

        # Combine bright and gradient masks
        flash_mask = cv2.bitwise_and(bright_mask, gradient_mask)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        flash_mask = cv2.morphologyEx(flash_mask, cv2.MORPH_OPEN, kernel)
        flash_mask = cv2.morphologyEx(flash_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(flash_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate score based on area and brightness
            area_score = min(1.0, area / self.area_normalization)

            # Calculate brightness score
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_brightness = np.mean(gray[mask > 0])
            brightness_score = min(1.0, mean_brightness / self.brightness_normalization)

            # Combined score
            score = (area_score + brightness_score) / 2.0

            # Create detection
            detection = Detection(label="flash", score=score, bbox=(x, y, w, h))
            detections.append(detection)

        return detections
