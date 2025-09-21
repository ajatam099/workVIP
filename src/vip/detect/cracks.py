"""Crack detection using ridge/edge emphasis and morphological analysis."""


import cv2
import numpy as np

from .base import BaseDetector, Detection


class CrackDetector(BaseDetector):
    """Detects cracks using ridge/edge emphasis and morphological analysis."""

    def __init__(
        self,
        name: str = "cracks",
        sobel_kernel: int = 3,
        threshold: int = 50,
        min_length: int = 15,
        min_width: int = 1,
    ):
        """
        Initialize crack detector.

        Args:
            name: Detector name
            sobel_kernel: Kernel size for Sobel operators
            threshold: Threshold for crack detection
            min_length: Minimum crack length in pixels
            min_width: Minimum crack width in pixels
        """
        super().__init__(name)
        self.sobel_kernel = sobel_kernel
        self.threshold = threshold
        self.min_length = min_length
        self.min_width = min_width

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
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

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
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

        # Apply horizontal and vertical morphological operations
        thresh_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        thresh_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

        # Combine horizontal and vertical detections
        thresh_combined = cv2.bitwise_or(thresh_h, thresh_v)

        # Clean up with closing operation
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel_close)

        # Find contours
        contours, _ = cv2.findContours(thresh_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by minimum area
            if area < 20:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by dimensions (cracks should be thin and elongated)
            if max(w, h) < self.min_length or min(w, h) < self.min_width:
                continue

            # Calculate aspect ratio (cracks should be elongated)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio < 1.5:  # Must be somewhat elongated
                continue

            # Calculate score based on elongation and gradient strength
            elongation_score = min(1.0, aspect_ratio / 8.0)

            # Calculate gradient strength score
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_gradient = np.mean(gradient_magnitude[mask > 0])
            gradient_score = min(1.0, mean_gradient / 100.0)

            # Combined score
            score = (elongation_score + gradient_score) / 2.0

            # Create detection
            detection = Detection(label="crack", score=score, bbox=(x, y, w, h))
            detections.append(detection)

        return detections
