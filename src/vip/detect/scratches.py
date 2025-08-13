"""Scratch detection using edge detection and morphology."""

import cv2
import numpy as np
from typing import List

from .base import BaseDetector, Detection


class ScratchDetector(BaseDetector):
    """Detects scratches using Canny edge detection and morphology."""
    
    def __init__(self, name: str = "scratches", 
                 canny_low: int = 50, canny_high: int = 150,
                 min_length: int = 20, min_width: int = 2):
        """
        Initialize scratch detector.
        
        Args:
            name: Detector name
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            min_length: Minimum scratch length in pixels
            min_width: Minimum scratch width in pixels
        """
        super().__init__(name)
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_length = min_length
        self.min_width = min_width
    
    def detect(self, image: np.ndarray) -> List[Detection]:
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
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, 8, cv2.CV_32S)
        
        detections = []
        
        # Process each connected component
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            
            # Filter by size and aspect ratio
            if area < 50:  # Minimum area
                continue
                
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio < 2.0:  # Must be elongated
                continue
            
            # Check minimum dimensions
            if max(w, h) < self.min_length or min(w, h) < self.min_width:
                continue
            
            # Calculate score based on elongation and contrast
            score = min(1.0, aspect_ratio / 10.0)  # Normalize score
            
            # Create detection
            detection = Detection(
                label="scratch",
                score=score,
                bbox=(x, y, w, h)
            )
            detections.append(detection)
        
        return detections
