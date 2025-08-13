"""Contamination detection using high-pass filtering and blob analysis."""

import cv2
import numpy as np
from typing import List

from .base import BaseDetector, Detection


class ContaminationDetector(BaseDetector):
    """Detects contamination using high-pass filtering and blob detection."""
    
    def __init__(self, name: str = "contamination", 
                 blur_size: int = 15, threshold: int = 30,
                 min_area: int = 100, max_area: int = 10000):
        """
        Initialize contamination detector.
        
        Args:
            name: Detector name
            blur_size: Size of Gaussian blur for high-pass filter
            threshold: Threshold for contamination detection
            min_area: Minimum contamination area in pixels
            max_area: Maximum contamination area in pixels
        """
        super().__init__(name)
        self.blur_size = blur_size
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
    
    def detect(self, image: np.ndarray) -> List[Detection]:
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
            area_score = min(1.0, area / 5000.0)
            
            # Calculate contrast score from high-pass response
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_response = np.mean(high_pass[mask > 0])
            contrast_score = min(1.0, mean_response / 100.0)
            
            # Combined score
            score = (area_score + contrast_score) / 2.0
            
            # Create detection
            detection = Detection(
                label="contamination",
                score=score,
                bbox=(x, y, w, h)
            )
            detections.append(detection)
        
        return detections
