"""Flash detection using gradient analysis and boundary proximity."""

import cv2
import numpy as np
from typing import List

from .base import BaseDetector, Detection


class FlashDetector(BaseDetector):
    """Detects flash (excess material) using gradient and boundary analysis."""
    
    def __init__(self, name: str = "flash", 
                 brightness_threshold: int = 200, 
                 gradient_threshold: int = 30,
                 min_area: int = 150):
        """
        Initialize flash detector.
        
        Args:
            name: Detector name
            brightness_threshold: Threshold for bright regions
            gradient_threshold: Threshold for gradient detection
            min_area: Minimum flash area in pixels
        """
        super().__init__(name)
        self.brightness_threshold = brightness_threshold
        self.gradient_threshold = gradient_threshold
        self.min_area = min_area
    
    def detect(self, image: np.ndarray) -> List[Detection]:
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
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = np.uint8(gradient_magnitude * 255 / gradient_magnitude.max())
        
        # Find high gradient regions (edges)
        _, gradient_mask = cv2.threshold(gradient_magnitude, self.gradient_threshold, 255, cv2.THRESH_BINARY)
        
        # Combine bright and gradient masks
        flash_mask = cv2.bitwise_and(bright_mask, gradient_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
            area_score = min(1.0, area / 5000.0)
            
            # Calculate brightness score
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_brightness = np.mean(gray[mask > 0])
            brightness_score = min(1.0, mean_brightness / 255.0)
            
            # Combined score
            score = (area_score + brightness_score) / 2.0
            
            # Create detection
            detection = Detection(
                label="flash",
                score=score,
                bbox=(x, y, w, h)
            )
            detections.append(detection)
        
        return detections
