"""Discoloration detection using LAB color space analysis."""

import cv2
import numpy as np
from typing import List

from .base import BaseDetector, Detection


class DiscolorationDetector(BaseDetector):
    """Detects discoloration using LAB color space analysis."""
    
    def __init__(self, name: str = "discoloration", 
                 window_size: int = 21, threshold: float = 15.0,
                 min_area: int = 200):
        """
        Initialize discoloration detector.
        
        Args:
            name: Detector name
            window_size: Size of local window for mean calculation
            threshold: Color difference threshold for detection
            min_area: Minimum discoloration area in pixels
        """
        super().__init__(name)
        self.window_size = window_size
        self.threshold = threshold
        self.min_area = min_area
    
    def detect(self, image: np.ndarray) -> List[Detection]:
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
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
            area_score = min(1.0, area / 10000.0)
            
            # Calculate color difference score
            mask = np.zeros_like(l_channel)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color_diff = np.mean(color_diff[mask > 0])
            color_score = min(1.0, mean_color_diff / 50.0)
            
            # Combined score
            score = (area_score + color_score) / 2.0
            
            # Create detection
            detection = Detection(
                label="discoloration",
                score=score,
                bbox=(x, y, w, h)
            )
            detections.append(detection)
        
        return detections
