"""Main pipeline for orchestrating defect detection."""

import cv2
import numpy as np
from typing import List, Dict, Type

from .config import RunConfig
from .detect.base import BaseDetector, Detection
from .detect.scratches import ScratchDetector
from .detect.contamination import ContaminationDetector
from .detect.discoloration import DiscolorationDetector
from .detect.cracks import CrackDetector
from .detect.flash import FlashDetector
from .utils.viz import overlay_detections


class Pipeline:
    """Main pipeline for defect detection."""
    
    # Registry of available detectors
    DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {
        "scratches": ScratchDetector,
        "contamination": ContaminationDetector,
        "discoloration": DiscolorationDetector,
        "cracks": CrackDetector,
        "flash": FlashDetector,
    }
    
    def __init__(self, config: RunConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.detectors = self.build_detectors(config.defects)
    
    def build_detectors(self, detector_names: List[str]) -> List[BaseDetector]:
        """
        Build detector instances from names.
        
        Args:
            detector_names: List of detector names to instantiate
            
        Returns:
            List of detector instances
            
        Raises:
            ValueError: If detector name is not in registry
        """
        detectors = []
        
        for name in detector_names:
            if name not in self.DETECTOR_REGISTRY:
                raise ValueError(f"Unknown detector: {name}. Available: {list(self.DETECTOR_REGISTRY.keys())}")
            
            detector_class = self.DETECTOR_REGISTRY[name]
            detector = detector_class(name=name)
            detectors.append(detector)
        
        return detectors
    
    def run_on_image(self, image: np.ndarray) -> List[Detection]:
        """
        Run all detectors on a single image.
        
        Args:
            image: BGR input image
            
        Returns:
            List of all detections from all detectors
        """
        all_detections = []
        
        for detector in self.detectors:
            try:
                detections = detector.detect(image)
                all_detections.extend(detections)
            except Exception as e:
                print(f"Warning: Detector {detector.name} failed: {e}")
                continue
        
        return all_detections
    
    def process_image(self, image: np.ndarray, stem: str) -> tuple[List[Detection], np.ndarray]:
        """
        Process a single image through the pipeline.
        
        Args:
            image: BGR input image
            stem: Image filename stem
            
        Returns:
            Tuple of (detections, overlay_image)
        """
        # Resize if specified
        if self.config.resize_width:
            height, width = image.shape[:2]
            new_width = self.config.resize_width
            new_height = int(height * new_width / width)
            image = cv2.resize(image, (new_width, new_height))
        
        # Run detection
        detections = self.run_on_image(image)
        
        # Create overlay
        overlay_image = overlay_detections(image, detections)
        
        return detections, overlay_image
    
    def add_detector(self, name: str, detector_class: Type[BaseDetector]) -> None:
        """
        Add a new detector to the registry.
        
        Args:
            name: Name for the detector
            detector_class: Detector class to register
        """
        if not issubclass(detector_class, BaseDetector):
            raise ValueError("Detector class must inherit from BaseDetector")
        
        self.DETECTOR_REGISTRY[name] = detector_class
        print(f"Added detector '{name}' to registry")
    
    def list_available_detectors(self) -> List[str]:
        """
        List all available detector names.
        
        Returns:
            List of detector names
        """
        return list(self.DETECTOR_REGISTRY.keys())
