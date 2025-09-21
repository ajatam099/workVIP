"""Main pipeline for orchestrating defect detection."""


import cv2
import numpy as np

from .config import RunConfig
from .detect.base import BaseDetector, Detection
from .detect.contamination import ContaminationDetector
from .detect.cracks import CrackDetector
from .detect.discoloration import DiscolorationDetector
from .detect.flash import FlashDetector
from .detect.scratches import ScratchDetector
from .utils.viz import overlay_detections


class Pipeline:
    """Main pipeline for defect detection."""

    # Registry of available detectors
    DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
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

    def build_detectors(self, detector_names: list[str]) -> list[BaseDetector]:
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
                raise ValueError(
                    f"Unknown detector: {name}. Available: {list(self.DETECTOR_REGISTRY.keys())}"
                )

            detector_class = self.DETECTOR_REGISTRY[name]
            detector = detector_class(name=name)
            detectors.append(detector)

        return detectors

    def run_on_image(self, image: np.ndarray) -> list[Detection]:
        """
        Run all detectors on a single image.

        Args:
            image: BGR input image

        Returns:
            List of all detections from all detectors
        """
        all_detections = []

        print(f"Running {len(self.detectors)} detectors: {[d.name for d in self.detectors]}")

        for detector in self.detectors:
            try:
                detections = detector.detect(image)
                print(f"  {detector.name}: {len(detections)} detections")
                all_detections.extend(detections)
            except Exception as e:
                print(f"  ERROR: {detector.name} failed: {e}")
                continue

        return all_detections

    def process_image(self, image: np.ndarray, stem: str) -> tuple[list[Detection], np.ndarray]:
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

    def add_detector(self, name: str, detector_class: type[BaseDetector]) -> None:
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

    def list_available_detectors(self) -> list[str]:
        """
        List all available detector names.

        Returns:
            List of detector names
        """
        return list(self.DETECTOR_REGISTRY.keys())
