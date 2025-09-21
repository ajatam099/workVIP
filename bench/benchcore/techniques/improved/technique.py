"""Improved technique combining all enhancements."""

import time
from typing import Any

import cv2
import numpy as np

from bench.benchcore.adapters.pipeline_adapter import convert_detections_to_standard
from bench.benchcore.techniques.base import BaseTechnique
from bench.benchcore.techniques.improved.background_removal import apply_background_removal
from bench.benchcore.techniques.improved.glare_removal import apply_glare_removal
from bench.benchcore.techniques.improved.hog_features import apply_hog_enhancement
from bench.benchcore.techniques.improved.region_growing import apply_region_growing
from src.vip.config import RunConfig
from src.vip.pipeline import Pipeline


class ImprovedTechnique(BaseTechnique):
    """Improved technique with background removal, glare handling, region growing, and HOG features."""

    def __init__(self, name: str = "improved", **params):
        """
        Initialize improved technique.

        Args:
            name: Technique name
            **params: Configuration parameters
        """
        super().__init__(name, **params)
        self.pipeline = None

        # Enhancement flags
        self.use_background_removal = params.get("use_background_removal", True)
        self.use_glare_removal = params.get("use_glare_removal", True)
        self.use_region_growing = params.get("use_region_growing", True)
        self.use_hog_enhancement = params.get("use_hog_enhancement", True)

        # Enhancement parameters
        self.bg_method = params.get("bg_method", "simple")
        self.glare_method = params.get("glare_method", "adaptive")
        self.intensity_threshold = params.get("intensity_threshold", 10.0)
        self.glare_intensity_threshold = params.get("glare_intensity_threshold", 0.8)

    def setup(self, device: str = "cpu") -> None:
        """Initialize the VIP pipeline with improvements."""
        # Create config with all defect types enabled
        defects = self.params.get(
            "defects", ["scratches", "contamination", "discoloration", "cracks", "flash"]
        )

        config = RunConfig(
            input_dir="",  # Not used for benchmarking
            output_dir="",  # Not used for benchmarking
            defects=defects,
            resize_width=self.params.get("resize_width", None),
            save_overlay=False,  # Don't save overlays during benchmarking
            save_json=False,  # Don't save JSON during benchmarking
        )

        self.pipeline = Pipeline(config)

        enhancements = []
        if self.use_background_removal:
            enhancements.append("background_removal")
        if self.use_glare_removal:
            enhancements.append("glare_removal")
        if self.use_region_growing:
            enhancements.append("region_growing")
        if self.use_hog_enhancement:
            enhancements.append("hog_enhancement")

        print(f"✅ Initialized improved technique with {len(defects)} detectors")
        print(f"🔧 Active enhancements: {', '.join(enhancements)}")

    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        """
        Run improved detection on a batch of images.

        Args:
            images: List of BGR images

        Returns:
            List of detection results in standard format
        """
        if self.pipeline is None:
            raise RuntimeError("Technique not initialized. Call setup() first.")

        results = []

        for i, image in enumerate(images):
            image_id = f"image_{i:04d}"

            # Measure processing time
            start_time = time.time()

            # Apply preprocessing enhancements
            processed_image, object_mask = self._preprocess_image(image)

            # Run baseline detection on processed image
            detections = self.pipeline.run_on_image(processed_image)

            # Apply post-processing enhancements
            enhanced_detections = self._postprocess_detections(
                image, processed_image, detections, object_mask
            )

            latency_ms = (time.time() - start_time) * 1000

            # Convert to standard format
            result = convert_detections_to_standard(enhanced_detections, image_id, latency_ms)
            results.append(result)

        return results

    def _preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply preprocessing enhancements.

        Args:
            image: Input BGR image

        Returns:
            Tuple of (processed_image, object_mask)
        """
        processed = image.copy()
        object_mask = None

        # Background removal
        if self.use_background_removal:
            processed, object_mask = apply_background_removal(processed, method=self.bg_method)

        # Glare removal
        if self.use_glare_removal:
            processed, glare_mask = apply_glare_removal(
                processed,
                method=self.glare_method,
                intensity_threshold=self.glare_intensity_threshold,
            )

        return processed, object_mask

    def _postprocess_detections(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray,
        detections: list,
        object_mask: np.ndarray,
    ) -> list:
        """
        Apply post-processing enhancements to detections.

        Args:
            original_image: Original input image
            processed_image: Preprocessed image
            detections: Raw detections from pipeline
            object_mask: Object mask from preprocessing

        Returns:
            Enhanced detections
        """
        enhanced_detections = detections.copy()

        # Region growing enhancement
        if self.use_region_growing and detections:
            enhanced_detections = self._apply_region_growing(
                processed_image, enhanced_detections, object_mask
            )

        # HOG-based confidence enhancement
        if self.use_hog_enhancement and detections:
            enhanced_detections = self._apply_hog_enhancement(original_image, enhanced_detections)

        return enhanced_detections

    def _apply_region_growing(
        self, image: np.ndarray, detections: list, object_mask: np.ndarray
    ) -> list:
        """Apply region growing to expand detection regions."""
        if not detections:
            return detections

        # Convert detections to bounding boxes
        bboxes = []
        for detection in detections:
            if detection.bbox is not None:
                bboxes.append(detection.bbox)

        if not bboxes:
            return detections

        # Apply region growing
        grown_masks = apply_region_growing(image, bboxes, object_mask, self.intensity_threshold)

        # Update detections with grown regions
        enhanced_detections = []
        mask_idx = 0

        for detection in detections:
            enhanced_detection = detection

            if detection.bbox is not None and mask_idx < len(grown_masks):
                grown_mask = grown_masks[mask_idx]

                # Update bounding box to encompass grown region
                contours, _ = cv2.findContours(
                    grown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # Create new detection with updated bbox and mask
                    from src.vip.detect.base import Detection

                    enhanced_detection = Detection(
                        label=detection.label,
                        score=detection.score * 1.1,  # Slightly boost confidence for grown regions
                        bbox=(x, y, w, h),
                        mask=grown_mask,
                    )

                mask_idx += 1

            enhanced_detections.append(enhanced_detection)

        return enhanced_detections

    def _apply_hog_enhancement(self, image: np.ndarray, detections: list) -> list:
        """Apply HOG-based confidence enhancement."""
        if not detections:
            return detections

        # Convert detections to dictionary format for HOG processing
        detection_dicts = []
        for detection in detections:
            det_dict = {
                "label": detection.label,
                "score": detection.score,
            }
            if detection.bbox is not None:
                det_dict["bbox"] = list(detection.bbox)
            detection_dicts.append(det_dict)

        # Apply HOG enhancement
        enhanced_dicts = apply_hog_enhancement(image, detection_dicts)

        # Convert back to Detection objects
        enhanced_detections = []
        for i, enhanced_dict in enumerate(enhanced_dicts):
            original_detection = detections[i]

            from src.vip.detect.base import Detection

            enhanced_detection = Detection(
                label=enhanced_dict["label"],
                score=enhanced_dict["score"],
                bbox=(
                    tuple(enhanced_dict["bbox"])
                    if "bbox" in enhanced_dict
                    else original_detection.bbox
                ),
                mask=original_detection.mask,
            )
            enhanced_detections.append(enhanced_detection)

        return enhanced_detections

    def teardown(self) -> None:
        """Clean up resources."""
        self.pipeline = None


class BackgroundRemovalTechnique(ImprovedTechnique):
    """Technique with only background removal enabled."""

    def __init__(self, name: str = "bg_removal", **params):
        params.update(
            {
                "use_background_removal": True,
                "use_glare_removal": False,
                "use_region_growing": False,
                "use_hog_enhancement": False,
            }
        )
        super().__init__(name, **params)


class GlareRemovalTechnique(ImprovedTechnique):
    """Technique with only glare removal enabled."""

    def __init__(self, name: str = "glare_removal", **params):
        params.update(
            {
                "use_background_removal": False,
                "use_glare_removal": True,
                "use_region_growing": False,
                "use_hog_enhancement": False,
            }
        )
        super().__init__(name, **params)


class RegionGrowingTechnique(ImprovedTechnique):
    """Technique with only region growing enabled."""

    def __init__(self, name: str = "region_growing", **params):
        params.update(
            {
                "use_background_removal": False,
                "use_glare_removal": False,
                "use_region_growing": True,
                "use_hog_enhancement": False,
            }
        )
        super().__init__(name, **params)


class HOGEnhancementTechnique(ImprovedTechnique):
    """Technique with only HOG enhancement enabled."""

    def __init__(self, name: str = "hog_enhancement", **params):
        params.update(
            {
                "use_background_removal": False,
                "use_glare_removal": False,
                "use_region_growing": False,
                "use_hog_enhancement": True,
            }
        )
        super().__init__(name, **params)
