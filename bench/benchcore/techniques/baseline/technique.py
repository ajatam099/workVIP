"""Baseline technique using the current VIP pipeline."""

import time
from typing import Any

import numpy as np

from bench.benchcore.adapters.pipeline_adapter import convert_detections_to_standard
from bench.benchcore.techniques.base import BaseTechnique
from src.vip.config import RunConfig
from src.vip.pipeline import Pipeline


class BaselineTechnique(BaseTechnique):
    """Baseline technique using the current VIP pipeline."""

    def __init__(self, name: str = "baseline", **params):
        """
        Initialize baseline technique.

        Args:
            name: Technique name
            **params: Additional parameters for pipeline configuration
        """
        super().__init__(name, **params)
        self.pipeline = None

    def setup(self, device: str = "cpu") -> None:
        """Initialize the VIP pipeline."""
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
        print(f"✅ Initialized baseline technique with {len(defects)} detectors")

    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        """
        Run detection on a batch of images.

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
            detections = self.pipeline.run_on_image(image)
            latency_ms = (time.time() - start_time) * 1000

            # Convert to standard format
            result = convert_detections_to_standard(detections, image_id, latency_ms)
            results.append(result)

        return results

    def teardown(self) -> None:
        """Clean up resources."""
        self.pipeline = None
