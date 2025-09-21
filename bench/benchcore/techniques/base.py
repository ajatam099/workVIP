"""Base technique interface for benchmarking."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseTechnique(ABC):
    """Base class for all detection techniques."""

    def __init__(self, name: str, **params):
        """
        Initialize technique.

        Args:
            name: Technique identifier
            **params: Technique-specific parameters
        """
        self.name = name
        self.params = params

    @abstractmethod
    def setup(self, device: str = "cpu") -> None:
        """
        Initialize technique (load models, etc.).

        Args:
            device: Device to run on (cpu/gpu)
        """
        pass

    @abstractmethod
    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        """
        Run detection on a batch of images.

        Args:
            images: List of BGR images

        Returns:
            List of detection results in standard format
        """
        pass

    def teardown(self) -> None:
        """Clean up resources."""
        pass

    def __str__(self) -> str:
        return f"{self.name}({self.__class__.__name__})"


class StandardDetectionResult:
    """Standard format for detection results."""

    def __init__(
        self, image_id: str, predictions: list[dict[str, Any]], latency_ms: float | None = None
    ):
        """
        Initialize detection result.

        Args:
            image_id: Unique image identifier
            predictions: List of predictions with label, score, bbox, mask
            latency_ms: Processing time in milliseconds
        """
        self.image_id = image_id
        self.predictions = predictions
        self.latency_ms = latency_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "image_id": self.image_id,
            "predictions": self.predictions,
            "latency_ms": self.latency_ms,
        }
