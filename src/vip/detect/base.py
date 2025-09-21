"""Base classes for defect detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """Represents a detected defect."""

    label: str
    score: float
    bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    mask: np.ndarray | None = None  # Binary mask

    def __post_init__(self):
        """Validate detection data."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")

        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError(f"Bbox must have 4 elements, got {len(self.bbox)}")
            x, y, w, h = self.bbox
            if w <= 0 or h <= 0:
                raise ValueError(f"Bbox width and height must be positive, got {w}x{h}")


class BaseDetector(ABC):
    """Abstract base class for defect detectors."""

    def __init__(self, name: str):
        """
        Initialize detector.

        Args:
            name: Name identifier for this detector
        """
        self.name = name

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect defects in the input image.

        Args:
            image: BGR input image

        Returns:
            List of Detection objects
        """
        pass

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
