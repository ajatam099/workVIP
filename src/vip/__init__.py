"""Vision Inspection Pipeline for defect detection on plastic workpieces."""

__version__ = "0.1.0"
__author__ = "VIP Team"

from .pipeline import Pipeline
from .config import RunConfig

__all__ = ["Pipeline", "RunConfig"]
