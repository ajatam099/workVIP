"""Vision Inspection Pipeline for defect detection on plastic workpieces."""

__version__ = "0.1.0"
__author__ = "VIP Team"

from .config import RunConfig
from .pipeline import Pipeline

__all__ = ["Pipeline", "RunConfig"]
