"""Image loading utilities for the Vision Inspection Pipeline."""

import os
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np


def iter_images(input_dir: str) -> Iterable[tuple[str, np.ndarray]]:
    """
    Iterate over images in the input directory.

    Args:
        input_dir: Path to directory containing images

    Yields:
        Tuple of (stem, image) where stem is filename without extension
        and image is BGR numpy array
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                # Read image in BGR format (OpenCV default)
                image = cv2.imread(str(file_path))
                if image is not None:
                    stem = file_path.stem
                    yield stem, image
                else:
                    print(f"Warning: Could not read image {file_path}")
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {e}")


def load_image(image_path: str) -> np.ndarray:
    """
    Load a single image from path.

    Args:
        image_path: Path to image file

    Returns:
        BGR numpy array of the image

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be read
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image {image_path}")

    return image
