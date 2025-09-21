"""Visualization utilities for the Vision Inspection Pipeline."""


import cv2
import numpy as np

from ..detect.base import Detection


def overlay_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """
    Overlay detection results on the input image.

    Args:
        image: BGR input image
        detections: List of Detection objects

    Returns:
        Image with detections overlaid
    """
    # Create a copy to avoid modifying the original
    overlay = image.copy()

    for detection in detections:
        # Draw bounding box if available
        if detection.bbox:
            x, y, w, h = detection.bbox
            # Draw green rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add label with score
            label_text = f"{detection.label}:{detection.score:.2f}"
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw background rectangle for text
            cv2.rectangle(
                overlay, (x, y - text_height - baseline - 5), (x + text_width, y), (0, 255, 0), -1
            )

            # Draw text
            cv2.putText(
                overlay,
                label_text,
                (x, y - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        # Draw mask if available
        if detection.mask is not None:
            # Create colored mask overlay
            mask_colored = np.zeros_like(image)
            mask_colored[detection.mask] = [0, 255, 0]  # Green mask

            # Blend with original image (semi-transparent)
            alpha = 0.3
            overlay = cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)

    return overlay


def create_heatmap(
    image: np.ndarray, detections: list[Detection], width: int = 100, height: int = 100
) -> np.ndarray:
    """
    Create a heatmap visualization of detection density.

    Args:
        image: Input image
        detections: List of Detection objects
        width: Heatmap width
        height: Heatmap height

    Returns:
        Heatmap as numpy array
    """
    heatmap = np.zeros((height, width), dtype=np.float32)

    for detection in detections:
        if detection.bbox:
            x, y, w, h = detection.bbox
            # Scale coordinates to heatmap size
            x_norm = int((x / image.shape[1]) * width)
            y_norm = int((y / image.shape[0]) * height)
            w_norm = max(1, int((w / image.shape[1]) * width))
            h_norm = max(1, int((h / image.shape[0]) * height))

            # Add detection score to heatmap
            heatmap[y_norm : y_norm + h_norm, x_norm : x_norm + w_norm] += detection.score

    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Convert to BGR for OpenCV
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return heatmap_colored
