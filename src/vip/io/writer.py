"""Output writing utilities for the Vision Inspection Pipeline."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_overlay_image(path: str, image: np.ndarray) -> None:
    """
    Save overlay image to path.
    
    Args:
        path: Output path for the image
        image: BGR numpy array to save
    """
    ensure_dir(os.path.dirname(path))
    success = cv2.imwrite(path, image)
    if not success:
        raise RuntimeError(f"Failed to save overlay image to {path}")


def save_json_data(path: str, data: Dict[str, Any]) -> None:
    """
    Save JSON data to path.
    
    Args:
        path: Output path for the JSON file
        data: Dictionary to serialize to JSON
    """
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_results(
    output_dir: str,
    stem: str,
    detections: list,
    overlay_image: np.ndarray,
    save_overlay: bool = True,
    save_json: bool = True
) -> None:
    """
    Save both overlay image and JSON results for an image.

    Args:
        output_dir: Directory to save results in
        stem: Base filename without extension
        detections: List of detection results
        overlay_image: Image with detections overlaid
        save_overlay: Whether to save overlay image
        save_json: Whether to save JSON results
    """
    if save_overlay:
        overlay_path = os.path.join(output_dir, f"{stem}_overlay.jpg")
        save_overlay_image(overlay_path, overlay_image)

    if save_json:
        json_path = os.path.join(output_dir, f"{stem}.json")
        # Convert detections to serializable format
        serializable_detections = []
        for detection in detections:
            det_dict = {
                "label": detection.label,
                "score": float(detection.score),
            }
            if detection.bbox:
                det_dict["bbox"] = [int(x) for x in detection.bbox]
            if detection.mask is not None:
                # Convert mask to list of coordinates or save as separate file
                det_dict["mask_shape"] = [int(x) for x in detection.mask.shape]
            serializable_detections.append(det_dict)
        
        results = {
            "image": stem,
            "detections": serializable_detections,
            "total_defects": len(serializable_detections)
        }
        save_json_data(json_path, results)
