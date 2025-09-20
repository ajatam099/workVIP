"""Adapter for converting VIP pipeline outputs to standard format."""

from typing import List, Dict, Any, Optional
import numpy as np

from src.vip.detect.base import Detection


def convert_detection_to_dict(detection: Detection) -> Dict[str, Any]:
    """
    Convert VIP Detection object to standard dict format.
    
    Args:
        detection: VIP Detection object
        
    Returns:
        Dictionary with label, score, bbox, mask fields
    """
    result = {
        "label": str(detection.label),
        "score": float(detection.score),
    }
    
    if detection.bbox is not None:
        x, y, w, h = detection.bbox
        # Convert numpy types to native Python types for JSON serialization
        result["bbox"] = [int(x), int(y), int(w), int(h)]
    
    if detection.mask is not None:
        # For now, we'll store mask as a boolean indicator
        # In a full implementation, we might save masks to files
        result["has_mask"] = True
        result["mask_shape"] = [int(dim) for dim in detection.mask.shape]
    else:
        result["has_mask"] = False
    
    return result


def convert_detections_to_standard(detections: List[Detection], 
                                 image_id: str,
                                 latency_ms: Optional[float] = None) -> Dict[str, Any]:
    """
    Convert list of VIP detections to standard benchmarking format.
    
    Args:
        detections: List of VIP Detection objects
        image_id: Unique image identifier  
        latency_ms: Processing time in milliseconds
        
    Returns:
        Standard detection result dictionary
    """
    predictions = [convert_detection_to_dict(det) for det in detections]
    
    return {
        "image_id": image_id,
        "predictions": predictions,
        "latency_ms": latency_ms,
        "detection_count": len(predictions)
    }
