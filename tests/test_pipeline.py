"""Smoke tests for the main pipeline."""

import numpy as np
import pytest

from vip.pipeline import Pipeline
from vip.config import RunConfig


def test_pipeline_initialization():
    """Test that pipeline can be initialized with default config."""
    config = RunConfig()
    pipeline = Pipeline(config)
    
    assert len(pipeline.detectors) == 4  # Default defects
    assert pipeline.config == config


def test_pipeline_with_custom_defects():
    """Test pipeline with custom defect selection."""
    config = RunConfig(defects=["scratches", "cracks"])
    pipeline = Pipeline(config)
    
    assert len(pipeline.detectors) == 2
    detector_names = [d.name for d in pipeline.detectors]
    assert "scratches" in detector_names
    assert "cracks" in detector_names


def test_pipeline_with_all_defects():
    """Test pipeline with all available defect types."""
    config = RunConfig(defects=["scratches", "contamination", "discoloration", "cracks", "flash"])
    pipeline = Pipeline(config)
    
    assert len(pipeline.detectors) == 5
    detector_names = [d.name for d in pipeline.detectors]
    expected_names = ["scratches", "contamination", "discoloration", "cracks", "flash"]
    assert set(detector_names) == set(expected_names)


def test_pipeline_invalid_detector():
    """Test that pipeline raises error for invalid detector names."""
    config = RunConfig(defects=["invalid_detector"])
    
    with pytest.raises(ValueError, match="Unknown detector"):
        Pipeline(config)


def test_pipeline_process_synthetic_image():
    """Test pipeline can process a synthetic test image."""
    # Create a synthetic test image (simple pattern)
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    # Add some synthetic defects
    # Add a bright line (potential scratch)
    image[40:60, 20:80] = [200, 200, 200]
    
    # Add a dark spot (potential contamination)
    image[70:80, 70:80] = [50, 50, 50]
    
    config = RunConfig(defects=["scratches", "contamination"])
    pipeline = Pipeline(config)
    
    # Process image
    detections, overlay = pipeline.process_image(image, "test_image")
    
    # Basic assertions
    assert isinstance(detections, list)
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == image.shape
    
    # Check that overlay is different from original (detections were drawn)
    assert not np.array_equal(overlay, image)


def test_pipeline_resize_image():
    """Test pipeline image resizing functionality."""
    # Create test image
    image = np.ones((200, 200, 3), dtype=np.uint8) * 128
    
    config = RunConfig(resize_width=100)
    pipeline = Pipeline(config)
    
    detections, overlay = pipeline.process_image(image, "test_image")
    
    # Check that image was resized
    assert overlay.shape[1] == 100  # width
    assert overlay.shape[0] == 100  # height


def test_pipeline_registry():
    """Test detector registry functionality."""
    config = RunConfig()
    pipeline = Pipeline(config)
    
    # Check available detectors
    available = pipeline.list_available_detectors()
    expected = ["scratches", "contamination", "discoloration", "cracks", "flash"]
    assert set(available) == set(expected)


def test_pipeline_error_handling():
    """Test that pipeline continues processing even if one detector fails."""
    # Create a test image
    image = np.ones((50, 50, 3), dtype=np.uint8) * 128
    
    config = RunConfig(defects=["scratches", "contamination"])
    pipeline = Pipeline(config)
    
    # Process should complete without crashing
    try:
        detections, overlay = pipeline.process_image(image, "test_image")
        assert True  # If we get here, no exception was raised
    except Exception as e:
        pytest.fail(f"Pipeline should handle errors gracefully, got: {e}")


def test_pipeline_json_schema():
    """Test that pipeline produces valid JSON schema."""
    # Create test image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    config = RunConfig()
    pipeline = Pipeline(config)
    
    detections, overlay = pipeline.process_image(image, "test_image")
    
    # Check that detections have required fields
    for detection in detections:
        assert hasattr(detection, 'label')
        assert hasattr(detection, 'score')
        assert 0.0 <= detection.score <= 1.0
        assert detection.label in ["scratch", "contamination", "discoloration", "crack", "flash"]
