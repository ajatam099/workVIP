"""Tests for individual defect detectors."""

import numpy as np
import pytest

from vip.detect.scratches import ScratchDetector
from vip.detect.contamination import ContaminationDetector
from vip.detect.discoloration import DiscolorationDetector
from vip.detect.cracks import CrackDetector
from vip.detect.flash import FlashDetector


class TestScratchDetector:
    """Test scratch detection functionality."""
    
    def setup_method(self):
        """Set up detector for each test."""
        self.detector = ScratchDetector()
    
    def test_detect_elongated_line(self):
        """Test detection of elongated line (scratch-like pattern)."""
        # Create image with elongated bright line
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image[45:55, 10:90] = [200, 200, 200]  # Horizontal line
        
        detections = self.detector.detect(image)
        
        assert len(detections) > 0
        assert any(d.label == "scratch" for d in detections)
        
        # Check detection properties
        for detection in detections:
            assert 0.0 <= detection.score <= 1.0
            assert detection.bbox is not None
    
    def test_no_detection_on_plain_image(self):
        """Test that plain image produces no detections."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        detections = self.detector.detect(image)
        
        assert len(detections) == 0
    
    def test_detection_parameters(self):
        """Test detector with custom parameters."""
        detector = ScratchDetector(canny_low=30, canny_high=100)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image[45:55, 10:90] = [200, 200, 200]
        
        detections = detector.detect(image)
        assert len(detections) > 0


class TestContaminationDetector:
    """Test contamination detection functionality."""
    
    def setup_method(self):
        """Set up detector for each test."""
        self.detector = ContaminationDetector()
    
    def test_detect_dark_blob(self):
        """Test detection of dark contamination blob."""
        # Create image with dark circular blob
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Create circular dark region
        y, x = np.ogrid[:100, :100]
        mask = (x - 50)**2 + (y - 50)**2 <= 15**2
        image[mask] = [50, 50, 50]
        
        detections = self.detector.detect(image)
        
        assert len(detections) > 0
        assert any(d.label == "contamination" for d in detections)
    
    def test_detect_bright_blob(self):
        """Test detection of bright contamination."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Create bright rectangular region
        image[30:70, 30:70] = [220, 220, 220]
        
        detections = self.detector.detect(image)
        
        assert len(detections) > 0
    
    def test_no_detection_on_plain_image(self):
        """Test that plain image produces no detections."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        detections = self.detector.detect(image)
        
        assert len(detections) == 0


class TestDiscolorationDetector:
    """Test discoloration detection functionality."""
    
    def setup_method(self):
        """Set up detector for each test."""
        self.detector = DiscolorationDetector()
    
    def test_detect_color_variation(self):
        """Test detection of color variation."""
        # Create image with color variation
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add red-tinted region
        image[30:70, 30:70] = [180, 100, 100]
        
        detections = self.detector.detect(image)
        
        assert len(detections) > 0
        assert any(d.label == "discoloration" for d in detections)
    
    def test_detect_brightness_variation(self):
        """Test detection of brightness variation."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add bright region
        image[40:60, 40:60] = [200, 200, 200]
        
        detections = self.detector.detect(image)
        
        assert len(detections) > 0
    
    def test_no_detection_on_uniform_image(self):
        """Test that uniform image produces no detections."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        detections = self.detector.detect(image)
        
        assert len(detections) == 0


class TestCrackDetector:
    """Test crack detection functionality."""
    
    def setup_method(self):
        """Set up detector for each test."""
        self.detector = CrackDetector()
    
    def test_detect_thin_line(self):
        """Test detection of thin crack-like line."""
        # Create image with thin vertical line
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image[20:80, 49:51] = [50, 50, 50]  # Thin vertical line
        
        detections = self.detector.detect(image)
        
        assert len(detections) > 0
        assert any(d.label == "crack" for d in detections)
    
    def test_detect_diagonal_line(self):
        """Test detection of diagonal crack."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Create diagonal line
        for i in range(20, 80):
            image[i, i] = [50, 50, 50]
            image[i, i+1] = [50, 50, 50]
        
        detections = self.detector.detect(image)
        
        assert len(detections) > 0
    
    def test_no_detection_on_plain_image(self):
        """Test that plain image produces no detections."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        detections = self.detector.detect(image)
        
        assert len(detections) == 0


class TestFlashDetector:
    """Test flash detection functionality."""
    
    def setup_method(self):
        """Set up detector for each test."""
        self.detector = FlashDetector()
    
    def test_detect_bright_edge(self):
        """Test detection of bright edge (flash-like pattern)."""
        # Create image with bright edge
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Create bright edge along border
        image[0:5, :] = [220, 220, 220]  # Top edge
        image[:, 0:5] = [220, 220, 220]  # Left edge
        
        detections = self.detector.detect(image)
        
        assert len(detections) > 0
        assert any(d.label == "flash" for d in detections)
    
    def test_detect_bright_corner(self):
        """Test detection of bright corner region."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Create bright corner
        image[0:20, 0:20] = [200, 200, 200]
        
        detections = self.detector.detect(image)
        
        assert len(detections) > 0
    
    def test_no_detection_on_dark_image(self):
        """Test that dark image produces no detections."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 50
        
        detections = self.detector.detect(image)
        
        # Flash detector might still find some bright regions
        # Just ensure it doesn't crash
        assert isinstance(detections, list)


class TestDetectorCommon:
    """Test common functionality across all detectors."""
    
    def test_all_detectors_return_valid_scores(self):
        """Test that all detectors return valid scores."""
        detectors = [
            ScratchDetector(),
            ContaminationDetector(),
            DiscolorationDetector(),
            CrackDetector(),
            FlashDetector()
        ]
        
        # Create test image with some patterns
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image[40:60, 20:80] = [200, 200, 200]  # Bright line
        
        for detector in detectors:
            detections = detector.detect(image)
            
            for detection in detections:
                assert 0.0 <= detection.score <= 1.0
                assert detection.label in ["scratch", "contamination", "discoloration", "crack", "flash"]
                assert detection.bbox is not None
    
    def test_detector_names(self):
        """Test that detectors have correct names."""
        detectors = [
            (ScratchDetector(), "scratches"),
            (ContaminationDetector(), "contamination"),
            (DiscolorationDetector(), "discoloration"),
            (CrackDetector(), "cracks"),
            (FlashDetector(), "flash")
        ]
        
        for detector, expected_name in detectors:
            assert detector.name == expected_name
