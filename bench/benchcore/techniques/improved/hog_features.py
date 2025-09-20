"""HOG (Histogram of Oriented Gradients) features for defect confidence scoring."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity


class HOGFeatureExtractor:
    """Extracts HOG features for defect confidence scoring."""
    
    def __init__(self, cell_size: Tuple[int, int] = (8, 8),
                 block_size: Tuple[int, int] = (2, 2),
                 num_bins: int = 9):
        """
        Initialize HOG feature extractor.
        
        Args:
            cell_size: Size of HOG cells in pixels
            block_size: Size of HOG blocks in cells
            num_bins: Number of orientation bins
        """
        self.cell_size = cell_size
        self.block_size = block_size
        self.num_bins = num_bins
        
        # Initialize HOG descriptor
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 64),  # Will be adjusted per detection
            _blockSize=(block_size[0] * cell_size[0], block_size[1] * cell_size[1]),
            _blockStride=(cell_size[0], cell_size[1]),
            _cellSize=cell_size,
            _nbins=num_bins
        )
        
        # Template features for different defect types
        self.defect_templates = {}
    
    def extract_hog_features(self, image_patch: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from an image patch.
        
        Args:
            image_patch: Grayscale image patch
            
        Returns:
            HOG feature vector
        """
        # Resize patch to standard size for consistent feature length
        standard_size = (64, 64)
        if image_patch.shape[:2] != standard_size:
            resized = cv2.resize(image_patch, standard_size)
        else:
            resized = image_patch
        
        # Extract HOG features
        features = self.hog.compute(resized)
        
        if features is not None:
            return features.flatten()
        else:
            return np.zeros(1764)  # Standard HOG feature length for 64x64
    
    def create_defect_template(self, defect_type: str, 
                             template_patches: List[np.ndarray]) -> np.ndarray:
        """
        Create a template feature vector for a defect type.
        
        Args:
            defect_type: Type of defect (e.g., 'crack', 'scratch')
            template_patches: List of example patches for this defect type
            
        Returns:
            Mean HOG feature vector for the defect type
        """
        feature_vectors = []
        
        for patch in template_patches:
            if len(patch.shape) == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
            features = self.extract_hog_features(patch)
            feature_vectors.append(features)
        
        if feature_vectors:
            template = np.mean(feature_vectors, axis=0)
            self.defect_templates[defect_type] = template
            return template
        else:
            return np.zeros(1764)
    
    def load_default_templates(self):
        """Load default templates for common defect types."""
        # Create synthetic templates based on expected defect characteristics
        
        # Crack template: Strong vertical/horizontal gradients
        crack_patch = np.zeros((64, 64), dtype=np.uint8)
        cv2.line(crack_patch, (32, 10), (32, 54), 255, 2)  # Vertical line
        crack_features = self.extract_hog_features(crack_patch)
        self.defect_templates['crack'] = crack_features
        
        # Scratch template: Elongated diagonal gradients
        scratch_patch = np.zeros((64, 64), dtype=np.uint8)
        cv2.line(scratch_patch, (10, 20), (54, 44), 255, 1)  # Diagonal line
        scratch_features = self.extract_hog_features(scratch_patch)
        self.defect_templates['scratch'] = scratch_features
        
        # Contamination template: Circular/blob-like gradients
        contamination_patch = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(contamination_patch, (32, 32), 15, 255, -1)  # Filled circle
        contamination_features = self.extract_hog_features(contamination_patch)
        self.defect_templates['contamination'] = contamination_features
        
        # Discoloration template: Smooth gradients
        discoloration_patch = np.zeros((64, 64), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                distance = np.sqrt((i - 32)**2 + (j - 32)**2)
                discoloration_patch[i, j] = max(0, 255 - int(distance * 4))
        discoloration_features = self.extract_hog_features(discoloration_patch)
        self.defect_templates['discoloration'] = discoloration_features
    
    def calculate_confidence_score(self, image_patch: np.ndarray, 
                                 defect_type: str) -> float:
        """
        Calculate confidence score for a detection based on HOG similarity.
        
        Args:
            image_patch: Detected region as grayscale image
            defect_type: Expected defect type
            
        Returns:
            Confidence score between 0 and 1
        """
        if defect_type not in self.defect_templates:
            # If no template available, return moderate confidence
            return 0.5
        
        # Extract features from the patch
        patch_features = self.extract_hog_features(image_patch)
        template_features = self.defect_templates[defect_type]
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            patch_features.reshape(1, -1),
            template_features.reshape(1, -1)
        )[0, 0]
        
        # Convert similarity to confidence score (0 to 1)
        # Cosine similarity ranges from -1 to 1, we map it to 0 to 1
        confidence = (similarity + 1) / 2
        
        # Apply sigmoid-like transformation to enhance discrimination
        confidence = 1 / (1 + np.exp(-5 * (confidence - 0.5)))
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def enhance_detection_confidence(self, image: np.ndarray,
                                   detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance detection confidence scores using HOG features.
        
        Args:
            image: Full input image (BGR)
            detections: List of detection dictionaries with bbox and label
            
        Returns:
            Enhanced detections with updated confidence scores
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        enhanced_detections = []
        
        for detection in detections:
            enhanced_detection = detection.copy()
            
            if 'bbox' in detection and 'label' in detection:
                x, y, w, h = detection['bbox']
                
                # Extract patch
                patch = gray[y:y+h, x:x+w]
                
                if patch.size > 0:
                    # Calculate HOG-based confidence
                    hog_confidence = self.calculate_confidence_score(
                        patch, detection['label']
                    )
                    
                    # Combine original confidence with HOG confidence
                    original_confidence = detection.get('score', 0.5)
                    
                    # Weighted combination (70% original, 30% HOG)
                    combined_confidence = (0.7 * original_confidence + 
                                         0.3 * hog_confidence)
                    
                    enhanced_detection['score'] = combined_confidence
                    enhanced_detection['hog_confidence'] = hog_confidence
                    enhanced_detection['original_confidence'] = original_confidence
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections


class DefectShapeAnalyzer:
    """Analyzes defect shapes using geometric and HOG features."""
    
    def __init__(self):
        self.hog_extractor = HOGFeatureExtractor()
        self.hog_extractor.load_default_templates()
    
    def analyze_defect_shape(self, mask: np.ndarray, 
                           defect_type: str) -> Dict[str, float]:
        """
        Analyze defect shape characteristics.
        
        Args:
            mask: Binary mask of the defect region
            defect_type: Expected defect type
            
        Returns:
            Dictionary of shape metrics
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'shape_score': 0.0}
        
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape metrics
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return {'shape_score': 0.0}
        
        # Shape metrics
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        
        # Convex hull
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Shape score based on defect type expectations
        shape_score = self._calculate_shape_score(
            defect_type, circularity, aspect_ratio, solidity
        )
        
        return {
            'shape_score': shape_score,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'area': area,
            'perimeter': perimeter
        }
    
    def _calculate_shape_score(self, defect_type: str, circularity: float,
                             aspect_ratio: float, solidity: float) -> float:
        """Calculate shape score based on defect type expectations."""
        
        if defect_type in ['crack', 'scratch']:
            # Cracks and scratches should be elongated (high aspect ratio)
            # and have low circularity
            elongation_score = min(1.0, aspect_ratio / 5.0)  # Normalize by expected max
            linearity_score = 1.0 - circularity  # Low circularity is good
            return (elongation_score + linearity_score) / 2
        
        elif defect_type == 'contamination':
            # Contamination should be more circular/blob-like
            blob_score = circularity
            compactness_score = solidity
            return (blob_score + compactness_score) / 2
        
        elif defect_type == 'discoloration':
            # Discoloration can be various shapes but should be solid
            return solidity
        
        else:
            # Default: balanced score
            return (circularity + solidity) / 2


def apply_hog_enhancement(image: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to apply HOG-based confidence enhancement.
    
    Args:
        image: Input BGR image
        detections: List of detection dictionaries
        
    Returns:
        Enhanced detections with improved confidence scores
    """
    extractor = HOGFeatureExtractor()
    extractor.load_default_templates()
    
    return extractor.enhance_detection_confidence(image, detections)
