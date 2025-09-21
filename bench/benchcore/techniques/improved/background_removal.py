"""Background removal preprocessing module."""


import cv2
import numpy as np


class BackgroundRemover:
    """Handles background subtraction and masking for defect detection."""

    def __init__(self, method: str = "mog2", learning_rate: float = 0.01):
        """
        Initialize background remover.

        Args:
            method: Background subtraction method ("mog2", "knn", "static")
            learning_rate: Learning rate for adaptive methods
        """
        self.method = method
        self.learning_rate = learning_rate
        self.bg_subtractor = None
        self.static_background = None

        if method == "mog2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, varThreshold=16
            )
        elif method == "knn":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    def set_static_background(self, background_image: np.ndarray) -> None:
        """
        Set a static background image for subtraction.

        Args:
            background_image: Reference background image
        """
        self.static_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)

    def remove_background(
        self, image: np.ndarray, mask_threshold: int = 127
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove background from image.

        Args:
            image: Input BGR image
            mask_threshold: Threshold for binary mask

        Returns:
            Tuple of (processed_image, foreground_mask)
        """
        if self.method == "static" and self.static_background is not None:
            return self._static_background_removal(image, mask_threshold)
        elif self.bg_subtractor is not None:
            return self._adaptive_background_removal(image, mask_threshold)
        else:
            # Fallback: simple edge-based background detection
            return self._simple_background_removal(image, mask_threshold)

    def _static_background_removal(
        self, image: np.ndarray, mask_threshold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove static background using reference image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(gray, self.static_background)

        # Apply Gaussian blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # Create binary mask
        _, mask = cv2.threshold(diff, mask_threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply mask to original image
        result = image.copy()
        result[mask == 0] = [0, 0, 0]  # Set background pixels to black

        return result, mask

    def _adaptive_background_removal(
        self, image: np.ndarray, mask_threshold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove background using adaptive background subtractor."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image, learningRate=self.learning_rate)

        # Post-process mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Apply mask to original image
        result = image.copy()
        result[fg_mask == 0] = [0, 0, 0]

        return result, fg_mask

    def _simple_background_removal(
        self, image: np.ndarray, mask_threshold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple background removal using edge detection and intensity analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use edge detection to find object boundaries
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to create connected regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours and create mask for largest regions
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create mask
        mask = np.zeros(gray.shape, dtype=np.uint8)

        # Fill contours that are large enough to be objects (not noise)
        min_area = image.shape[0] * image.shape[1] * 0.01  # At least 1% of image
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)

        # Fill holes in the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply mask to original image
        result = image.copy()
        result[mask == 0] = [0, 0, 0]

        return result, mask

    def create_object_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a mask highlighting the main object(s) in the image.

        Args:
            image: Input BGR image

        Returns:
            Binary mask where 255 = object, 0 = background
        """
        _, mask = self.remove_background(image)
        return mask


def apply_background_removal(
    image: np.ndarray, method: str = "simple", background_image: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to apply background removal.

    Args:
        image: Input BGR image
        method: Background removal method
        background_image: Optional reference background image

    Returns:
        Tuple of (processed_image, foreground_mask)
    """
    remover = BackgroundRemover(method=method)

    if background_image is not None and method == "static":
        remover.set_static_background(background_image)

    return remover.remove_background(image)
