"""Glare detection and removal for shiny surfaces."""


import cv2
import numpy as np


class GlareRemover:
    """Handles specular highlight detection and removal."""

    def __init__(self, intensity_threshold: float = 0.8, saturation_threshold: float = 0.2):
        """
        Initialize glare remover.

        Args:
            intensity_threshold: Minimum intensity (V in HSV) for highlight detection
            saturation_threshold: Maximum saturation (S in HSV) for highlight detection
        """
        self.intensity_threshold = intensity_threshold
        self.saturation_threshold = saturation_threshold

    def detect_glare(self, image: np.ndarray) -> np.ndarray:
        """
        Detect specular highlights/glare in the image.

        Args:
            image: Input BGR image

        Returns:
            Binary mask where 255 = glare, 0 = no glare
        """
        # Convert to HSV for better highlight detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Normalize V channel to [0, 1]
        v_norm = v.astype(np.float32) / 255.0
        s_norm = s.astype(np.float32) / 255.0

        # Detect highlights: high intensity AND low saturation
        highlight_mask = (v_norm > self.intensity_threshold) & (s_norm < self.saturation_threshold)

        # Convert to uint8 mask
        mask = (highlight_mask * 255).astype(np.uint8)

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def remove_glare_inpainting(
        self, image: np.ndarray, glare_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Remove glare using inpainting.

        Args:
            image: Input BGR image
            glare_mask: Optional pre-computed glare mask

        Returns:
            Image with glare removed
        """
        if glare_mask is None:
            glare_mask = self.detect_glare(image)

        # Use OpenCV inpainting to fill glare regions
        result = cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)

        return result

    def remove_glare_intensity_reduction(
        self,
        image: np.ndarray,
        glare_mask: np.ndarray | None = None,
        reduction_factor: float = 0.7,
    ) -> np.ndarray:
        """
        Remove glare by reducing intensity in highlight regions.

        Args:
            image: Input BGR image
            glare_mask: Optional pre-computed glare mask
            reduction_factor: Factor to reduce intensity (0.0 = black, 1.0 = no change)

        Returns:
            Image with reduced glare
        """
        if glare_mask is None:
            glare_mask = self.detect_glare(image)

        result = image.copy().astype(np.float32)

        # Reduce intensity in glare regions
        glare_pixels = glare_mask > 0
        result[glare_pixels] *= reduction_factor

        return result.astype(np.uint8)

    def remove_glare_adaptive(
        self, image: np.ndarray, glare_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Remove glare using adaptive method based on surrounding pixels.

        Args:
            image: Input BGR image
            glare_mask: Optional pre-computed glare mask

        Returns:
            Image with adaptive glare removal
        """
        if glare_mask is None:
            glare_mask = self.detect_glare(image)

        result = image.copy()

        # For each glare pixel, replace with local median of non-glare neighbors
        kernel_size = 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Dilate glare mask to get neighborhood
        dilated_mask = cv2.dilate(glare_mask, kernel, iterations=1)

        # Find pixels that are in the dilated region but not in original glare
        neighbor_mask = dilated_mask - glare_mask

        # Apply median filter to the image
        median_filtered = cv2.medianBlur(image, kernel_size)

        # Replace glare pixels with median-filtered values
        glare_pixels = glare_mask > 0
        result[glare_pixels] = median_filtered[glare_pixels]

        return result

    def enhance_after_glare_removal(
        self, image: np.ndarray, glare_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Apply enhancement after glare removal to restore detail.

        Args:
            image: Glare-removed image
            glare_mask: Mask of original glare regions

        Returns:
            Enhanced image
        """
        if glare_mask is None:
            return image

        result = image.copy()

        # Apply sharpening to previously glare-affected regions
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)

        sharpened = cv2.filter2D(image, -1, kernel)

        # Blend sharpened result only in previously glare regions
        glare_pixels = glare_mask > 0
        alpha = 0.3  # Blending factor
        result[glare_pixels] = (
            alpha * sharpened[glare_pixels] + (1 - alpha) * result[glare_pixels]
        ).astype(np.uint8)

        return result


def apply_glare_removal(
    image: np.ndarray,
    method: str = "adaptive",
    intensity_threshold: float = 0.8,
    saturation_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to apply glare removal.

    Args:
        image: Input BGR image
        method: Removal method ("inpainting", "intensity", "adaptive")
        intensity_threshold: Intensity threshold for glare detection
        saturation_threshold: Saturation threshold for glare detection

    Returns:
        Tuple of (processed_image, glare_mask)
    """
    remover = GlareRemover(intensity_threshold, saturation_threshold)
    glare_mask = remover.detect_glare(image)

    if method == "inpainting":
        processed = remover.remove_glare_inpainting(image, glare_mask)
    elif method == "intensity":
        processed = remover.remove_glare_intensity_reduction(image, glare_mask)
    elif method == "adaptive":
        processed = remover.remove_glare_adaptive(image, glare_mask)
    else:
        raise ValueError(f"Unknown glare removal method: {method}")

    # Apply enhancement
    processed = remover.enhance_after_glare_removal(processed, glare_mask)

    return processed, glare_mask
