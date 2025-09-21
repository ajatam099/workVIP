"""Adaptive region growing for defect detection enhancement."""

from collections import deque

import cv2
import numpy as np


class RegionGrower:
    """Implements adaptive region growing for defect region expansion."""

    def __init__(
        self,
        intensity_threshold: float = 10.0,
        gradient_threshold: float = 20.0,
        max_region_size: int = 10000,
    ):
        """
        Initialize region grower.

        Args:
            intensity_threshold: Maximum intensity difference for growing
            gradient_threshold: Maximum gradient difference for growing
            max_region_size: Maximum pixels in a region
        """
        self.intensity_threshold = intensity_threshold
        self.gradient_threshold = gradient_threshold
        self.max_region_size = max_region_size

    def grow_region(
        self, image: np.ndarray, seed_point: tuple[int, int], mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Grow a region from a seed point.

        Args:
            image: Grayscale input image
            seed_point: (x, y) coordinates of seed pixel
            mask: Optional mask to constrain growing

        Returns:
            Binary mask of grown region
        """
        height, width = image.shape
        grown_mask = np.zeros((height, width), dtype=np.uint8)

        if mask is None:
            mask = np.ones((height, width), dtype=np.uint8)

        # Check if seed point is valid
        seed_x, seed_y = seed_point
        if (
            seed_x < 0
            or seed_x >= width
            or seed_y < 0
            or seed_y >= height
            or mask[seed_y, seed_x] == 0
        ):
            return grown_mask

        # Initialize queue with seed point
        queue = deque([(seed_x, seed_y)])
        grown_mask[seed_y, seed_x] = 255

        seed_intensity = float(image[seed_y, seed_x])
        region_size = 1

        # 8-connected neighbors
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while queue and region_size < self.max_region_size:
            current_x, current_y = queue.popleft()
            current_intensity = float(image[current_y, current_x])

            for dx, dy in directions:
                next_x, next_y = current_x + dx, current_y + dy

                # Check bounds
                if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                    continue

                # Check if already processed or masked out
                if grown_mask[next_y, next_x] > 0 or mask[next_y, next_x] == 0:
                    continue

                next_intensity = float(image[next_y, next_x])

                # Check intensity similarity to seed
                intensity_diff = abs(next_intensity - seed_intensity)
                if intensity_diff > self.intensity_threshold:
                    continue

                # Check intensity similarity to current pixel
                local_diff = abs(next_intensity - current_intensity)
                if local_diff > self.gradient_threshold:
                    continue

                # Add pixel to region
                grown_mask[next_y, next_x] = 255
                queue.append((next_x, next_y))
                region_size += 1

        return grown_mask

    def grow_from_edges(
        self, image: np.ndarray, edge_mask: np.ndarray, object_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Grow regions from detected edges.

        Args:
            image: Grayscale input image
            edge_mask: Binary mask of detected edges
            object_mask: Optional mask to constrain growing to object regions

        Returns:
            Binary mask of grown regions
        """
        result_mask = np.zeros_like(edge_mask)

        # Find edge pixels as seed points
        edge_points = np.where(edge_mask > 0)
        seed_points = list(zip(edge_points[1], edge_points[0], strict=False))  # (x, y) format

        for seed_point in seed_points:
            # Skip if already part of a grown region
            x, y = seed_point
            if result_mask[y, x] > 0:
                continue

            # Grow region from this seed
            grown_region = self.grow_region(image, seed_point, object_mask)

            # Add to result (logical OR)
            result_mask = cv2.bitwise_or(result_mask, grown_region)

        return result_mask

    def grow_from_detections(
        self,
        image: np.ndarray,
        detection_masks: list[np.ndarray],
        object_mask: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        Grow regions from existing detection masks.

        Args:
            image: Grayscale input image
            detection_masks: List of binary detection masks
            object_mask: Optional mask to constrain growing

        Returns:
            List of grown detection masks
        """
        grown_masks = []

        for detection_mask in detection_masks:
            # Find boundary pixels of the detection
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(detection_mask, kernel, iterations=1)
            boundary = cv2.subtract(detection_mask, eroded)

            # Use boundary pixels as seeds for growing
            grown_region = self.grow_from_edges(image, boundary, object_mask)

            # Combine original detection with grown region
            combined = cv2.bitwise_or(detection_mask, grown_region)
            grown_masks.append(combined)

        return grown_masks

    def adaptive_grow_from_bbox(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        object_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Adaptively grow a region starting from a bounding box.

        Args:
            image: Grayscale input image
            bbox: (x, y, w, h) bounding box
            object_mask: Optional mask to constrain growing

        Returns:
            Binary mask of grown region
        """
        x, y, w, h = bbox

        # Create initial mask from bounding box
        initial_mask = np.zeros_like(image)
        initial_mask[y : y + h, x : x + w] = 255

        # Find seed points within the bounding box (use center and edges)
        center_x, center_y = x + w // 2, y + h // 2
        seed_points = [
            (center_x, center_y),  # Center
            (x + w // 4, y + h // 4),  # Quarter points
            (x + 3 * w // 4, y + h // 4),
            (x + w // 4, y + 3 * h // 4),
            (x + 3 * w // 4, y + 3 * h // 4),
        ]

        result_mask = np.zeros_like(image)

        for seed_point in seed_points:
            # Check if seed is within image bounds and object mask
            seed_x, seed_y = seed_point
            if seed_x >= 0 and seed_x < image.shape[1] and seed_y >= 0 and seed_y < image.shape[0]:

                if object_mask is None or object_mask[seed_y, seed_x] > 0:
                    grown_region = self.grow_region(image, seed_point, object_mask)
                    result_mask = cv2.bitwise_or(result_mask, grown_region)

        return result_mask


def apply_region_growing(
    image: np.ndarray,
    initial_detections: list[tuple[int, int, int, int]],
    object_mask: np.ndarray | None = None,
    intensity_threshold: float = 10.0,
) -> list[np.ndarray]:
    """
    Convenience function to apply region growing to detections.

    Args:
        image: Input BGR image
        initial_detections: List of (x, y, w, h) bounding boxes
        object_mask: Optional object mask to constrain growing
        intensity_threshold: Intensity threshold for growing

    Returns:
        List of grown region masks
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    grower = RegionGrower(intensity_threshold=intensity_threshold)
    grown_masks = []

    for bbox in initial_detections:
        grown_mask = grower.adaptive_grow_from_bbox(gray, bbox, object_mask)
        grown_masks.append(grown_mask)

    return grown_masks
