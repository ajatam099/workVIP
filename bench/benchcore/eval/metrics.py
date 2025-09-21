"""Evaluation metrics for defect detection."""


import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def calculate_iou(bbox1: list[int], bbox2: list[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: [x, y, w, h] format
        bbox2: [x, y, w, h] format

    Returns:
        IoU score between 0 and 1
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to [x1, y1, x2, y2] format
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]

    # Calculate intersection
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def match_detections(
    predictions: list[dict], ground_truths: list[dict], iou_threshold: float = 0.5
) -> tuple[list[bool], list[bool]]:
    """
    Match predictions to ground truths based on IoU threshold.

    Args:
        predictions: List of prediction dicts with 'bbox' and 'label'
        ground_truths: List of ground truth dicts with 'bbox' and 'label'
        iou_threshold: Minimum IoU for a match

    Returns:
        Tuple of (prediction_matched, gt_matched) boolean lists
    """
    pred_matched = [False] * len(predictions)
    gt_matched = [False] * len(ground_truths)

    # Match predictions to ground truths
    for i, pred in enumerate(predictions):
        if "bbox" not in pred:
            continue

        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(ground_truths):
            if gt_matched[j] or "bbox" not in gt:
                continue

            # Only match same label types
            if pred.get("label") != gt.get("label"):
                continue

            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = j

        if best_gt_idx >= 0:
            pred_matched[i] = True
            gt_matched[best_gt_idx] = True

    return pred_matched, gt_matched


class ClassificationMetrics:
    """Calculate classification metrics."""

    @staticmethod
    def calculate(
        y_true: list[str], y_pred: list[str], y_scores: list[float] | None = None
    ) -> dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores (for AUC calculation)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["precision_macro"] = precision
        metrics["recall_macro"] = recall
        metrics["f1_macro"] = f1

        # Weighted metrics
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["precision_weighted"] = precision_w
        metrics["recall_weighted"] = recall_w
        metrics["f1_weighted"] = f1_w

        # AUC for binary classification
        unique_labels = list(set(y_true + y_pred))
        if len(unique_labels) == 2 and y_scores is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    [1 if label == unique_labels[1] else 0 for label in y_true], y_scores
                )
            except ValueError:
                metrics["roc_auc"] = 0.0

        # Log loss if scores available
        if y_scores is not None and len(unique_labels) == 2:
            try:
                binary_true = [1 if label == unique_labels[1] else 0 for label in y_true]
                metrics["log_loss"] = log_loss(binary_true, y_scores)
            except ValueError:
                metrics["log_loss"] = float("inf")

        return metrics


class DetectionMetrics:
    """Calculate detection metrics (mAP, etc.)."""

    @staticmethod
    def calculate_ap(precisions: list[float], recalls: list[float]) -> float:
        """Calculate Average Precision from precision-recall curve."""
        # Add sentinel values
        precisions = [0.0] + precisions + [0.0]
        recalls = [0.0] + recalls + [1.0]

        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Calculate AP
        ap = 0.0
        for i in range(len(recalls) - 1):
            ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]

        return ap

    @staticmethod
    def calculate_map(
        predictions_per_image: list[list[dict]],
        ground_truths_per_image: list[list[dict]],
        iou_thresholds: list[float] = None,
    ) -> dict[str, float]:
        """
        Calculate mean Average Precision (mAP).

        Args:
            predictions_per_image: List of prediction lists per image
            ground_truths_per_image: List of ground truth lists per image
            iou_thresholds: IoU thresholds to evaluate at

        Returns:
            Dictionary with mAP metrics
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        # Collect all predictions and ground truths with image indices
        all_predictions = []
        all_ground_truths = []

        for img_idx, (preds, gts) in enumerate(zip(predictions_per_image, ground_truths_per_image, strict=False)):
            for pred in preds:
                pred_copy = pred.copy()
                pred_copy["image_idx"] = img_idx
                all_predictions.append(pred_copy)

            for gt in gts:
                gt_copy = gt.copy()
                gt_copy["image_idx"] = img_idx
                all_ground_truths.append(gt_copy)

        # Sort predictions by confidence score (descending)
        all_predictions.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        metrics = {}
        aps_per_threshold = []

        for iou_thresh in iou_thresholds:
            # Calculate precision and recall at this IoU threshold
            tp = 0
            fp = 0
            gt_matched = set()

            precisions = []
            recalls = []

            for pred in all_predictions:
                # Find matching ground truth
                matched = False

                for gt_idx, gt in enumerate(all_ground_truths):
                    if (
                        gt["image_idx"] == pred["image_idx"]
                        and gt.get("label") == pred.get("label")
                        and "bbox" in pred
                        and "bbox" in gt
                    ):

                        iou = calculate_iou(pred["bbox"], gt["bbox"])
                        if iou >= iou_thresh and gt_idx not in gt_matched:
                            tp += 1
                            gt_matched.add(gt_idx)
                            matched = True
                            break

                if not matched:
                    fp += 1

                # Calculate precision and recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / len(all_ground_truths) if len(all_ground_truths) > 0 else 0.0

                precisions.append(precision)
                recalls.append(recall)

            # Calculate AP for this threshold
            ap = DetectionMetrics.calculate_ap(precisions, recalls)
            aps_per_threshold.append(ap)

            # Store specific thresholds
            if iou_thresh == 0.5:
                metrics["ap_50"] = ap
            elif iou_thresh == 0.75:
                metrics["ap_75"] = ap

        # mAP is average over all IoU thresholds
        metrics["map"] = np.mean(aps_per_threshold) if aps_per_threshold else 0.0

        # Average recall (AR)
        if all_ground_truths:
            final_recall = len(gt_matched) / len(all_ground_truths)
            metrics["average_recall"] = final_recall
        else:
            metrics["average_recall"] = 0.0

        return metrics


class PerformanceMetrics:
    """Calculate performance metrics (speed, memory, etc.)."""

    @staticmethod
    def calculate(latencies_ms: list[float]) -> dict[str, float]:
        """
        Calculate performance metrics from latency measurements.

        Args:
            latencies_ms: List of processing times in milliseconds

        Returns:
            Dictionary of performance metrics
        """
        if not latencies_ms:
            return {
                "mean_latency_ms": 0.0,
                "median_latency_ms": 0.0,
                "images_per_second": 0.0,
                "std_latency_ms": 0.0,
            }

        latencies = np.array(latencies_ms)

        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "images_per_second": 1000.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0.0,
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
        }
