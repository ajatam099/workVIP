#!/usr/bin/env python3
"""
Create standardized manifests for downloaded datasets.
Converts various annotation formats to unified JSONL manifests.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ManifestCreator:
    """Creates standardized dataset manifests."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize manifest creator.

        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.configs_dir = Path("bench/configs/datasets")

    def load_dataset_config(self, dataset_name: str) -> dict[str, Any]:
        """Load dataset configuration YAML."""
        config_file = self.configs_dir / f"{dataset_name}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_file}")

        with open(config_file) as f:
            return yaml.safe_load(f)

    def create_manifest_entry(
        self,
        image_path: str,
        split: str,
        dataset_config: dict[str, Any],
        labels: list[str] | None = None,
        boxes: list[list[float]] | None = None,
        masks_path: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a standardized manifest entry.

        Args:
            image_path: Relative path to image
            split: Data split (train/val/test)
            dataset_config: Dataset configuration
            labels: List of defect labels
            boxes: List of bounding boxes [x, y, w, h]
            masks_path: Path to mask file
            meta: Additional metadata

        Returns:
            Manifest entry dictionary
        """
        # Generate unique ID
        image_id = f"{dataset_config['name']}_{Path(image_path).stem}"

        # Determine task type
        task = dataset_config.get("task_type", "classification")

        # Map labels if mapping provided
        label_map = dataset_config.get("label_map", {})
        if labels and label_map:
            labels = [label_map.get(label, label) for label in labels]

        entry = {
            "id": image_id,
            "split": split,
            "image_path": image_path,
            "task": task,
            "gt_labels": labels or [],
            "dataset": dataset_config["name"],
        }

        # Add detection-specific fields
        if boxes:
            entry["gt_boxes"] = boxes

        if masks_path:
            entry["gt_masks_path"] = masks_path

        # Add metadata
        if meta:
            entry["meta"] = meta
        else:
            entry["meta"] = {}

        # Add dataset-specific metadata
        entry["meta"].update(
            {
                "license": dataset_config.get("license", "Unknown"),
                "source": dataset_config.get("source_url", ""),
                "vip_relevance": dataset_config.get("vip_relevance", "unknown"),
            }
        )

        return entry

    def create_mvtec_manifest(self, dataset_name: str) -> list[dict[str, Any]]:
        """Create manifest for MVTec AD dataset."""
        config = self.load_dataset_config(dataset_name)
        dataset_dir = self.raw_dir / dataset_name
        manifest_entries = []

        # Focus on relevant categories
        relevant_categories = config.get("categories", ["bottle", "capsule"])

        for category in relevant_categories:
            category_dir = dataset_dir / category
            if not category_dir.exists():
                print(f"Warning: Category {category} not found in {dataset_name}")
                continue

            # Process train split (only normal images)
            train_dir = category_dir / "train" / "good"
            if train_dir.exists():
                for img_file in train_dir.glob("*.png"):
                    rel_path = str(img_file.relative_to(self.raw_dir))
                    entry = self.create_manifest_entry(
                        image_path=rel_path,
                        split="train",
                        dataset_config=config,
                        labels=["good"],
                        meta={"category": category, "anomaly_type": "good"},
                    )
                    manifest_entries.append(entry)

            # Process test split (normal + anomalous)
            test_dir = category_dir / "test"
            if test_dir.exists():
                for anomaly_dir in test_dir.iterdir():
                    if anomaly_dir.is_dir():
                        anomaly_type = anomaly_dir.name
                        for img_file in anomaly_dir.glob("*.png"):
                            rel_path = str(img_file.relative_to(self.raw_dir))
                            entry = self.create_manifest_entry(
                                image_path=rel_path,
                                split="test",
                                dataset_config=config,
                                labels=[anomaly_type],
                                meta={"category": category, "anomaly_type": anomaly_type},
                            )
                            manifest_entries.append(entry)

        return manifest_entries

    def create_classification_manifest(self, dataset_name: str) -> list[dict[str, Any]]:
        """Create manifest for classification datasets (like NEU)."""
        config = self.load_dataset_config(dataset_name)
        dataset_dir = self.raw_dir / dataset_name
        manifest_entries = []

        classes = config.get("classes", [])
        splits = config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})

        # Handle special dataset structures
        if dataset_name == "neu_surface_defects":
            neu_det_dir = dataset_dir / "NEU-DET"
            if neu_det_dir.exists():
                # Process train split
                train_images_dir = neu_det_dir / "train" / "images"
                if train_images_dir.exists():
                    for class_name in classes:
                        class_dir = train_images_dir / class_name
                        if class_dir.exists():
                            for img_file in class_dir.glob("*.jpg"):
                                rel_path = str(img_file.relative_to(self.raw_dir))
                                entry = self.create_manifest_entry(
                                    image_path=rel_path,
                                    split="train",
                                    dataset_config=config,
                                    labels=[class_name],
                                    meta={"class": class_name},
                                )
                                manifest_entries.append(entry)

                # Process validation split
                val_images_dir = neu_det_dir / "validation" / "images"
                if val_images_dir.exists():
                    for class_name in classes:
                        class_dir = val_images_dir / class_name
                        if class_dir.exists():
                            for img_file in class_dir.glob("*.jpg"):
                                rel_path = str(img_file.relative_to(self.raw_dir))
                                entry = self.create_manifest_entry(
                                    image_path=rel_path,
                                    split="val",
                                    dataset_config=config,
                                    labels=[class_name],
                                    meta={"class": class_name},
                                )
                                manifest_entries.append(entry)
                return manifest_entries

        elif dataset_name == "tarros_dataset":
            # Tarros dataset has DATASET_improved/[angle]/[split]/[class]/ structure
            dataset_improved_dir = dataset_dir / "DATASET_improved"
            if dataset_improved_dir.exists():
                angles = ["1_front", "2_back", "3_up"]
                splits_map = {"train": "train", "validation": "val", "test": "test"}
                class_map = {"1_true": "good", "2_false": "defective"}

                for angle in angles:
                    angle_dir = dataset_improved_dir / angle
                    if angle_dir.exists():
                        for split_name, split_key in splits_map.items():
                            split_dir = angle_dir / split_name
                            if split_dir.exists():
                                for class_dir_name, class_label in class_map.items():
                                    class_dir = split_dir / class_dir_name
                                    if class_dir.exists():
                                        for img_file in class_dir.glob("*.jpg"):
                                            rel_path = str(img_file.relative_to(self.raw_dir))
                                            entry = self.create_manifest_entry(
                                                image_path=rel_path,
                                                split=split_key,
                                                dataset_config=config,
                                                labels=[class_label],
                                                meta={
                                                    "class": class_label,
                                                    "angle": angle,
                                                    "original_class": class_dir_name,
                                                },
                                            )
                                            manifest_entries.append(entry)
                return manifest_entries

        # Original logic for other datasets
        for class_name in classes:
            class_dir = dataset_dir / class_name
            if not class_dir.exists():
                # Try alternative structure
                class_files = list(dataset_dir.glob(f"*{class_name}*"))
                if not class_files:
                    print(f"Warning: Class {class_name} not found in {dataset_name}")
                    continue
                class_files = [f for f in class_files if f.is_file()]
            else:
                class_files = (
                    list(class_dir.glob("*.jpg"))
                    + list(class_dir.glob("*.png"))
                    + list(class_dir.glob("*.bmp"))
                )

            # Split files according to configuration
            total_files = len(class_files)
            train_count = int(total_files * splits.get("train", 0.7))
            val_count = int(total_files * splits.get("val", 0.15))

            for i, img_file in enumerate(class_files):
                rel_path = str(img_file.relative_to(self.raw_dir))

                # Determine split
                if i < train_count:
                    split = "train"
                elif i < train_count + val_count:
                    split = "val"
                else:
                    split = "test"

                entry = self.create_manifest_entry(
                    image_path=rel_path,
                    split=split,
                    dataset_config=config,
                    labels=[class_name],
                    meta={"class": class_name},
                )
                manifest_entries.append(entry)

        return manifest_entries

    def create_detection_manifest(self, dataset_name: str) -> list[dict[str, Any]]:
        """Create manifest for detection datasets (like GC10-DET)."""
        config = self.load_dataset_config(dataset_name)
        dataset_dir = self.raw_dir / dataset_name
        manifest_entries = []

        # Handle COCO format datasets (like Roboflow)
        if config.get("annotation_format") == "coco":
            return self.create_coco_manifest(dataset_name, config, dataset_dir)

        # Handle GC10-DET special structure
        if dataset_name == "gc10_det":
            # GC10-DET has numbered directories (1-10) for different defect types
            defect_map = {
                "1": "punching_hole",
                "2": "welding_line",
                "3": "crescent_gap",
                "4": "water_spot",
                "5": "oil_spot",
                "6": "silk_spot",
                "7": "inclusion",
                "8": "rolled_pit",
                "9": "crease",
                "10": "waist_folding",
            }

            for class_num, class_name in defect_map.items():
                class_dir = dataset_dir / class_num
                if class_dir.exists():
                    image_files = list(class_dir.glob("*.jpg"))

                    # Split files according to configuration
                    splits = config.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
                    total_files = len(image_files)
                    train_count = int(total_files * splits.get("train", 0.8))
                    val_count = int(total_files * splits.get("val", 0.1))

                    for i, img_file in enumerate(image_files):
                        rel_path = str(img_file.relative_to(self.raw_dir))

                        # Determine split
                        if i < train_count:
                            split = "train"
                        elif i < train_count + val_count:
                            split = "val"
                        else:
                            split = "test"

                        entry = self.create_manifest_entry(
                            image_path=rel_path,
                            split=split,
                            dataset_config=config,
                            labels=[class_name],
                            meta={"class": class_name, "class_number": class_num},
                        )
                        manifest_entries.append(entry)

            return manifest_entries

        # Look for annotation files (common formats)
        annotation_files = (
            list(dataset_dir.glob("*.json"))  # COCO format
            + list(dataset_dir.glob("*.xml"))  # Pascal VOC format
            + list(dataset_dir.glob("*.txt"))  # YOLO format
        )

        if annotation_files:
            print(
                f"Found annotation files for {dataset_name}: {[f.name for f in annotation_files]}"
            )
            # TODO: Parse specific annotation formats
            # For now, create basic entries for images

        # Fallback: create entries for all images
        image_extensions = ["*.jpg", "*.png", "*.bmp", "*.jpeg"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_dir.glob(ext))

        splits = config.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
        total_files = len(image_files)
        train_count = int(total_files * splits.get("train", 0.8))
        val_count = int(total_files * splits.get("val", 0.1))

        for i, img_file in enumerate(image_files):
            rel_path = str(img_file.relative_to(self.raw_dir))

            # Determine split
            if i < train_count:
                split = "train"
            elif i < train_count + val_count:
                split = "val"
            else:
                split = "test"

            entry = self.create_manifest_entry(
                image_path=rel_path,
                split=split,
                dataset_config=config,
                labels=["unknown"],  # Would need annotation parsing
                meta={"needs_annotation_parsing": True},
            )
            manifest_entries.append(entry)

        return manifest_entries

    def create_coco_manifest(
        self, dataset_name: str, config: dict[str, Any], dataset_dir: Path
    ) -> list[dict[str, Any]]:
        """Create manifest for COCO format datasets."""
        manifest_entries = []

        # Process each split (train, valid, test)
        splits = ["train", "valid", "test"]
        split_mapping = {"valid": "val"}  # Roboflow uses 'valid', we use 'val'

        for split in splits:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                print(f"Warning: Split {split} not found in {dataset_name}")
                continue

            # Look for COCO annotation file
            annotation_file = split_dir / "_annotations.coco.json"
            if not annotation_file.exists():
                print(f"Warning: COCO annotations not found for {split} split")
                continue

            # Load COCO annotations
            try:
                with open(annotation_file) as f:
                    coco_data = json.load(f)

                # Create image ID to filename mapping
                images_info = {img["id"]: img for img in coco_data["images"]}

                # Create category ID to name mapping
                categories_info = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

                # Process annotations
                image_annotations = {}
                for ann in coco_data["annotations"]:
                    image_id = ann["image_id"]
                    if image_id not in image_annotations:
                        image_annotations[image_id] = []

                    # Convert COCO bbox format [x, y, width, height] to our format
                    bbox = ann["bbox"]
                    category_name = categories_info[ann["category_id"]]

                    image_annotations[image_id].append(
                        {
                            "label": category_name,
                            "bbox": bbox,  # Already in [x, y, w, h] format
                            "area": ann.get("area", bbox[2] * bbox[3]),
                        }
                    )

                # Create manifest entries
                for image_id, image_info in images_info.items():
                    # Check if images are in images/ subdirectory or directly in split directory
                    images_subdir = split_dir / "images"
                    if images_subdir.exists():
                        image_path = f"{dataset_name}/{split}/images/{image_info['file_name']}"
                    else:
                        image_path = f"{dataset_name}/{split}/{image_info['file_name']}"

                    # Get annotations for this image
                    annotations = image_annotations.get(image_id, [])
                    labels = [ann["label"] for ann in annotations]
                    boxes = [ann["bbox"] for ann in annotations]

                    # Map split name
                    manifest_split = split_mapping.get(split, split)

                    entry = self.create_manifest_entry(
                        image_path=image_path,
                        split=manifest_split,
                        dataset_config=config,
                        labels=labels,
                        boxes=boxes,
                        meta={
                            "coco_image_id": image_id,
                            "image_width": image_info["width"],
                            "image_height": image_info["height"],
                            "num_annotations": len(annotations),
                        },
                    )
                    manifest_entries.append(entry)

                print(
                    f"✓ Processed {split}: {len(images_info)} images, {len(coco_data['annotations'])} annotations"
                )

            except Exception as e:
                print(f"Error processing {split} annotations: {e}")

        return manifest_entries

    def create_manifest(self, dataset_name: str, force: bool = False) -> str:
        """
        Create manifest for a dataset.

        Args:
            dataset_name: Name of dataset
            force: Force recreation if manifest exists

        Returns:
            Path to created manifest file
        """
        # Load dataset configuration
        try:
            config = self.load_dataset_config(dataset_name)
        except FileNotFoundError:
            print(f"Error: No configuration found for dataset '{dataset_name}'")
            return ""

        # Create processed directory
        processed_dataset_dir = self.processed_dir / dataset_name
        processed_dataset_dir.mkdir(parents=True, exist_ok=True)

        manifest_file = processed_dataset_dir / "manifest.jsonl"

        # Check if manifest already exists
        if manifest_file.exists() and not force:
            print(f"Manifest already exists: {manifest_file} (use --force to recreate)")
            return str(manifest_file)

        # Check if raw data exists
        raw_dataset_dir = self.raw_dir / dataset_name
        if not raw_dataset_dir.exists():
            print(f"Warning: Raw data not found for {dataset_name} at {raw_dataset_dir}")
            print("Run download script first: python scripts/download_datasets.py")
            return ""

        print(f"Creating manifest for {dataset_name}...")

        # Create manifest based on task type
        task_type = config.get("task_type", "classification")

        if dataset_name == "mvtec_ad":
            manifest_entries = self.create_mvtec_manifest(dataset_name)
        elif task_type == "classification":
            manifest_entries = self.create_classification_manifest(dataset_name)
        elif task_type == "detection":
            manifest_entries = self.create_detection_manifest(dataset_name)
        else:
            print(f"Warning: Unknown task type '{task_type}' for {dataset_name}")
            manifest_entries = []

        # Write manifest
        if manifest_entries:
            with open(manifest_file, "w") as f:
                for entry in manifest_entries:
                    f.write(json.dumps(entry) + "\n")

            print(f"✓ Created manifest with {len(manifest_entries)} entries: {manifest_file}")

            # Print summary statistics
            splits = {}
            labels = set()
            for entry in manifest_entries:
                split = entry["split"]
                splits[split] = splits.get(split, 0) + 1
                labels.update(entry["gt_labels"])

            print(f"  Splits: {splits}")
            print(f"  Labels: {sorted(labels)}")

        else:
            print(f"✗ No entries created for {dataset_name}")

        return str(manifest_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create dataset manifests for VIP benchmarking")
    parser.add_argument("--dataset", type=str, help="Create manifest for specific dataset")
    parser.add_argument(
        "--all", action="store_true", help="Create manifests for all configured datasets"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force recreation of existing manifests"
    )
    parser.add_argument("--data-dir", default="data", help="Base data directory (default: data)")
    parser.add_argument("--list", action="store_true", help="List available dataset configurations")

    args = parser.parse_args()

    creator = ManifestCreator(args.data_dir)

    if args.list:
        configs_dir = Path("bench/configs/datasets")
        if configs_dir.exists():
            config_files = list(configs_dir.glob("*.yaml"))
            print("Available dataset configurations:")
            for config_file in config_files:
                dataset_name = config_file.stem
                print(f"  {dataset_name}")
        else:
            print("No dataset configurations found")
        return

    if args.dataset:
        # Create manifest for specific dataset
        manifest_file = creator.create_manifest(args.dataset, args.force)
        if manifest_file:
            print(f"✓ Manifest created: {manifest_file}")

    elif args.all:
        # Create manifests for all configured datasets
        configs_dir = Path("bench/configs/datasets")
        if not configs_dir.exists():
            print("Error: No dataset configurations found")
            return

        config_files = list(configs_dir.glob("*.yaml"))
        success_count = 0

        for config_file in config_files:
            dataset_name = config_file.stem
            manifest_file = creator.create_manifest(dataset_name, args.force)
            if manifest_file:
                success_count += 1

        print(f"\n✓ Created manifests for {success_count}/{len(config_files)} datasets")

    else:
        print("Error: Specify --dataset <name> or --all")
        parser.print_help()


if __name__ == "__main__":
    main()
