#!/usr/bin/env python3
"""
Dataset downloader for VIP benchmarking framework.
Supports direct downloads, Kaggle API, and idempotent operations.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DatasetDownloader:
    """Handles dataset downloading, extraction, and staging."""

    def __init__(self, data_dir: str = "data", dry_run: bool = False):
        """
        Initialize downloader.

        Args:
            data_dir: Base data directory
            dry_run: If True, only simulate downloads
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.registry_file = self.data_dir / "download_registry.jsonl"
        self.dry_run = dry_run

        # Create directories
        if not dry_run:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.processed_dir.mkdir(parents=True, exist_ok=True)

    def calculate_sha256(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def load_registry(self) -> dict[str, dict]:
        """Load download registry."""
        registry = {}
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    registry[entry["dataset"]] = entry
        return registry

    def update_registry(self, dataset: str, entry: dict) -> None:
        """Update download registry with new entry."""
        if self.dry_run:
            print(f"[DRY RUN] Would update registry for {dataset}")
            return

        # Append new entry
        with open(self.registry_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def download_file(
        self, url: str, output_path: str, expected_size: int | None = None
    ) -> tuple[bool, str]:
        """
        Download a file from URL.

        Args:
            url: Download URL
            output_path: Local file path
            expected_size: Expected file size in bytes

        Returns:
            Tuple of (success, error_message)
        """
        if self.dry_run:
            print(f"[DRY RUN] Would download {url} -> {output_path}")
            return True, ""

        try:
            print(f"Downloading {url}...")

            # Use requests for better control
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Check content length if expected size provided
            content_length = response.headers.get("content-length")
            if content_length and expected_size:
                if int(content_length) != expected_size:
                    return False, f"Size mismatch: expected {expected_size}, got {content_length}"

            # Download with progress
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if content_length:
                            progress = (downloaded / int(content_length)) * 100
                            print(f"\rProgress: {progress:.1f}%", end="", flush=True)

            print()  # New line after progress
            return True, ""

        except Exception as e:
            return False, str(e)

    def extract_archive(self, archive_path: str, extract_to: str) -> tuple[bool, str]:
        """
        Extract archive to directory.

        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to

        Returns:
            Tuple of (success, error_message)
        """
        if self.dry_run:
            print(f"[DRY RUN] Would extract {archive_path} -> {extract_to}")
            return True, ""

        try:
            print(f"Extracting {archive_path}...")

            if archive_path.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_to)
            else:
                # Try using shutil for other formats
                shutil.unpack_archive(archive_path, extract_to)

            return True, ""

        except Exception as e:
            return False, str(e)

    def download_kaggle_dataset(self, dataset_id: str, output_dir: str) -> tuple[bool, str]:
        """
        Download dataset from Kaggle using kaggle CLI.

        Args:
            dataset_id: Kaggle dataset identifier (e.g., "alex000kim/gc10det")
            output_dir: Directory to download to

        Returns:
            Tuple of (success, error_message)
        """
        if self.dry_run:
            print(f"[DRY RUN] Would download Kaggle dataset {dataset_id} -> {output_dir}")
            return True, ""

        try:
            import subprocess

            # Check if kaggle CLI is available
            result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                return False, "Kaggle CLI not available. Install with: pip install kaggle"

            # Download dataset
            os.makedirs(output_dir, exist_ok=True)
            cmd = ["kaggle", "datasets", "download", "-d", dataset_id, "-p", output_dir, "--unzip"]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                return False, f"Kaggle download failed: {result.stderr}"

            return True, ""

        except ImportError:
            return False, "Kaggle package not installed. Install with: pip install kaggle"
        except Exception as e:
            return False, str(e)

    def create_dataset_structure(self, dataset_slug: str, source_dir: str) -> None:
        """
        Create standardized dataset structure.

        Args:
            dataset_slug: Dataset identifier
            source_dir: Source directory with extracted files
        """
        dataset_dir = self.raw_dir / dataset_slug

        if self.dry_run:
            print(f"[DRY RUN] Would create structure for {dataset_slug}")
            return

        # Create train/val/test structure if not exists
        for split in ["train", "val", "test"]:
            (dataset_dir / split).mkdir(parents=True, exist_ok=True)

        # If source has no splits, move everything to train
        if not any((Path(source_dir) / split).exists() for split in ["train", "val", "test"]):
            train_dir = dataset_dir / "train"

            # Move all files to train directory
            for item in Path(source_dir).iterdir():
                if item.is_file() or item.is_dir():
                    dest = train_dir / item.name
                    if not dest.exists():
                        shutil.move(str(item), str(dest))


# Dataset definitions with download information
DATASETS = {
    "mvtec_ad": {
        "name": "MVTec Anomaly Detection",
        "url": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
        "type": "direct",
        "license": "CC BY-NC-SA 4.0",
        "size_mb": 4800,
        "defect_types": ["scratches", "dents", "contamination", "cracks"],
        "description": "Industrial anomaly detection with bottle and capsule categories",
    },
    "roboflow_plastic_defects": {
        "name": "Roboflow Plastic Defects",
        "url": "https://universe.roboflow.com/panops/plastic-defects",
        "type": "roboflow",
        "license": "CC BY 4.0",
        "size_mb": 50,
        "defect_types": ["black_speck", "dirty_mark", "scratches"],
        "description": "149 plastic container images with defect annotations",
    },
    "severstal_steel": {
        "name": "Severstal Steel Defect Detection",
        "dataset_id": "severstal-steel-defect-detection",
        "type": "kaggle_competition",
        "license": "Competition Rules",
        "size_mb": 1700,
        "defect_types": ["steel_defects"],
        "description": "Steel strip defect detection competition data",
    },
    "neu_surface_defects": {
        "name": "NEU Surface Defect Database",
        "dataset_id": "kaustubhdikshit/neu-surface-defect-database",
        "type": "kaggle",
        "license": "Unknown",
        "size_mb": 100,
        "defect_types": [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches",
        ],
        "description": "Steel surface defect classification dataset",
    },
    "gc10_det": {
        "name": "GC10-DET Metal Surface Defects",
        "dataset_id": "alex000kim/gc10det",
        "type": "kaggle",
        "license": "CC BY 4.0",
        "size_mb": 919,
        "defect_types": [
            "punching_hole",
            "welding_line",
            "crescent_gap",
            "water_spot",
            "oil_spot",
            "silk_spot",
            "inclusion",
            "rolled_pit",
            "crease",
            "waist_folding",
        ],
        "description": "Large-scale metal surface defect detection dataset",
    },
    "magnetic_tile_defects": {
        "name": "Magnetic Tile Surface Defect",
        "url": "https://github.com/abin24/Magnetic-tile-defect-datasets/archive/refs/heads/master.zip",
        "type": "direct",
        "license": "Unknown",
        "size_mb": 48,
        "defect_types": ["blowhole", "break", "crack", "fray", "uneven", "free"],
        "description": "Magnetic tile defect dataset with pixel-level masks",
    },
    "sdnet2018": {
        "name": "SDNET2018 Concrete Crack Dataset",
        "url": "https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1047&context=all_datasets",
        "type": "manual",  # Requires manual download
        "license": "CC BY 4.0",
        "size_mb": 16000,
        "defect_types": ["cracks"],
        "description": "56,000 concrete surface crack images",
    },
    "tarros_dataset": {
        "name": "Tarros Dataset Final",
        "dataset_id": "sebastianfrancogomez/tarros-dataset-final-for-real",
        "type": "kaggle",
        "license": "Unknown",
        "size_mb": 200,  # Estimated
        "defect_types": ["container_defects"],
        "description": "Container/jar dataset - highly relevant for VIP plastic container inspection",
    },
}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download datasets for VIP benchmarking")
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate downloads without actually downloading"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )
    parser.add_argument("--only", type=str, help="Download only specific dataset")
    parser.add_argument("--data-dir", default="data", help="Base data directory (default: data)")
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for slug, info in DATASETS.items():
            license_note = f" [{info['license']}]" if info["license"] != "Unknown" else ""
            print(f"  {slug}: {info['name']}{license_note}")
            print(f"    Size: ~{info['size_mb']}MB, Types: {', '.join(info['defect_types'])}")
        return

    downloader = DatasetDownloader(args.data_dir, args.dry_run)
    registry = downloader.load_registry()

    # Filter datasets if --only specified
    datasets_to_process = DATASETS
    if args.only:
        if args.only not in DATASETS:
            print(f"Error: Dataset '{args.only}' not found")
            print("Available datasets:", list(DATASETS.keys()))
            sys.exit(1)
        datasets_to_process = {args.only: DATASETS[args.only]}

    success_count = 0
    total_count = len(datasets_to_process)

    for dataset_slug, dataset_info in datasets_to_process.items():
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_info['name']}")
        print(f"License: {dataset_info['license']}")
        print(f"{'='*60}")

        # Check if already downloaded
        if dataset_slug in registry and not args.force:
            print(f"✓ {dataset_slug} already downloaded (use --force to re-download)")
            success_count += 1
            continue

        dataset_dir = downloader.raw_dir / dataset_slug
        success = False
        error_msg = ""

        try:
            if dataset_info["type"] == "direct":
                # Direct download
                temp_file = tempfile.mktemp(suffix=".zip")
                success, error_msg = downloader.download_file(dataset_info["url"], temp_file)

                if success:
                    success, error_msg = downloader.extract_archive(temp_file, str(dataset_dir))
                    if not args.dry_run:
                        os.unlink(temp_file)  # Clean up temp file

            elif dataset_info["type"] == "kaggle":
                # Kaggle dataset
                success, error_msg = downloader.download_kaggle_dataset(
                    dataset_info["dataset_id"], str(dataset_dir)
                )

            elif dataset_info["type"] == "kaggle_competition":
                # Kaggle competition data
                print("⚠️  Kaggle competition data requires manual setup:")
                print(
                    f"   1. Join competition: https://www.kaggle.com/c/{dataset_info['dataset_id']}"
                )
                print(
                    f"   2. Download data: kaggle competitions download -c {dataset_info['dataset_id']}"
                )
                success = False
                error_msg = "Manual download required"

            elif dataset_info["type"] == "roboflow":
                print("⚠️  Roboflow dataset requires account:")
                print(f"   1. Create free account at: {dataset_info['url']}")
                print("   2. Download dataset in desired format")
                success = False
                error_msg = "Manual download required"

            elif dataset_info["type"] == "manual":
                print("⚠️  Manual download required:")
                print(f"   Visit: {dataset_info['url']}")
                success = False
                error_msg = "Manual download required"

            if success:
                # Create standardized structure
                downloader.create_dataset_structure(dataset_slug, str(dataset_dir))

                # Update registry
                registry_entry = {
                    "dataset": dataset_slug,
                    "version": None,
                    "source_url": dataset_info.get("url", dataset_info.get("dataset_id", "")),
                    "license": dataset_info["license"],
                    "downloaded_at": datetime.now().isoformat(),
                    "sha256": "computed_if_single_file",  # Would compute for single files
                    "status": "success",
                    "size_mb": dataset_info["size_mb"],
                    "defect_types": dataset_info["defect_types"],
                }

                downloader.update_registry(dataset_slug, registry_entry)
                print(f"✓ Successfully processed {dataset_slug}")
                success_count += 1

            else:
                print(f"✗ Failed to download {dataset_slug}: {error_msg}")

        except Exception as e:
            print(f"✗ Error processing {dataset_slug}: {e}")

    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{total_count} datasets processed successfully")

    if not args.dry_run and success_count > 0:
        print(f"Registry updated: {downloader.registry_file}")
        print(f"Raw data location: {downloader.raw_dir}")
        print("\nNext steps:")
        print("1. Create dataset manifests with: python scripts/create_manifests.py")
        print("2. Run benchmarks with: python scripts/bench_run.py")


if __name__ == "__main__":
    main()
