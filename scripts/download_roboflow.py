#!/usr/bin/env python3
"""
Download custom Roboflow dataset using API key.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int = 1,
    format: str = "yolov5pytorch",
    output_dir: str = "data/raw/roboflow_plastic_defects",
):
    """
    Download dataset from Roboflow using API.

    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version (default: 1)
        format: Download format (default: yolov5pytorch)
        output_dir: Output directory
    """
    try:
        from roboflow import Roboflow

        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)

        # Get project
        project_obj = rf.workspace(workspace).project(project)

        # Get dataset version
        dataset = project_obj.version(version)

        # Download dataset
        print(f"Downloading {workspace}/{project} v{version} in {format} format...")
        dataset.download(format, location=output_dir)

        print(f"✅ Successfully downloaded to: {output_dir}")
        return True

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download custom Roboflow dataset")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace name")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version (default: 1)")
    parser.add_argument(
        "--format", default="yolov5pytorch", help="Download format (default: yolov5pytorch)"
    )
    parser.add_argument(
        "--output-dir", default="data/raw/roboflow_plastic_defects", help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Download dataset
    success = download_roboflow_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        format=args.format,
        output_dir=args.output_dir,
    )

    if success:
        print("\n🎉 Download complete!")
        print("Next steps:")
        print(
            "1. Create manifest: python scripts/create_manifests.py --dataset roboflow_plastic_defects"
        )
        print(
            "2. Run benchmark: python scripts/bench_run.py --config bench/configs/experiments/container_inspection_test.yaml"
        )
    else:
        print("\n❌ Download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
