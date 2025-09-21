#!/usr/bin/env python3
"""
Direct download of Roboflow dataset with detailed debugging.
"""

from pathlib import Path


def download_with_debugging(api_key: str):
    """Download with detailed debugging."""
    try:
        from roboflow import Roboflow

        print("🔧 Initializing Roboflow...")
        rf = Roboflow(api_key=api_key)

        print("🔧 Accessing workspace 'defects'...")
        workspace = rf.workspace("defects")

        print("🔧 Accessing project 'plastic-containers'...")
        project = workspace.project("plastic-containers")

        print("🔧 Getting latest version...")
        dataset = project.version(2)  # Try version 2 since it exists

        # Create output directory
        output_dir = Path("data/raw/roboflow_plastic_defects")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔧 Downloading to: {output_dir.absolute()}")
        print("📥 Starting download in COCO format...")

        # Download with explicit format
        result = dataset.download(model_format="coco", location=str(output_dir), overwrite=True)

        print(f"✅ Download result: {result}")

        # Check what was downloaded
        print("\n📁 Checking downloaded files...")
        if output_dir.exists():
            files = list(output_dir.rglob("*"))
            print(f"Found {len(files)} files/directories:")
            for f in files[:10]:  # Show first 10
                print(f"  {f.relative_to(output_dir)}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    api_key = "dnEo9Ba8KiaFJM4lamTg"
    download_with_debugging(api_key)
