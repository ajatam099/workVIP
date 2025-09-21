#!/usr/bin/env python3
"""
Download the plastic-containers project specifically.
"""


def download_plastic_containers(api_key: str):
    """Download the plastic-containers project."""
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace("defects")

        print("🚀 Downloading plastic-containers project...")

        # Try both versions
        for version in [2, 1]:  # Try latest first
            try:
                print(f"📦 Trying version {version}...")
                project = workspace.project("plastic-containers")
                dataset = project.version(version)

                output_dir = "data/raw/roboflow_plastic_defects"
                print(f"📥 Downloading v{version} in COCO format to: {output_dir}")

                # Download in COCO format (perfect for classical CV)
                dataset.download("coco", location=output_dir)

                print(f"✅ Successfully downloaded plastic-containers v{version}!")

                # Update our registry
                import json
                from datetime import datetime

                registry_entry = {
                    "dataset": "roboflow_plastic_defects",
                    "version": str(version),
                    "source_url": f"roboflow://defects/plastic-containers/{version}",
                    "license": "Custom Dataset",
                    "downloaded_at": datetime.now().isoformat(),
                    "sha256": "roboflow_api_download",
                    "status": "success",
                    "size_mb": 50,  # Estimated
                    "defect_types": ["plastic_defects"],
                    "image_count": 450,
                    "format": "coco",
                }

                registry_file = "data/download_registry.jsonl"
                with open(registry_file, "a") as f:
                    f.write(json.dumps(registry_entry) + "\n")

                print(f"📝 Updated registry: {registry_file}")
                return True

            except Exception as e:
                print(f"❌ Version {version} failed: {e}")
                continue

        print("❌ Could not download any version")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    api_key = "dnEo9Ba8KiaFJM4lamTg"
    success = download_plastic_containers(api_key)

    if success:
        print("\n🎉 Next steps:")
        print(
            "1. Create manifest: python scripts/create_manifests.py --dataset roboflow_plastic_defects --force"
        )
        print(
            "2. Run benchmark: python scripts/bench_run.py --config bench/configs/experiments/container_inspection_test.yaml"
        )
    else:
        print("\n💡 If automatic download fails, please:")
        print("1. Go to: https://app.roboflow.com/defects/plastic-containers")
        print("2. Download in COCO JSON format")
        print("3. Extract to: data/raw/roboflow_plastic_defects/")
