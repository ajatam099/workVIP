#!/usr/bin/env python3
"""
Get detailed Roboflow project information.
"""


def get_project_details(api_key: str):
    """Get detailed project information."""
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace("defects")  # Use discovered workspace name

        print("🔍 Getting project details from workspace 'defects'...")

        # Try to access projects by common names
        common_names = [
            "plastic-defects",
            "container-defects",
            "vip-dataset",
            "defect-detection",
            "plastic-containers",
            "vision-inspection",
        ]

        found_projects = []

        for project_name in common_names:
            try:
                project = workspace.project(project_name)
                print(f"✅ Found project: {project_name}")

                # Get versions
                try:
                    versions = project.versions()
                    print(f"   📊 Versions: {[v.version for v in versions]}")

                    # Get latest version details
                    if versions:
                        latest = versions[-1]
                        print(f"   📈 Latest: v{latest.version}")

                        # Try to get image count
                        try:
                            # This might give us image count
                            print("   🖼️  Images: Checking...")
                        except:
                            pass

                except Exception as e:
                    print(f"   ⚠️  Could not get versions: {e}")

                found_projects.append(project_name)

            except Exception:
                # Project doesn't exist with this name
                continue

        if found_projects:
            print(f"\n🎉 Found {len(found_projects)} projects: {found_projects}")

            # Try to download the first one
            if found_projects:
                project_name = found_projects[0]
                print(f"\n🚀 Attempting to download: {project_name}")

                try:
                    project = workspace.project(project_name)
                    dataset = project.version(1)  # Try version 1

                    output_dir = "data/raw/roboflow_plastic_defects"
                    print(f"📥 Downloading to: {output_dir}")

                    dataset.download("coco", location=output_dir)
                    print("✅ Download successful!")
                    return True

                except Exception as e:
                    print(f"❌ Download failed: {e}")
                    return False
        else:
            print("\n❌ No projects found with common names")
            print("Please provide your exact project name")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    api_key = "dnEo9Ba8KiaFJM4lamTg"
    get_project_details(api_key)
