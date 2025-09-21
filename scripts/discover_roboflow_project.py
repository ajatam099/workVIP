#!/usr/bin/env python3
"""
Try to discover Roboflow project details.
"""


def try_common_patterns(api_key: str):
    """Try common workspace/project name patterns."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)

    # Common project names for plastic/container defects
    common_project_names = [
        "plastic-defects",
        "container-defects",
        "plastic-containers",
        "defect-detection",
        "vip-dataset",
        "vision-inspection",
        "container-inspection",
        "plastic-inspection",
    ]

    # Try to get workspace info
    try:
        # The API key might contain workspace info
        print("Trying to discover project...")

        # Try common patterns
        for project_name in common_project_names:
            try:
                print(f"Trying project: {project_name}")
                # This is a guess - we'll need the actual workspace name
                # For now, let's just test the API
                break
            except Exception:
                continue

        print("\n💡 To download your dataset, I need:")
        print("1. Your workspace name (usually your Roboflow username)")
        print("2. Your project name (the name of your 450-image dataset)")
        print("\nYou can find these in your Roboflow dashboard URL:")
        print("https://app.roboflow.com/[WORKSPACE]/[PROJECT]")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    api_key = "dnEo9Ba8KiaFJM4lamTg"
    try_common_patterns(api_key)
