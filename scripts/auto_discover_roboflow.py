#!/usr/bin/env python3
"""
Auto-discover Roboflow workspace and project from API key.
"""


def discover_roboflow_info(api_key: str):
    """Try to discover workspace and project info from API key."""
    try:
        import requests
        from roboflow import Roboflow

        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)

        # Try to get workspace info through the API
        print("🔍 Discovering your Roboflow workspace and projects...")

        # Method 1: Try to access default workspace
        try:
            workspace = rf.workspace()
            print("✅ Found workspace!")

            # Try to list projects
            try:
                # Get workspace name by inspecting the object
                workspace_name = getattr(workspace, "name", None) or getattr(workspace, "id", None)
                if workspace_name:
                    print(f"📁 Workspace: {workspace_name}")

                # Try to get projects list
                projects_list = []
                try:
                    # Different ways to access projects
                    if hasattr(workspace, "projects"):
                        projects_list = workspace.projects()
                    elif hasattr(workspace, "list_projects"):
                        projects_list = workspace.list_projects()
                except:
                    pass

                if projects_list:
                    print(f"📊 Found {len(projects_list)} projects:")
                    for i, project in enumerate(projects_list):
                        project_name = getattr(project, "name", f"project_{i}")
                        print(f"  {i+1}. {project_name}")

                        # Try to get project details
                        try:
                            versions = project.versions()
                            print(f"     Versions: {len(versions)}")
                            if versions:
                                latest_version = versions[-1]
                                print(f"     Latest: v{latest_version.version}")
                        except:
                            pass
                else:
                    print("📊 No projects found or couldn't list them")

            except Exception as e:
                print(f"Could not list projects: {e}")

        except Exception as e:
            print(f"Could not access workspace: {e}")

        # Method 2: Try API endpoint directly
        print("\n🌐 Trying direct API approach...")
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://api.roboflow.com/", headers=headers)
            if response.status_code == 200:
                print("✅ API endpoint accessible")
            else:
                print(f"❌ API response: {response.status_code}")
        except Exception as e:
            print(f"Direct API failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        return False


if __name__ == "__main__":
    api_key = "dnEo9Ba8KiaFJM4lamTg"
    discover_roboflow_info(api_key)
