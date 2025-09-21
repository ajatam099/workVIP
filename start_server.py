#!/usr/bin/env python3
"""
VIP Server Startup Script
Simple script to start the VIP web application server.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Start the VIP server with proper configuration."""
    print("🚀 Starting VIP Vision Inspection Pipeline Server...")
    print("=" * 60)

    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Check if API file exists
    api_file = project_root / "api" / "main.py"
    if not api_file.exists():
        print("❌ Error: API file not found at api/main.py")
        print("Please ensure you're running this from the project root directory.")
        sys.exit(1)

    print(f"📁 Project root: {project_root}")
    print(f"🔧 API file: {api_file}")
    print()

    try:
        # Start the server
        print("🌐 Starting server on http://127.0.0.1:8000")
        print("📱 Frontend: http://127.0.0.1:8000")
        print("📚 API Docs: http://127.0.0.1:8000/docs")
        print("🔍 Health Check: http://127.0.0.1:8000/health")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 60)

        # Run the server
        subprocess.run([sys.executable, str(api_file)], check=True)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
