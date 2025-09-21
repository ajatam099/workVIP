#!/usr/bin/env python3
"""
VIP Setup Script - Quick setup for Vision Inspection Pipeline
For thesis evaluation and research use.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_status(message, status="info"):
    """Print colored status messages."""
    colors = {
        "info": "\033[94m",     # Blue
        "success": "\033[92m",  # Green
        "warning": "\033[93m",  # Yellow
        "error": "\033[91m",    # Red
        "reset": "\033[0m"      # Reset
    }
    
    symbols = {
        "info": "[INFO]",
        "success": "[SUCCESS]",
        "warning": "[WARNING]",
        "error": "[ERROR]"
    }
    
    print(f"{colors[status]}{symbols[status]} {message}{colors['reset']}")

def run_command(cmd, description):
    """Run a command and handle errors."""
    print_status(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print_status(f"{description} completed", "success")
            return True
        else:
            print_status(f"{description} failed: {result.stderr}", "error")
            return False
    except Exception as e:
        print_status(f"{description} error: {str(e)}", "error")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_status("Python 3.8+ required. Current version: {}.{}.{}".format(
            version.major, version.minor, version.micro), "error")
        return False
    
    print_status(f"Python {version.major}.{version.minor}.{version.micro} OK", "success")
    return True

def install_dependencies():
    """Install required dependencies."""
    if not os.path.exists("requirements.txt"):
        print_status("requirements.txt not found. Creating basic requirements...", "warning")
        requirements = [
            "opencv-python>=4.5.0",
            "numpy>=1.21.0",
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0",
            "pillow>=8.0.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "pyyaml>=5.4.0",
            "pytest>=6.2.0",
            "black>=21.0.0",
            "ruff>=0.1.0"
        ]
        
        with open("requirements.txt", "w") as f:
            f.write("\n".join(requirements))
    
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Installing dependencies")

def test_installation():
    """Test if VIP can be imported and basic functionality works."""
    test_commands = [
        (f"{sys.executable} -c \"from src.vip.pipeline import Pipeline; from src.vip.config import RunConfig; print('✅ VIP core modules loaded')\"", 
         "Testing VIP core import"),
        (f"{sys.executable} -m pytest tests/ -q --tb=no", 
         "Running test suite")
    ]
    
    all_passed = True
    for cmd, desc in test_commands:
        if not run_command(cmd, desc):
            all_passed = False
    
    return all_passed

def check_datasets():
    """Check if datasets are available."""
    data_dir = Path("data/raw")
    datasets = ["roboflow_plastic_defects", "neu_surface_defects", "gc10_det", "tarros_dataset"]
    
    available = []
    missing = []
    
    for dataset in datasets:
        if (data_dir / dataset).exists():
            available.append(dataset)
        else:
            missing.append(dataset)
    
    if available:
        print_status(f"Available datasets: {', '.join(available)}", "success")
    
    if missing:
        print_status(f"Missing datasets: {', '.join(missing)}", "warning")
        print_status("Run 'python scripts/download_datasets.py' to download missing datasets", "info")
    
    return len(available) > 0

def main():
    """Main setup function."""
    print("VIP - Vision Inspection Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print_status("Failed to install dependencies. Please check your Python environment.", "error")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print_status("Installation test failed. Please check error messages above.", "error")
        sys.exit(1)
    
    # Check datasets
    has_datasets = check_datasets()
    
    print("\n" + "=" * 50)
    print_status("VIP Setup Complete!", "success")
    
    print("\nNext Steps:")
    print("1. Test web interface: python start_server.py")
    print("2. Open browser to: http://localhost:8000")
    
    if not has_datasets:
        print("3. [Optional] Download datasets: python scripts/download_datasets.py")
        print("4. [Optional] Run benchmarks: python scripts/bench_run.py --config bench/configs/experiments/roboflow_plastic_test.yaml")
    else:
        print("3. Run benchmarks: python scripts/bench_run.py --config bench/configs/experiments/roboflow_plastic_test.yaml")
    
    print("\nDocumentation:")
    print("- README.md - Complete setup guide")
    print("- TECHNICAL_IMPLEMENTATION_SUMMARY.md - Technical details")
    print("- data/README.md - Dataset information")

if __name__ == "__main__":
    main()
