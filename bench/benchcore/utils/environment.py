"""Environment capture and seeding utilities."""

import os
import sys
import platform
import subprocess
import random
import json
from typing import Dict, Any
from datetime import datetime
import numpy as np


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    # Set environment variable for CUBLAS determinism
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_git_info() -> Dict[str, str]:
    """
    Get current git information.
    
    Returns:
        Dictionary with git commit, branch, and status info
    """
    try:
        # Get current commit hash
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        # Get current branch
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL, 
            text=True
        ).strip()
        
        # Check if working directory is clean
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        return {
            'commit': commit,
            'branch': branch,
            'is_clean': len(status) == 0,
            'status': status
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            'commit': 'unknown',
            'branch': 'unknown', 
            'is_clean': False,
            'status': 'git not available'
        }


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary with system details
    """
    info = {
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'python_executable': sys.executable
    }
    
    # Add CPU count if available
    try:
        info['cpu_count'] = os.cpu_count()
    except:
        info['cpu_count'] = 'unknown'
    
    return info


def get_package_versions() -> Dict[str, str]:
    """
    Get versions of key packages.
    
    Returns:
        Dictionary with package versions
    """
    packages = ['numpy', 'opencv-python', 'scikit-learn', 'pydantic', 'fastapi']
    versions = {}
    
    for package in packages:
        try:
            if package == 'opencv-python':
                import cv2
                versions[package] = cv2.__version__
            else:
                module = __import__(package.replace('-', '_'))
                versions[package] = getattr(module, '__version__', 'unknown')
        except ImportError:
            versions[package] = 'not installed'
    
    return versions


def capture_environment() -> Dict[str, Any]:
    """
    Capture complete environment information.
    
    Returns:
        Dictionary with all environment details
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'git': get_git_info(),
        'system': get_system_info(),
        'packages': get_package_versions()
    }


def save_environment_info(output_dir: str) -> str:
    """
    Save environment information to file.
    
    Args:
        output_dir: Directory to save environment info
        
    Returns:
        Path to saved environment file
    """
    env_info = capture_environment()
    env_file = os.path.join(output_dir, 'environment.json')
    
    os.makedirs(output_dir, exist_ok=True)
    with open(env_file, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    return env_file


def generate_run_id(prefix: str = "") -> str:
    """
    Generate a unique run ID.
    
    Args:
        prefix: Optional prefix for the run ID
        
    Returns:
        Unique run identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        return f"{prefix}_{timestamp}"
    return timestamp
