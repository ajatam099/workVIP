"""Configuration loader for VIP tunable parameters."""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Loads and manages tunable parameters from YAML configuration."""
    
    def __init__(self, config_path: str = "configs/tunables.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Resolve path relative to project root
        if not Path(config_path).is_absolute():
            # Go up from src/vip/ to project root, then to config file
            project_root = Path(__file__).parent.parent.parent
            self.config_path = project_root / config_path
        else:
            self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get_global_params(self) -> Dict[str, Any]:
        """Get global pipeline parameters."""
        return self._config.get('global', {})
    
    def get_detector_params(self, detector_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific detector.
        
        Args:
            detector_name: Name of the detector (scratches, cracks, etc.)
            
        Returns:
            Dictionary of parameters for the detector
        """
        return self._config.get(detector_name, {})
    
    def get_morphology_params(self) -> Dict[str, Any]:
        """Get morphological operation parameters."""
        return self._config.get('morphology', {})
    
    def reload_config(self):
        """Reload configuration from file."""
        self._load_config()
    
    def get_all_params(self) -> Dict[str, Any]:
        """Get all configuration parameters."""
        return self._config


# Global configuration instance
_config_loader = None

def get_config_loader(config_path: str = "configs/tunables.yaml") -> ConfigLoader:
    """
    Get the global configuration loader instance.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader
