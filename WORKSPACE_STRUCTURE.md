# VIP Workspace Structure

This document outlines the organized structure of the VIP Vision Inspection Pipeline workspace.

## 📁 Directory Structure

```
workVIP/
├── 📁 api/                          # API server implementation
│   ├── main.py                      # FastAPI server
│   └── static/                      # Static web assets
│
├── 📁 assets/                       # Static assets and media
│   ├── 📁 images/                   # Test images and sample data
│   │   ├── 274.jpg
│   │   └── test_image.jpg
│   ├── 📁 icons/                    # UI icons and graphics
│   └── 📁 fonts/                    # Custom fonts
│
├── 📁 bench/                        # Benchmarking framework
│   ├── __init__.py
│   └── benchcore/                   # Core benchmarking components
│       ├── adapters/                # Pipeline adapters
│       ├── eval/                    # Evaluation metrics
│       ├── hitl/                    # Human-in-the-loop
│       ├── techniques/              # Detection techniques
│       └── utils/                   # Benchmarking utilities
│
├── 📁 configs/                      # Configuration files
│   ├── tunables.yaml               # Main tunable parameters
│   └── bench/                      # Benchmarking configurations
│       ├── datasets/               # Dataset configurations
│       ├── experiments/            # Experiment configurations
│       └── techniques/             # Technique configurations
│
├── 📁 data/                         # Data storage
│   ├── download_registry.jsonl     # Dataset download registry
│   ├── 📁 processed/               # Processed datasets
│   │   ├── demo_dataset/
│   │   ├── demo_surface_defects/
│   │   ├── gc10_det/
│   │   ├── mvtec_ad/
│   │   ├── neu_surface_defects/
│   │   ├── roboflow_plastic_defects/
│   │   └── tarros_dataset/
│   ├── 📁 raw/                     # Raw datasets
│   └── README.md
│
├── 📁 docs/                         # Documentation
│   ├── CHANGELOG.md                # Project changelog
│   ├── README.md                   # Main project documentation
│   ├── README_BENCHMARKING.md      # Benchmarking documentation
│   ├── QUICK_START_FOR_SUPERVISORS.md
│   ├── TECHNICAL_IMPLEMENTATION_SUMMARY.md
│   ├── TECHNICAL_README.md
│   └── Vision Inspection Pipeline_ Improvement and Benchmarking Plan.docx
│
├── 📁 logs/                         # Log files and runtime data
│   └── bench_runs.jsonl            # Benchmark run logs
│
├── 📁 models/                       # Machine learning models (future)
│
├── 📁 reports/                      # Generated reports
│   ├── benchmarking_report.md
│   └── README.md
│
├── 📁 results/                      # Experiment results
│   └── bench_*/                    # Individual benchmark runs
│       ├── environment.json
│       ├── per_image.jsonl
│       ├── report.md
│       ├── results.json
│       ├── summary.csv
│       └── figs/                   # Generated figures
│
├── 📁 scripts/                      # Utility scripts
│   ├── auto_discover_roboflow.py
│   ├── bench_report.py
│   ├── bench_run.py
│   ├── create_manifests.py
│   ├── discover_roboflow_project.py
│   ├── download_datasets.py
│   ├── download_plastic_containers.py
│   ├── download_roboflow.py
│   ├── get_roboflow_projects.py
│   ├── roboflow_direct_download.py
│   └── test_roboflow_access.py
│
├── 📁 src/                          # Source code
│   └── vip/                         # VIP package
│       ├── __init__.py
│       ├── cli.py                   # Command line interface
│       ├── config_loader.py         # Configuration management
│       ├── config.py                # Configuration classes
│       ├── pipeline.py              # Main pipeline
│       ├── 📁 detect/               # Detection algorithms
│       │   ├── base.py              # Base detector class
│       │   ├── contamination.py     # Contamination detector
│       │   ├── cracks.py            # Crack detector
│       │   ├── discoloration.py     # Discoloration detector
│       │   ├── flash.py             # Flash detector
│       │   └── scratches.py         # Scratch detector
│       ├── 📁 io/                   # Input/output utilities
│       └── 📁 utils/                # General utilities
│
├── 📁 tests/                        # Test suite
│   ├── test_benchmarking.py
│   ├── test_detectors.py
│   └── test_pipeline.py
│
├── .gitignore                       # Git ignore rules
├── pyproject.toml                   # Python project configuration
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package setup
├── start_server.py                 # Server startup script
├── uv.lock                         # UV package lock file
└── WORKSPACE_STRUCTURE.md          # This file
```

## 🎯 Organization Principles

### **Separation of Concerns**
- **`/src/`**: Core application logic and algorithms
- **`/api/`**: Web API and server implementation
- **`/configs/`**: All configuration files centralized
- **`/docs/`**: All documentation in one place
- **`/assets/`**: Static files and media resources
- **`/data/`**: Raw and processed datasets
- **`/results/`**: Generated outputs and experiment results
- **`/logs/`**: Runtime logs and monitoring data

### **Scalability**
- **`/models/`**: Reserved for future ML model storage
- **`/bench/`**: Comprehensive benchmarking framework
- **`/scripts/`**: Utility and automation scripts
- **`/tests/`**: Comprehensive test coverage

### **Maintainability**
- Clear directory naming conventions
- Logical grouping of related files
- Easy navigation and file discovery
- Consistent structure across project

## 📋 File Categories

### **Configuration Files**
- `configs/tunables.yaml` - Main parameter configuration
- `configs/bench/` - Benchmarking configurations
- `pyproject.toml` - Python project settings
- `requirements.txt` - Dependencies

### **Documentation**
- `docs/CHANGELOG.md` - Project history and changes
- `docs/README.md` - Main project documentation
- `docs/TECHNICAL_*` - Technical documentation
- `docs/*.mdx` - Additional guides and references

### **Source Code**
- `src/vip/` - Main VIP package
- `src/vip/detect/` - Detection algorithms
- `src/vip/config_loader.py` - Configuration management
- `api/main.py` - Web API server

### **Data & Results**
- `data/` - Raw and processed datasets
- `results/` - Experiment outputs
- `logs/` - Runtime logs and monitoring
- `reports/` - Generated analysis reports

### **Assets & Media**
- `assets/images/` - Test images and samples
- `assets/icons/` - UI graphics and icons
- `assets/fonts/` - Custom typography

### **Utilities & Scripts**
- `scripts/` - Automation and utility scripts
- `tests/` - Test suite and validation
- `start_server.py` - Application startup

## 🔄 Migration Notes

### **Path Updates Required**
1. **Configuration Loading**: Updated to use `configs/tunables.yaml`
2. **Image Assets**: Moved from `input/` to `assets/images/`
3. **Documentation**: Centralized in `docs/` directory
4. **Logs**: Moved to dedicated `logs/` directory

### **Benefits of New Structure**
- **Cleaner Root Directory**: Only essential files at project root
- **Logical Grouping**: Related files organized together
- **Easy Navigation**: Clear directory hierarchy
- **Scalable Design**: Room for future expansion
- **Professional Layout**: Industry-standard organization

## 🚀 Usage

### **Starting the Server**
```bash
python start_server.py
```

### **Configuration Management**
```python
from src.vip.config_loader import get_config_loader
config = get_config_loader()  # Loads from configs/tunables.yaml
```

### **Accessing Assets**
```python
# Images are now in assets/images/
image_path = "assets/images/test_image.jpg"
```

### **Documentation**
- Main docs: `docs/README.md`
- Changelog: `docs/CHANGELOG.md`
- Technical docs: `docs/TECHNICAL_*.md`

This organized structure provides a professional, maintainable, and scalable foundation for the VIP Vision Inspection Pipeline project.
