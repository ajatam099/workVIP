# Vision Inspection Pipeline (VIP)

A small but extensible image-processing pipeline for detecting defects on plastic workpieces from still images. Built with classical computer vision techniques, this pipeline provides a solid foundation for quality control applications.

## Features

- **Multiple Defect Detectors**: 5 built-in detectors for common manufacturing defects
- **Extensible Architecture**: Easy to add new detectors without modifying existing code
- **Rich CLI Interface**: Command-line tools with progress bars and formatted output
- **Flexible Configuration**: Configurable input/output, defect types, and image processing
- **Evaluation Tools**: Built-in evaluation against the Kaggle Tarros dataset
- **Professional Quality**: Type hints, comprehensive testing, and modern Python practices

## Defect Detectors

1. **Scratches** - Uses Canny edge detection and morphology to find elongated surface defects
2. **Contamination** - High-pass filtering to detect blotches, dust, and foreign materials
3. **Discoloration** - LAB color space analysis to find color variations from local means
4. **Cracks** - Ridge/edge emphasis with Sobel operators for thin crack detection
5. **Flash** - Gradient and brightness analysis for excess material detection

## Project Structure

```
inspection-pipeline/
├── README.md
├── pyproject.toml            # uv configuration
├── requirements.txt          # pip dependencies
├── .gitignore
├── input/                    # user-provided images
├── output/                   # overlays + json results
├── data/
│  └── datasets/
│     └── kaggle_tarros/      # optional local copy of Kaggle data
├── src/
│  └── vip/                   # vip = vision inspection pipeline
│     ├── __init__.py
│     ├── cli.py              # Typer entrypoints
│     ├── pipeline.py         # orchestrates processing
│     ├── config.py           # Pydantic config
│     ├── io/
│     │  ├── loader.py
│     │  └── writer.py
│     ├── detect/
│     │  ├── base.py
│     │  ├── scratches.py
│     │  ├── contamination.py
│     │  ├── discoloration.py
│     │  ├── cracks.py
│     │  └── flash.py
│     └── utils/
│        ├── image.py
│        └── viz.py
└── tests/
   ├── test_pipeline.py
   └── test_detectors.py
```

## Setup

### Prerequisites

- Python 3.11 or 3.12
- Windows, macOS, or Linux

### Option 1: Using uv (Recommended)

```bash
# Install uv if you don't have it
pip install uv

# Clone and setup
git clone <repository-url>
cd inspection-pipeline

# Install dependencies
uv sync

# Verify installation
uv run vip --help
```

### Option 2: Using pip

```bash
# Clone repository
git clone <repository-url>
cd inspection-pipeline

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m vip.cli --help
```

## Quick Start

1. **Place your images** in the `input/` directory
2. **Run detection**:
   ```bash
   # Using uv
   uv run vip run
   
   # Using pip
   python -m vip.cli run
   ```
3. **Check results** in the `output/` directory:
   - `*_overlay.jpg` - Images with detected defects highlighted
   - `*.json` - Detailed detection results

## Usage Examples

### Basic Defect Detection

```bash
# Run with default settings (all defect types)
uv run vip run

# Specify specific defect types
uv run vip run --defects scratches,contamination

# Custom input/output directories
uv run vip run --input my_images --output my_results

# Resize images for faster processing
uv run vip run --resize-width 1024
```

### Advanced Configuration

```bash
# Save only overlays (no JSON)
uv run vip run --no-save-json

# Save only JSON results (no overlays)
uv run vip run --no-save-overlay

# Process with specific defect subset
uv run vip run --defects scratches,cracks --resize-width 800
```

### List Available Detectors

```bash
uv run vip list-detectors
```

## Kaggle Dataset Evaluation

This pipeline includes evaluation tools for the [Tarros Dataset](https://www.kaggle.com/datasets/sebastianfrancogomez/tarros-dataset-final-for-real/data), a collection of plastic workpiece images with defect annotations.

### Dataset Structure

Download and organize the dataset as follows:
```
data/datasets/kaggle_tarros/
├── 1_front/
│   ├── test/
│   │   ├── 1_true/      # Defective images
│   │   └── 2_false/     # Non-defective images
│   └── train/            # Training data (if needed)
└── 2_back/               # Back view data
```

### Run Evaluation

```bash
# Evaluate on default test set
uv run vip eval-simple

# Evaluate on custom dataset path
uv run vip eval-simple --dataset-root data/datasets/kaggle_tarros/2_back/test
```

### Sample Evaluation Output

```json
{
  "dataset": "data/datasets/kaggle_tarros/1_front/test",
  "metrics": {
    "true_positives": 45,
    "false_positives": 12,
    "true_negatives": 38,
    "false_negatives": 5,
    "precision": 0.7895,
    "recall": 0.9000,
    "f1_score": 0.8407
  },
  "total_images": 100
}
```

## Extending the Pipeline

### Adding a New Detector

1. **Create detector class** in `src/vip/detect/`:
   ```python
   from .base import BaseDetector, Detection
   
   class MyDetector(BaseDetector):
       def __init__(self, name: str = "my_defect"):
           super().__init__(name)
       
       def detect(self, image: np.ndarray) -> List[Detection]:
           # Your detection logic here
           # Return list of Detection objects
           pass
   ```

2. **Register in pipeline** by adding to `DETECTOR_REGISTRY` in `pipeline.py`:
   ```python
   DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {
       # ... existing detectors ...
       "my_defect": MyDetector,
   }
   ```

3. **Use in CLI**:
   ```bash
   uv run vip run --defects scratches,my_defect
   ```

### Detector Requirements

- Inherit from `BaseDetector`
- Implement `detect(image)` method returning `List[Detection]`
- Return `Detection` objects with valid scores (0.0 to 1.0)
- Handle errors gracefully (pipeline will continue with other detectors)

## Development

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Run tests
uv run pytest tests/
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_detectors.py
```

## Troubleshooting

### Common Issues

1. **OpenCV import errors**: Use `opencv-python-headless` for headless environments
2. **Memory issues**: Use `--resize-width` to reduce image size
3. **No detections**: Adjust detector parameters or check image quality

### Performance Tips

- Use `--resize-width` for large images
- Process images in batches for large datasets
- Consider GPU acceleration for OpenCV operations

## Headless Environment Notes

For servers or CI/CD environments without display:

```bash
# Install headless OpenCV
pip install opencv-python-headless

# Or update requirements.txt
echo "opencv-python-headless>=4.8.0" > requirements.txt
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **Kaggle Tarros Dataset**: Sebastian Franco Gomez for the comprehensive defect dataset
- **OpenCV**: Computer vision library providing core image processing capabilities
- **scikit-image**: Advanced image processing algorithms
- **Pydantic**: Data validation and settings management
- **Typer**: Modern CLI framework built on top of Click

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure code passes linting and formatting
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review existing GitHub issues
- Create a new issue with detailed information

---

**Built with ❤️ for quality control and computer vision enthusiasts**
