# VIP - Vision Inspection Pipeline

A comprehensive computer vision system for automated defect detection in manufacturing and quality control applications.

## 🎯 Features

- **Multi-Defect Detection**: Automatically detects cracks, scratches, contamination, discoloration, and flash defects
- **Real-time Processing**: Live camera feed with real-time defect detection and visualization
- **Web Interface**: Modern web application with intuitive UI for image upload and live monitoring
- **Color-coded Results**: Different colored bounding boxes for each defect type
- **High Accuracy**: Advanced computer vision algorithms with configurable confidence thresholds
- **RESTful API**: Complete FastAPI backend for integration with other systems

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenCV
- FastAPI
- uvicorn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vip.git
   cd vip
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the web application**
   ```bash
   python api/working_enhanced.py
   ```

4. **Open your browser**
   Navigate to `http://127.0.0.1:8000`

## 📁 Project Structure

```
vip/
├── src/vip/                    # Core detection pipeline
│   ├── detect/                 # Individual defect detectors
│   │   ├── cracks.py          # Crack detection algorithm
│   │   ├── scratches.py       # Scratch detection algorithm
│   │   ├── contamination.py   # Contamination detection
│   │   ├── discoloration.py   # Discoloration detection
│   │   └── flash.py           # Flash detection
│   ├── pipeline.py            # Main processing pipeline
│   ├── config.py              # Configuration management
│   └── utils/                 # Utility functions
├── api/                       # Web application
│   └── working_enhanced.py    # FastAPI server with web interface
├── input/                     # Sample images for testing
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎨 Defect Types & Colors

| Defect Type | Color | Description |
|-------------|-------|-------------|
| **Scratches** | 🟢 Green | Surface scratches and abrasions |
| **Cracks** | 🟡 Yellow | Structural cracks and fractures |
| **Contamination** | 🔴 Red | Foreign particles and dirt |
| **Discoloration** | 🔵 Blue | Color variations and stains |
| **Flash** | 🟣 Magenta | Excess material from molding |

## 🔧 API Endpoints

### Web Interface
- `GET /` - Main web interface
- `GET /health` - Health check endpoint

### Image Processing
- `POST /process` - Upload and process an image
  - **Input**: Multipart form with image file
  - **Output**: JSON with detection results and processed image

### Live Camera
- `POST /camera/start` - Start live camera feed
- `POST /camera/stop` - Stop live camera feed
- `GET /camera/frame` - Get latest processed frame

## 📊 Detection Results Format

```json
{
  "detections": [
    {
      "label": "crack",
      "score": 0.85,
      "bbox": {
        "x": 100,
        "y": 150,
        "width": 50,
        "height": 30
      }
    }
  ],
  "image_url": "data:image/jpeg;base64,/9j/4AAQ...",
  "detection_count": 1,
  "total_detections": 5,
  "confidence_threshold": 0.3
}
```

## ⚙️ Configuration

### Detection Thresholds
- **Confidence Threshold**: 0.3 (adjustable in code)
- **Minimum Bounding Box Size**: 15 pixels
- **Supported Image Formats**: JPG, PNG, BMP

### Camera Settings
- **Default Camera**: Index 0 (first available camera)
- **Frame Rate**: ~2 FPS for live processing
- **Resolution**: Auto-detected from camera

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 🚀 Deployment

### Development
```bash
uvicorn api.working_enhanced:app --reload --host 127.0.0.1 --port 8000
```

### Production
```bash
uvicorn api.working_enhanced:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📈 Performance

- **Processing Speed**: ~2-5 seconds per image (depending on size)
- **Detection Accuracy**: 85-95% (varies by defect type)
- **Memory Usage**: ~200-500MB (depending on image size)
- **Supported Resolutions**: Up to 4K images

## 🔍 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Ensure camera is connected and not used by other applications
   - Try different camera indices (0, 1, 2...)

2. **Low detection accuracy**
   - Adjust confidence threshold in the code
   - Ensure good lighting conditions
   - Use high-resolution images

3. **Server won't start**
   - Check if port 8000 is available
   - Install all dependencies: `pip install -r requirements.txt`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- FastAPI for the web framework
- The computer vision research community

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**VIP - Making Quality Control Smarter, Faster, and More Reliable** 🎯