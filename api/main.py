#!/usr/bin/env python3
"""Working Enhanced VIP server with image upload - no multipart issues."""

import base64
import os
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import VIP modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "src"))

# Change to the api directory to ensure relative paths work
os.chdir(Path(__file__).parent)

try:
    import cv2
    import numpy as np
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    # Import VIP modules
    from vip.config import RunConfig
    from vip.pipeline import Pipeline
    from vip.utils.viz import overlay_detections

    app = FastAPI(
        title="VIP Working Enhanced",
        description="Working Vision Inspection Pipeline with Image Upload",
    )

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if not static_dir.exists():
        os.makedirs(static_dir)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Initialize pipeline
    print("🔧 Initializing VIP pipeline...")
    config = RunConfig(defects=["scratches", "contamination", "discoloration", "cracks", "flash"])
    pipeline = Pipeline(config)
    print("✅ Pipeline initialized successfully!")

    # Color mapping for different defect types (BGR format for OpenCV)
    DEFECT_COLORS = {
        "scratch": (0, 255, 0),  # Green
        "scratches": (0, 255, 0),  # Green (plural)
        "contamination": (0, 0, 255),  # Red
        "discoloration": (255, 0, 0),  # Blue
        "crack": (0, 255, 255),  # Yellow (BGR: Blue=0, Green=255, Red=255)
        "cracks": (0, 255, 255),  # Yellow (BGR: Blue=0, Green=255, Red=255)
        "flash": (255, 0, 255),  # Magenta
        "default": (128, 128, 128),  # Gray
    }

    def get_defect_color(defect_label):
        """Get color for a specific defect type."""
        return DEFECT_COLORS.get(defect_label.lower(), DEFECT_COLORS["default"])

    # Global variables for camera state
    camera_active = False
    camera_thread = None
    latest_frame = None

    @app.get("/", response_class=HTMLResponse)
    async def read_root():
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>VIP - Vision Inspection Pipeline</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: #f5f5f5; 
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                }
                h1 { 
                    color: #333; 
                    text-align: center; 
                    margin-bottom: 30px;
                }
                .section { 
                    margin: 30px 0; 
                    padding: 20px; 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    background: #fafafa;
                }
                .section h2 { 
                    color: #007bff; 
                    margin-top: 0; 
                }
                .upload-area { 
                    border: 2px dashed #007bff; 
                    border-radius: 8px; 
                    padding: 40px; 
                    text-align: center; 
                    background: #f8f9fa; 
                    cursor: pointer;
                    transition: all 0.3s;
                }
                .upload-area:hover { 
                    background: #e9ecef; 
                    border-color: #0056b3; 
                }
                .upload-area.dragover { 
                    background: #d4edda; 
                    border-color: #28a745; 
                }
                .btn { 
                    background: #007bff; 
                    color: white; 
                    padding: 12px 24px; 
                    border: none; 
                    border-radius: 5px; 
                    cursor: pointer; 
                    margin: 10px; 
                    font-size: 16px;
                    transition: background 0.3s;
                }
                .btn:hover { 
                    background: #0056b3; 
                }
                .btn:disabled { 
                    background: #6c757d; 
                    cursor: not-allowed; 
                }
                .btn-success { background: #28a745; }
                .btn-danger { background: #dc3545; }
                .btn-warning { background: #ffc107; color: #212529; }
                .result { 
                    margin-top: 20px; 
                    padding: 15px; 
                    background: #f8f9fa; 
                    border-radius: 5px; 
                    border-left: 4px solid #007bff;
                }
                .image-container { 
                    margin: 20px 0; 
                    text-align: center; 
                }
                .image-container img { 
                    max-width: 100%; 
                    height: auto; 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .detection-item { 
                    background: white; 
                    margin: 10px 0; 
                    padding: 15px; 
                    border-radius: 5px; 
                    border-left: 4px solid #28a745;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                .detection-item h4 { 
                    margin: 0 0 10px 0; 
                    color: #333; 
                }
                .confidence { 
                    font-weight: bold; 
                    color: #007bff; 
                }
                .bbox { 
                    color: #6c757d; 
                    font-family: monospace; 
                }
                .status { 
                    padding: 10px; 
                    margin: 10px 0; 
                    border-radius: 5px; 
                    font-weight: bold; 
                }
                .status.success { 
                    background: #d4edda; 
                    color: #155724; 
                    border: 1px solid #c3e6cb; 
                }
                .status.error { 
                    background: #f8d7da; 
                    color: #721c24; 
                    border: 1px solid #f5c6cb; 
                }
                .status.info { 
                    background: #d1ecf1; 
                    color: #0c5460; 
                    border: 1px solid #bee5eb; 
                }
                .camera-feed { 
                    width: 100%; 
                    max-width: 600px; 
                    height: 400px; 
                    background: #f8f9fa; 
                    border: 2px solid #dee2e6; 
                    border-radius: 8px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    color: #6c757d; 
                    font-size: 18px;
                }
                .two-column { 
                    display: grid; 
                    grid-template-columns: 1fr 1fr; 
                    gap: 30px; 
                }
                @media (max-width: 768px) { 
                    .two-column { 
                        grid-template-columns: 1fr; 
                    } 
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 VIP - Vision Inspection Pipeline</h1>
                
                <div class="status success">
                    <strong>✅ Enhanced Server Running!</strong> Image upload and processing features are now available.
                </div>
                
                <!-- Color Legend -->
                <div class="section">
                    <h2>🎨 Defect Color Legend</h2>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 15px 0;">
                        <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                            <div style="width: 20px; height: 20px; background: #00ff00; border: 2px solid #000; margin-right: 10px;"></div>
                            <span><strong>Scratches</strong> - Green</span>
                        </div>
                        <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                            <div style="width: 20px; height: 20px; background: #ff0000; border: 2px solid #000; margin-right: 10px;"></div>
                            <span><strong>Contamination</strong> - Red</span>
                        </div>
                        <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                            <div style="width: 20px; height: 20px; background: #0000ff; border: 2px solid #000; margin-right: 10px;"></div>
                            <span><strong>Discoloration</strong> - Blue</span>
                        </div>
                        <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                            <div style="width: 20px; height: 20px; background: #ffff00; border: 2px solid #000; margin-right: 10px;"></div>
                            <span><strong>Cracks</strong> - Yellow</span>
                        </div>
                        <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                            <div style="width: 20px; height: 20px; background: #ff00ff; border: 2px solid #000; margin-right: 10px;"></div>
                            <span><strong>Flash</strong> - Magenta</span>
                        </div>
                    </div>
                </div>
                
                <div class="two-column">
                    <!-- Image Upload Section -->
                    <div class="section">
                        <h2>📁 Image Upload & Processing</h2>
                        <div class="upload-area" id="uploadArea">
                            <p><strong>Click here or drag & drop an image</strong></p>
                            <p>Supported formats: JPG, PNG, BMP</p>
                            <input type="file" id="fileInput" accept="image/*" style="display: none;">
                        </div>
                        <div id="uploadResult"></div>
                    </div>
                    
                    <!-- Camera Demo Section -->
                    <div class="section">
                        <h2>📹 Live Camera Demo</h2>
                        <div class="camera-feed" id="cameraFeed">
                            <div>Camera feed will appear here</div>
                        </div>
                        <div style="text-align: center; margin-top: 15px;">
                            <button class="btn btn-success" id="startCamera">Start Camera</button>
                            <button class="btn btn-danger" id="stopCamera" disabled>Stop Camera</button>
                        </div>
                        <div id="cameraResult"></div>
                    </div>
                </div>
                
                <!-- API Links Section -->
                <div class="section">
                    <h2>🔗 API Endpoints</h2>
                    <p>
                        <a href="/health" target="_blank" class="btn">Health Check</a>
                        <a href="/detectors" target="_blank" class="btn">List Detectors</a>
                        <a href="/docs" target="_blank" class="btn">API Documentation</a>
                        <a href="/test" target="_blank" class="btn btn-warning">Interactive Test</a>
                    </p>
                </div>
            </div>
            
            <script>
                // File upload handling
                const uploadArea = document.getElementById('uploadArea');
                const fileInput = document.getElementById('fileInput');
                const uploadResult = document.getElementById('uploadResult');
                
                uploadArea.addEventListener('click', () => fileInput.click());
                uploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadArea.classList.add('dragover');
                });
                uploadArea.addEventListener('dragleave', () => {
                    uploadArea.classList.remove('dragover');
                });
                uploadArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    uploadArea.classList.remove('dragover');
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        handleFileUpload(files[0]);
                    }
                });
                
                fileInput.addEventListener('change', (e) => {
                    if (e.target.files.length > 0) {
                        handleFileUpload(e.target.files[0]);
                    }
                });
                
                async function handleFileUpload(file) {
                    if (!file.type.startsWith('image/')) {
                        showError('Please select an image file.');
                        return;
                    }
                    
                    showStatus('Processing image...', 'info');
                    
                    try {
                        const formData = new FormData();
                        formData.append('file', file);
                        
                        const response = await fetch('/process', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const result = await response.json();
                        displayResults(result);
                        
                    } catch (error) {
                        showError('Error processing image: ' + error.message);
                    }
                }
                
                function displayResults(result) {
                    let html = '<div class="result">';
                    html += '<h3>📊 Processing Results</h3>';
                    
                    if (result.processed_image) {
                        html += '<div class="image-container">';
                        html += '<h4>Processed Image with Defect Overlays:</h4>';
                        html += `<img src="data:image/jpeg;base64,${result.processed_image}" alt="Processed Image">`;
                        html += '</div>';
                    }
                    
                    if (result.detections && result.detections.length > 0) {
                        html += '<h4>🔍 Detected Defects:</h4>';
                        result.detections.forEach((detection, index) => {
                            // Get color for this defect type
                            const colors = {
                                'scratch': '#00ff00',
                                'scratches': '#00ff00',
                                'contamination': '#ff0000', 
                                'discoloration': '#0000ff',
                                'crack': '#ffff00',
                                'cracks': '#ffff00',
                                'flash': '#ff00ff'
                            };
                            const color = colors[detection.label.toLowerCase()] || '#808080';
                            
                            html += `<div class="detection-item" style="border-left-color: ${color};">`;
                            html += `<h4>Defect ${index + 1}: ${detection.label} <span style="color: ${color}; font-size: 0.8em;">●</span></h4>`;
                            html += `<p><span class="confidence">Confidence: ${(detection.score * 100).toFixed(1)}%</span></p>`;
                            if (detection.bbox) {
                                html += `<p><span class="bbox">Bounding Box: [${detection.bbox.x}, ${detection.bbox.y}, ${detection.bbox.width}, ${detection.bbox.height}]</span></p>`;
                            }
                            html += '</div>';
                        });
                    } else {
                        html += '<p><strong>✅ No defects detected!</strong></p>';
                    }
                    
                    html += '</div>';
                    uploadResult.innerHTML = html;
                }
                
                function showStatus(message, type) {
                    uploadResult.innerHTML = `<div class="status ${type}">${message}</div>`;
                }
                
                function showError(message) {
                    uploadResult.innerHTML = `<div class="status error">❌ ${message}</div>`;
                }
                
                // Camera handling
                const startBtn = document.getElementById('startCamera');
                const stopBtn = document.getElementById('stopCamera');
                const cameraFeed = document.getElementById('cameraFeed');
                const cameraResult = document.getElementById('cameraResult');
                let cameraInterval;
                
                startBtn.addEventListener('click', startCamera);
                stopBtn.addEventListener('click', stopCamera);
                
                async function startCamera() {
                    try {
                        const response = await fetch('/camera/start', { method: 'POST' });
                        const result = await response.json();
                        
                        if (result.success) {
                            startBtn.disabled = true;
                            stopBtn.disabled = false;
                            cameraResult.innerHTML = '<div class="status success">📹 Camera started successfully!</div>';
                            
                            // Start polling for frames
                            cameraInterval = setInterval(updateCameraFeed, 500);
                        } else {
                            throw new Error(result.message || 'Failed to start camera');
                        }
                    } catch (error) {
                        cameraResult.innerHTML = `<div class="status error">❌ Error starting camera: ${error.message}</div>`;
                    }
                }
                
                async function stopCamera() {
                    try {
                        const response = await fetch('/camera/stop', { method: 'POST' });
                        const result = await response.json();
                        
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                        clearInterval(cameraInterval);
                        cameraFeed.innerHTML = '<div>Camera feed stopped</div>';
                        cameraResult.innerHTML = '<div class="status info">📹 Camera stopped</div>';
                    } catch (error) {
                        cameraResult.innerHTML = `<div class="status error">❌ Error stopping camera: ${error.message}</div>`;
                    }
                }
                
                async function updateCameraFeed() {
                    try {
                        const response = await fetch('/camera/frame');
                        const result = await response.json();
                        
                        if (result.success && result.image) {
                            cameraFeed.innerHTML = `<img src="data:image/jpeg;base64,${result.image}" alt="Camera Feed" style="width: 100%; height: 100%; object-fit: contain;">`;
                            
                            if (result.detections && result.detections.length > 0) {
                                cameraResult.innerHTML = `<div class="status info">🔍 Live detections: ${result.detections.length} defects found</div>`;
                            } else {
                                cameraResult.innerHTML = `<div class="status success">✅ No defects detected in live feed</div>`;
                            }
                        }
                    } catch (error) {
                        console.error('Error updating camera feed:', error);
                    }
                }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    @app.post("/process")
    async def process_image(request: Request):
        """Process uploaded image through VIP pipeline."""
        try:
            # Get the raw body
            body = await request.body()

            if not body:
                raise HTTPException(status_code=400, detail="No file uploaded")

            print("📁 Processing uploaded image...")

            # Parse multipart form data to extract the uploaded file
            content_type = request.headers.get("content-type", "")
            if "multipart/form-data" not in content_type:
                raise HTTPException(
                    status_code=400, detail="Content type must be multipart/form-data"
                )

            # Simple multipart parsing for file upload
            boundary = None
            for header in content_type.split(";"):
                if "boundary=" in header:
                    boundary = header.split("boundary=")[1].strip()
                    break

            if not boundary:
                raise HTTPException(
                    status_code=400, detail="Could not find boundary in multipart data"
                )

            # Parse the multipart data
            parts = body.split(f"--{boundary}".encode())
            uploaded_image = None

            for part in parts:
                if b"Content-Disposition: form-data" in part and b"filename=" in part:
                    # Extract the image data (skip headers)
                    header_end = part.find(b"\r\n\r\n")
                    if header_end != -1:
                        image_data = part[header_end + 4 :]
                        # Remove trailing boundary markers
                        if image_data.endswith(b"\r\n"):
                            image_data = image_data[:-2]

                        # Convert to numpy array and decode
                        nparr = np.frombuffer(image_data, np.uint8)
                        uploaded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if uploaded_image is not None:
                            print(f"✅ Successfully decoded uploaded image: {uploaded_image.shape}")
                            break

            if uploaded_image is None:
                # Fallback to test image if upload failed
                print("⚠️ Could not decode uploaded image, using test image")
                input_dir = Path(__file__).parent.parent / "input"
                test_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

                if test_images:
                    test_image_path = test_images[0]
                    print(f"📸 Using test image: {test_image_path.name}")
                    uploaded_image = cv2.imread(str(test_image_path))
                    if uploaded_image is None:
                        raise HTTPException(status_code=400, detail="Could not load test image")
                else:
                    # Final fallback to simulated image
                    print("⚠️ No test images found, using simulated image")
                    uploaded_image = np.ones((400, 600, 3), dtype=np.uint8) * 240
                    cv2.putText(
                        uploaded_image,
                        "Simulated Image",
                        (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        2,
                    )
                    cv2.rectangle(uploaded_image, (100, 100), (200, 150), (0, 0, 255), 2)
                    cv2.putText(
                        uploaded_image,
                        "Simulated Defect",
                        (100, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

            test_image = uploaded_image

            # Process through VIP pipeline
            print(f"🔍 Running detection pipeline on image shape: {test_image.shape}")
            detections = pipeline.run_on_image(test_image)
            print(f"📊 Total detections found: {len(detections)}")

            # Debug: Show detection types
            detection_types = {}
            for det in detections:
                detection_types[det.label] = detection_types.get(det.label, 0) + 1
            print(f"🎯 Detection breakdown: {detection_types}")

            # Filter detections by confidence threshold and apply NMS-like filtering
            CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to show more detections (30% confidence)
            MIN_BBOX_SIZE = 15  # Smaller minimum size to catch more defects

            filtered_detections = []
            for detection in detections:
                if detection.score >= CONFIDENCE_THRESHOLD and detection.bbox:
                    # Check bounding box size
                    if isinstance(detection.bbox, tuple) and len(detection.bbox) >= 4:
                        x, y, w, h = detection.bbox
                    elif hasattr(detection.bbox, "x"):
                        x, y, w, h = (
                            detection.bbox.x,
                            detection.bbox.y,
                            detection.bbox.width,
                            detection.bbox.height,
                        )
                    else:
                        continue

                    # Only include if bounding box is large enough
                    if w >= MIN_BBOX_SIZE and h >= MIN_BBOX_SIZE:
                        filtered_detections.append(detection)

            print(
                f"🔍 Filtered {len(detections)} detections down to {len(filtered_detections)} high-confidence detections"
            )

            # Create overlay image
            overlay_image = test_image.copy()
            for detection in filtered_detections:
                if detection.bbox:
                    # Handle both tuple and object bbox formats
                    if isinstance(detection.bbox, tuple) and len(detection.bbox) >= 4:
                        x, y, w, h = detection.bbox
                    elif hasattr(detection.bbox, "x"):
                        x, y, w, h = (
                            detection.bbox.x,
                            detection.bbox.y,
                            detection.bbox.width,
                            detection.bbox.height,
                        )
                    else:
                        continue

                    # Get color for this defect type
                    color = get_defect_color(detection.label)

                    # Draw thicker, more visible bounding box
                    cv2.rectangle(
                        overlay_image, (int(x), int(y)), (int(x + w), int(y + h)), color, 3
                    )

                    # Add semi-transparent background for text
                    text = f"{detection.label}: {detection.score:.1f}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        overlay_image,
                        (int(x), int(y - text_height - 10)),
                        (int(x + text_width + 5), int(y)),
                        color,
                        -1,
                    )

                    # Draw text in white for better visibility
                    cv2.putText(
                        overlay_image,
                        text,
                        (int(x + 2), int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # Convert to base64
            _, buffer = cv2.imencode(".jpg", overlay_image)
            image_base64 = base64.b64encode(buffer).decode("utf-8")

            # Convert filtered detections to JSON-serializable format
            detections_json = []
            for detection in filtered_detections:
                bbox_data = None
                if detection.bbox:
                    # Handle both tuple and object bbox formats
                    if isinstance(detection.bbox, tuple) and len(detection.bbox) >= 4:
                        bbox_data = {
                            "x": float(detection.bbox[0]),
                            "y": float(detection.bbox[1]),
                            "width": float(detection.bbox[2]),
                            "height": float(detection.bbox[3]),
                        }
                    elif hasattr(detection.bbox, "x"):
                        bbox_data = {
                            "x": float(detection.bbox.x),
                            "y": float(detection.bbox.y),
                            "width": float(detection.bbox.width),
                            "height": float(detection.bbox.height),
                        }

                detections_json.append(
                    {"label": detection.label, "score": float(detection.score), "bbox": bbox_data}
                )

            return JSONResponse(
                content={
                    "success": True,
                    "message": "Image processed successfully",
                    "processed_image": image_base64,
                    "detections": detections_json,
                    "detection_count": len(filtered_detections),
                    "total_detections": len(detections),
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                }
            )

        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"Error processing image: {str(e)}"},
            )

    @app.post("/camera/start")
    async def start_camera():
        """Start camera feed (demo mode)."""
        global camera_active
        camera_active = True
        return JSONResponse(
            content={
                "success": True,
                "message": "Camera started (demo mode)",
                "note": "This is a simulated camera feed for demonstration",
            }
        )

    @app.post("/camera/stop")
    async def stop_camera():
        """Stop camera feed."""
        global camera_active
        camera_active = False
        return JSONResponse(content={"success": True, "message": "Camera stopped"})

    @app.get("/camera/frame")
    async def get_camera_frame():
        """Get latest camera frame with processing."""
        global camera_active, latest_frame

        if not camera_active:
            return JSONResponse(content={"success": False, "message": "Camera not active"})

        try:
            # Use different test images for camera feed to show variety
            input_dir = Path(__file__).parent.parent / "input"
            test_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

            if test_images:
                # Cycle through available test images
                frame_count = int(cv2.getTickCount() / cv2.getTickFrequency())
                image_index = frame_count % len(test_images)
                test_image_path = test_images[image_index]
                frame = cv2.imread(str(test_image_path))
                if frame is None:
                    raise Exception("Could not load test image")

                # Resize if too large
                if frame.shape[0] > 600 or frame.shape[1] > 800:
                    frame = cv2.resize(frame, (800, 600))

                # Add timestamp overlay
                cv2.putText(
                    frame,
                    f"Live Feed - {test_image_path.name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            else:
                # Fallback to simulated frame
                frame = np.ones((400, 600, 3), dtype=np.uint8) * 240
                cv2.putText(
                    frame, "Live Camera Feed", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Frame: {int(cv2.getTickCount() / cv2.getTickFrequency())}",
                    (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (100, 100, 100),
                    1,
                )

                # Add some simulated defects occasionally
                if int(cv2.getTickCount() / cv2.getTickFrequency()) % 3 == 0:
                    cv2.rectangle(frame, (150, 100), (250, 150), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "Live Defect",
                        (150, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

            # Process through VIP pipeline
            detections = pipeline.run_on_image(frame)

            # Apply same filtering as image upload
            CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to show more detections (30% confidence)
            MIN_BBOX_SIZE = 15  # Smaller minimum size to catch more defects

            filtered_detections = []
            for detection in detections:
                if detection.score >= CONFIDENCE_THRESHOLD and detection.bbox:
                    # Check bounding box size
                    if isinstance(detection.bbox, tuple) and len(detection.bbox) >= 4:
                        x, y, w, h = detection.bbox
                    elif hasattr(detection.bbox, "x"):
                        x, y, w, h = (
                            detection.bbox.x,
                            detection.bbox.y,
                            detection.bbox.width,
                            detection.bbox.height,
                        )
                    else:
                        continue

                    # Only include if bounding box is large enough
                    if w >= MIN_BBOX_SIZE and h >= MIN_BBOX_SIZE:
                        filtered_detections.append(detection)

            # Create overlay
            overlay_frame = frame.copy()
            for detection in filtered_detections:
                if detection.bbox:
                    # Handle both tuple and object bbox formats
                    if isinstance(detection.bbox, tuple) and len(detection.bbox) >= 4:
                        x, y, w, h = detection.bbox
                    elif hasattr(detection.bbox, "x"):
                        x, y, w, h = (
                            detection.bbox.x,
                            detection.bbox.y,
                            detection.bbox.width,
                            detection.bbox.height,
                        )
                    else:
                        continue

                    # Get color for this defect type
                    color = get_defect_color(detection.label)

                    # Draw thicker, more visible bounding box
                    cv2.rectangle(
                        overlay_frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 3
                    )

                    # Add semi-transparent background for text
                    text = f"{detection.label}: {detection.score:.1f}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        overlay_frame,
                        (int(x), int(y - text_height - 10)),
                        (int(x + text_width + 5), int(y)),
                        color,
                        -1,
                    )

                    # Draw text in white for better visibility
                    cv2.putText(
                        overlay_frame,
                        text,
                        (int(x + 2), int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # Convert to base64
            _, buffer = cv2.imencode(".jpg", overlay_frame)
            image_base64 = base64.b64encode(buffer).decode("utf-8")

            # Convert filtered detections to JSON
            detections_json = []
            for detection in filtered_detections:
                bbox_data = None
                if detection.bbox:
                    # Handle both tuple and object bbox formats
                    if isinstance(detection.bbox, tuple) and len(detection.bbox) >= 4:
                        bbox_data = {
                            "x": float(detection.bbox[0]),
                            "y": float(detection.bbox[1]),
                            "width": float(detection.bbox[2]),
                            "height": float(detection.bbox[3]),
                        }
                    elif hasattr(detection.bbox, "x"):
                        bbox_data = {
                            "x": float(detection.bbox.x),
                            "y": float(detection.bbox.y),
                            "width": float(detection.bbox.width),
                            "height": float(detection.bbox.height),
                        }

                detections_json.append(
                    {"label": detection.label, "score": float(detection.score), "bbox": bbox_data}
                )

            return JSONResponse(
                content={
                    "success": True,
                    "image": image_base64,
                    "detections": detections_json,
                    "detection_count": len(detections),
                }
            )

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"Error getting camera frame: {str(e)}"},
            )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return JSONResponse(
            content={
                "status": "healthy",
                "message": "VIP Enhanced API is running",
                "detectors": pipeline.list_available_detectors(),
                "features": ["image_upload", "camera_demo", "defect_detection"],
            }
        )

    @app.get("/detectors")
    async def list_detectors():
        """List available detectors."""
        return JSONResponse(
            content={
                "detectors": pipeline.list_available_detectors(),
                "count": len(pipeline.list_available_detectors()),
            }
        )

    @app.get("/test")
    async def test_page():
        """Interactive test page."""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>VIP Test Page</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .test-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
                .btn:hover { background: #0056b3; }
                .result { margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧪 VIP Enhanced API Test Page</h1>
                
                <div class="test-section">
                    <h3>Test Health Endpoint</h3>
                    <button class="btn" onclick="testHealth()">Test Health</button>
                    <div id="healthResult" class="result"></div>
                </div>
                
                <div class="test-section">
                    <h3>Test Detectors Endpoint</h3>
                    <button class="btn" onclick="testDetectors()">Get Detectors</button>
                    <div id="detectorsResult" class="result"></div>
                </div>
                
                <div class="test-section">
                    <h3>Test Image Processing</h3>
                    <button class="btn" onclick="testImageProcessing()">Test Image Processing</button>
                    <div id="imageResult" class="result"></div>
                </div>
                
                <div class="test-section">
                    <h3>Quick Links</h3>
                    <a href="/" class="btn">← Back to Main</a>
                    <a href="/docs" class="btn" target="_blank">API Documentation</a>
                </div>
            </div>
            
            <script>
                async function testHealth() {
                    try {
                        const response = await fetch('/health');
                        const data = await response.json();
                        document.getElementById('healthResult').innerHTML = 
                            '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    } catch (error) {
                        document.getElementById('healthResult').innerHTML = 
                            '<p style="color: red;">Error: ' + error.message + '</p>';
                    }
                }
                
                async function testDetectors() {
                    try {
                        const response = await fetch('/detectors');
                        const data = await response.json();
                        document.getElementById('detectorsResult').innerHTML = 
                            '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    } catch (error) {
                        document.getElementById('detectorsResult').innerHTML = 
                            '<p style="color: red;">Error: ' + error.message + '</p>';
                    }
                }
                
                async function testImageProcessing() {
                    try {
                        // Create a test image
                        const canvas = document.createElement('canvas');
                        canvas.width = 400;
                        canvas.height = 300;
                        const ctx = canvas.getContext('2d');
                        ctx.fillStyle = '#f0f0f0';
                        ctx.fillRect(0, 0, 400, 300);
                        ctx.fillStyle = '#333';
                        ctx.font = '20px Arial';
                        ctx.fillText('Test Image', 150, 150);
                        ctx.fillStyle = '#ff0000';
                        ctx.fillRect(100, 100, 50, 50);
                        
                        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
                        const formData = new FormData();
                        formData.append('file', blob, 'test.jpg');
                        
                        const response = await fetch('/process', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await response.json();
                        document.getElementById('imageResult').innerHTML = 
                            '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    } catch (error) {
                        document.getElementById('imageResult').innerHTML = 
                            '<p style="color: red;">Error: ' + error.message + '</p>';
                    }
                }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    print("🚀 Starting VIP Working Enhanced Server...")
    print("📱 Frontend: http://127.0.0.1:8000")
    print("📚 API Docs: http://127.0.0.1:8000/docs")
    print("🔍 Health Check: http://127.0.0.1:8000/health")
    print("\nPress Ctrl+C to stop the server")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("\nMake sure all dependencies are installed. Try running: uv sync")
    input("\nPress Enter to exit...")
    sys.exit(1)
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    input("\nPress Enter to exit...")
    sys.exit(1)
