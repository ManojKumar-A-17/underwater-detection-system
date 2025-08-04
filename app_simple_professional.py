import gradio as gr
from PIL import Image
from ultralytics import YOLO
import os
import json
import numpy as np
import cv2
import time
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Model paths
MODEL_PATHS = {
    "YOLOv8n": "../yolov8n/runs/detect_train/weights/best.pt",
    "YOLOv8s": "../yolov8s/runs/detect_train/weights/best.pt"
}

# Load models
models = {}
available_models = []

for model_name, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path):
        try:
            models[model_name] = YOLO(model_path)
            available_models.append(model_name)
            print(f"✅ Loaded {model_name} model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load {model_name} model: {e}")
    else:
        print(f"⚠️ {model_name} model not found: {model_path}")

# Fallback to default model if no custom models found
if not available_models:
    models["YOLOv8n"] = YOLO("yolov8n.pt")
    models["YOLOv8s"] = YOLO("yolov8s.pt")
    available_models = ["YOLOv8n", "YOLOv8s"]
    print("⚠️ Using default YOLO models")

# Underwater object classes
UNDERWATER_CLASSES = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']

# Class colors for visualization
CLASS_COLORS = {
    'fish': '#FF6B6B',
    'jellyfish': '#4ECDC4', 
    'penguin': '#45B7D1',
    'puffin': '#96CEB4',
    'shark': '#FECA57',
    'starfish': '#FF9FF3',
    'stingray': '#54A0FF'
}

# Detection history storage
detection_history = []

def create_visualization(image, results, model_name, inference_time):
    """Create detection visualization with enhanced graphics"""
    img_array = np.array(image)
    img_height, img_width = img_array.shape[:2]
    
    # Create a copy for annotation
    annotated_img = img_array.copy()
    
    detections = results[0]
    detection_data = []
    
    if detections.boxes is not None:
        for i, box in enumerate(detections.boxes):
            if box.conf is not None and box.cls is not None:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get class name
                if class_id < len(UNDERWATER_CLASSES):
                    class_name = UNDERWATER_CLASSES[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get color for this class
                color = CLASS_COLORS.get(class_name, '#FFFFFF')
                color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                
                # Draw bounding box
                thickness = max(2, int(min(img_width, img_height) / 300))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color_rgb, thickness)
                
                # Create label
                label = f"{class_name} {confidence:.2f}"
                
                # Calculate label size
                font_scale = max(0.6, min(img_width, img_height) / 1000)
                font_thickness = max(1, thickness // 2)
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                
                # Draw label background
                label_bg_top = max(0, y1 - label_h - 10)
                cv2.rectangle(annotated_img, (x1, label_bg_top), (x1 + label_w + 10, y1), color_rgb, -1)
                
                # Add text
                cv2.putText(annotated_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 255, 255), font_thickness)
                
                detection_data.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'color': color
                })
    
    return Image.fromarray(annotated_img), detection_data

def create_detection_summary(detection_data, model_name, inference_time):
    """Create detection summary"""
    if not detection_data:
        return "🔍 **No objects detected**\n\nTry adjusting the confidence threshold or uploading a different image."
    
    # Count detections by class
    class_counts = {}
    total_detections = len(detection_data)
    avg_confidence = sum(d['confidence'] for d in detection_data) / total_detections
    
    for detection in detection_data:
        class_name = detection['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Create summary
    summary = f"## 🎯 Detection Summary\n"
    summary += f"**Model Used:** {model_name}\n"
    summary += f"**Inference Time:** {inference_time:.2f}s\n"
    summary += f"**Total Objects:** {total_detections}\n"
    summary += f"**Average Confidence:** {avg_confidence:.2f}\n\n"
    
    summary += "### 📊 Detected Objects:\n"
    for class_name, count in class_counts.items():
        emoji = {"fish": "🐠", "jellyfish": "🪼", "penguin": "🐧", 
                "puffin": "🦅", "shark": "🦈", "starfish": "⭐", "stingray": "🐙"}.get(class_name, "🔹")
        summary += f"{emoji} **{class_name.capitalize()}:** {count}\n"
    
    return summary

def create_detection_chart(detection_data):
    """Create detection statistics chart"""
    if not detection_data:
        return None
    
    # Count detections by class
    class_counts = {}
    for detection in detection_data:
        class_name = detection['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Create bar chart
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [CLASS_COLORS.get(cls, '#888888') for cls in classes]
    
    fig = go.Figure(data=[
        go.Bar(x=classes, y=counts, marker_color=colors, text=counts, textposition='auto')
    ])
    
    fig.update_layout(
        title="Detection Statistics",
        xaxis_title="Object Classes",
        yaxis_title="Count",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig

def detect_objects(image, model_choice, confidence_threshold=0.25, iou_threshold=0.45):
    """Object detection with enhanced features"""
    if image is None:
        return None, "Please upload an image", None, ""
    
    if model_choice not in models:
        return None, f"Model {model_choice} not available", None, ""
    
    # Start timing
    start_time = time.time()
    
    # Run detection
    model = models[model_choice]
    results = model(image, conf=confidence_threshold, iou=iou_threshold)
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Create visualization
    annotated_image, detection_data = create_visualization(image, results, model_choice, inference_time)
    
    # Create summary
    summary = create_detection_summary(detection_data, model_choice, inference_time)
    
    # Create chart
    chart = create_detection_chart(detection_data)
    
    # Add to history
    detection_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': model_choice,
        'detections': len(detection_data),
        'inference_time': inference_time,
        'objects': detection_data
    })
    
    # Keep only last 50 detections
    if len(detection_history) > 50:
        detection_history.pop(0)
    
    return annotated_image, summary, f"Detection completed in {inference_time:.2f}s"

def get_detection_history():
    """Get formatted detection history"""
    if not detection_history:
        return "No detection history available."
    
    history_text = "## 📈 Detection History\n\n"
    for i, record in enumerate(reversed(detection_history[-10:])):  # Show last 10
        history_text += f"**{i+1}.** {record['timestamp']} | "
        history_text += f"Model: {record['model']} | "
        history_text += f"Objects: {record['detections']} | "
        history_text += f"Time: {record['inference_time']:.2f}s\n"
    
    return history_text

# CSS styling with aquatic effects
AQUATIC_CSS = """
/* Aquatic Background Effects */
body {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #4ca1af 100%) !important;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 20% 80%, rgba(120, 200, 255, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(75, 180, 200, 0.1) 0%, transparent 50%);
    animation: waterMovement 15s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: -1;
}

@keyframes waterMovement {
    0% { transform: translateX(-20px) translateY(-10px); }
    100% { transform: translateX(20px) translateY(10px); }
}

/* Floating Bubbles */
.bubble {
    position: fixed;
    background: radial-gradient(circle, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.3) 70%, transparent 100%);
    border-radius: 50%;
    animation: float 8s infinite ease-in-out;
    pointer-events: none;
    z-index: 1;
}

.bubble:nth-child(1) { width: 20px; height: 20px; left: 10%; animation-delay: 0s; }
.bubble:nth-child(2) { width: 15px; height: 15px; left: 20%; animation-delay: 2s; }
.bubble:nth-child(3) { width: 25px; height: 25px; left: 30%; animation-delay: 4s; }
.bubble:nth-child(4) { width: 18px; height: 18px; left: 40%; animation-delay: 1s; }
.bubble:nth-child(5) { width: 12px; height: 12px; left: 50%; animation-delay: 3s; }
.bubble:nth-child(6) { width: 22px; height: 22px; left: 60%; animation-delay: 5s; }
.bubble:nth-child(7) { width: 16px; height: 16px; left: 70%; animation-delay: 2.5s; }
.bubble:nth-child(8) { width: 14px; height: 14px; left: 80%; animation-delay: 4.5s; }
.bubble:nth-child(9) { width: 20px; height: 20px; left: 90%; animation-delay: 1.5s; }

@keyframes float {
    0% { 
        transform: translateY(100vh) scale(0);
        opacity: 0;
    }
    10% {
        opacity: 1;
    }
    90% {
        opacity: 1;
    }
    100% { 
        transform: translateY(-100px) scale(1);
        opacity: 0;
    }
}

/* Styling with aquatic theme */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    background: rgba(20, 30, 40, 0.95) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4) !important;
    margin-top: 20px !important;
    margin-bottom: 20px !important;
    position: relative;
    z-index: 2;
    color: #e0e6ed !important;
}

.header-section {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 30%, #4ca1af 70%, #c4e0e5 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.header-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        radial-gradient(circle at 30% 70%, rgba(255,255,255,0.1) 0%, transparent 50%),
        radial-gradient(circle at 70% 30%, rgba(255,255,255,0.05) 0%, transparent 50%);
    animation: aquaticShimmer 6s ease-in-out infinite alternate;
}

@keyframes aquaticShimmer {
    0% { opacity: 0.3; transform: scale(1); }
    100% { opacity: 0.7; transform: scale(1.1); }
}

.stats-card {
    background: rgba(30, 40, 50, 0.8) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 15px !important;
    padding: 0.8rem !important;
    margin: 0.3rem !important;
    border: 1px solid rgba(100, 150, 200, 0.3) !important;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease !important;
    color: #e0e6ed !important;
}

.stats-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2) !important;
}

.detection-panel {
    border: 2px solid rgba(30, 60, 114, 0.5) !important;
    border-radius: 20px !important;
    padding: 1rem !important;
    background: linear-gradient(145deg, rgba(25, 35, 45, 0.9), rgba(30, 40, 50, 0.9)) !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3) !important;
    color: #e0e6ed !important;
}

.control-panel {
    background: linear-gradient(145deg, rgba(25, 35, 45, 0.95), rgba(30, 40, 50, 0.95)) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 15px !important;
    padding: 1rem !important;
    border: 2px solid rgba(30, 60, 114, 0.4) !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
    color: #e0e6ed !important;
}

/* Enhanced Button styling with aquatic theme */
.primary-button {
    background: linear-gradient(45deg, #1e3c72, #2a5298, #4ca1af) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 15px 30px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    color: white !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    box-shadow: 0 8px 16px rgba(30, 60, 114, 0.3) !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.primary-button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent) !important;
    transition: left 0.5s ease !important;
}

.primary-button:hover::before {
    left: 100% !important;
}

.primary-button:hover {
    transform: translateY(-3px) scale(1.05) !important;
    box-shadow: 0 12px 24px rgba(30, 60, 114, 0.4) !important;
    background: linear-gradient(45deg, #2a5298, #4ca1af, #c4e0e5) !important;
}

/* Aquatic Tab Styling */
.tab-nav button {
    background: linear-gradient(135deg, rgba(30, 60, 114, 0.1), rgba(76, 161, 175, 0.1)) !important;
    border: 1px solid rgba(30, 60, 114, 0.2) !important;
    border-radius: 10px 10px 0 0 !important;
    color: #1e3c72 !important;
    transition: all 0.3s ease !important;
}

.tab-nav button:hover {
    background: linear-gradient(135deg, rgba(30, 60, 114, 0.2), rgba(76, 161, 175, 0.2)) !important;
    transform: translateY(-2px) !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #1e3c72, #4ca1af) !important;
    color: white !important;
    box-shadow: 0 5px 15px rgba(30, 60, 114, 0.3) !important;
}

/* Accordion styling */
.accordion {
    background: linear-gradient(135deg, rgba(25, 35, 45, 0.9), rgba(30, 40, 50, 0.9)) !important;
    border: 1px solid rgba(30, 60, 114, 0.4) !important;
    border-radius: 15px !important;
    backdrop-filter: blur(10px) !important;
    color: #e0e6ed !important;
}

/* Input field enhancements */
input[type="range"] {
    background: linear-gradient(90deg, #1e3c72, #4ca1af) !important;
    border-radius: 10px !important;
}

/* Image upload area */
.image-upload {
    border: 3px dashed rgba(30, 60, 114, 0.5) !important;
    border-radius: 15px !important;
    background: linear-gradient(135deg, rgba(25, 35, 45, 0.8), rgba(30, 40, 50, 0.8)) !important;
    transition: all 0.3s ease !important;
    color: #e0e6ed !important;
}

.image-upload:hover {
    border-color: rgba(30, 60, 114, 0.8) !important;
    background: linear-gradient(135deg, rgba(30, 40, 50, 0.9), rgba(35, 45, 55, 0.9)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3) !important;
}

/* Hide footer */
footer {
    display: none !important;
}

/* Responsive aquatic effects */
@media (max-width: 768px) {
    .bubble { display: none; }
    .gradio-container { margin: 10px; }
}

/* Prevent tab content resizing and maintain consistent width */
.gradio-tabs {
    width: 100% !important;
    max-width: 1400px !important;
    min-width: 1200px !important;
}

.gradio-tabs .tab-content {
    width: 100% !important;
    max-width: 1400px !important;
    min-width: 1200px !important;
    overflow: visible !important;
    height: auto !important;
}

.gradio-tab-item {
    width: 100% !important;
    max-width: 1400px !important;
    min-width: 1200px !important;
    box-sizing: border-box !important;
}

/* Force all tab content containers to have same width */
.gradio-tabs .gradio-row {
    width: 100% !important;
    max-width: 1400px !important;
    min-width: 1200px !important;
}

.gradio-tabs .gradio-column {
    flex-shrink: 0 !important;
    box-sizing: border-box !important;
}

/* Maintain consistent container width */
.gradio-container .gradio-tabs {
    width: 100% !important;
    max-width: 1400px !important;
    min-width: 1200px !important;
}

/* Lock the main container width */
.gradio-container {
    width: 100% !important;
    max-width: 1400px !important;
    min-width: 1200px !important;
    box-sizing: border-box !important;
}
"""

def build_app():
    """Build the underwater detection application"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="gray"
        ),
        css=AQUATIC_CSS + "\n.full-cover-image img, .full-cover-image canvas { object-fit: contain !important; width: 100% !important; height: 100% !important; background: #101c2c !important; }\n.full-cover-image { padding: 0 !important; margin: 0 !important; background: #101c2c !important; }\n" ,
        title="🐬 Underwater Object Detection System"
    ) as demo:
        
        # Header Section with Aquatic Effects
        with gr.Row():
            gr.HTML("""
            <!-- Floating Bubbles -->
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
            
            <div class="header-section">
                <h1>🐬 Underwater Object Detection System</h1>
                <p>🌊 Advanced AI-powered detection for marine life identification using state-of-the-art YOLOv8 models 🌊</p>
                <div style="display: flex; justify-content: center; gap: 1.5rem; margin-top: 1rem; flex-wrap: wrap;">
                    <div class="stats-card">
                        <h3>🐠 7</h3>
                        <p>Marine Species</p>
                    </div>
                    <div class="stats-card">
                        <h3>🤖 2</h3>
                        <p>AI Models</p>
                    </div>
                    <div class="stats-card">
                        <h3>⚡ Real-time</h3>
                        <p>Detection</p>
                    </div>
                    <div class="stats-card">
                        <h3>🎯 Advanced</h3>
                        <p>Analytics</p>
                    </div>
                </div>
                <div style="margin-top: 0.8rem; font-size: 1em; opacity: 0.9;">
                    🪼 Jellyfish • 🐧 Penguin • 🦅 Puffin • 🦈 Shark • ⭐ Starfish • 🐙 Stingray 🪼
                </div>
            </div>
            """)
        
        # Aquatic Divider
        with gr.Row():
            gr.HTML("""
            <div style="text-align: center; margin: 1rem 0; padding: 1rem;">
                <div style="height: 3px; background: linear-gradient(90deg, transparent, #4ca1af, #2a5298, #4ca1af, transparent); border-radius: 2px; margin: 0.5rem 0;"></div>
                <div style="color: #00d4aa; font-size: 1.1em; font-weight: 500; text-shadow: 0 2px 4px rgba(0, 212, 170, 0.3);">
                    🌊 Dive into Advanced Marine Life Detection 🌊
                </div>
                <div style="height: 3px; background: linear-gradient(90deg, transparent, #4ca1af, #2a5298, #4ca1af, transparent); border-radius: 2px; margin: 0.5rem 0;"></div>
            </div>
            """)
        
        # Model Information Section with Aquatic Theme - Below Title Card
        with gr.Accordion("🌊 Model Information & Aquatic System Details", open=False):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, rgba(30,60,114,0.2), rgba(76,161,175,0.2)); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(30,60,114,0.3); color: #e0e6ed;">
                <h3 style="color: #4ca1af; margin-bottom: 1rem;">🤖 Available AI Models:</h3>
                <div style="margin-bottom: 1.5rem;">
            """)
            
            for model in available_models:
                gr.HTML(f"""
                <div style="background: rgba(76,161,175,0.2); padding: 0.8rem; margin: 0.5rem 0; border-radius: 10px; border-left: 4px solid #4ca1af;">
                    <strong style="color: #4ca1af;">🔹 {model}:</strong> <span style="color: #a8c8ec;">Ready for marine detection</span>
                </div>
                """)
            
            gr.HTML("""
                </div>
                <h3 style="color: #4ca1af; margin-bottom: 1rem;">🎯 Supported Marine Species:</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="background: rgba(255,107,107,0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid rgba(255,107,107,0.4);">
                        <div style="font-size: 2em;">🐠</div>
                        <strong style="color: #FF6B6B;">Fish</strong>
                    </div>
                    <div style="background: rgba(78,205,196,0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid rgba(78,205,196,0.4);">
                        <div style="font-size: 2em;">🪼</div>
                        <strong style="color: #4ECDC4;">Jellyfish</strong>
                    </div>
                    <div style="background: rgba(69,183,209,0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid rgba(69,183,209,0.4);">
                        <div style="font-size: 2em;">🐧</div>
                        <strong style="color: #45B7D1;">Penguin</strong>
                    </div>
                    <div style="background: rgba(150,206,180,0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid rgba(150,206,180,0.4);">
                        <div style="font-size: 2em;">🦅</div>
                        <strong style="color: #96CEB4;">Puffin</strong>
                    </div>
                    <div style="background: rgba(254,202,87,0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid rgba(254,202,87,0.4);">
                        <div style="font-size: 2em;">🦈</div>
                        <strong style="color: #FECA57;">Shark</strong>
                    </div>
                    <div style="background: rgba(255,159,243,0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid rgba(255,159,243,0.4);">
                        <div style="font-size: 2em;">⭐</div>
                        <strong style="color: #FF9FF3;">Starfish</strong>
                    </div>
                    <div style="background: rgba(84,160,255,0.2); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid rgba(84,160,255,0.4);">
                        <div style="font-size: 2em;">🐙</div>
                        <strong style="color: #54A0FF;">Stingray</strong>
                    </div>
                </div>
                
                <h3 style="color: #4ca1af; margin-bottom: 1rem;">⚙️ System Specifications:</h3>
                <div style="background: rgba(35, 45, 55, 0.8); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(30,60,114,0.4); color: #e0e6ed;">
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        <li style="padding: 0.5rem 0; border-bottom: 1px solid rgba(76,161,175,0.3);"><strong style="color: #4ca1af;">🔧 Framework:</strong> YOLOv8 (Ultralytics) - State-of-the-art object detection</li>
                        <li style="padding: 0.5rem 0; border-bottom: 1px solid rgba(76,161,175,0.3);"><strong style="color: #4ca1af;">📸 Input:</strong> RGB Images (any resolution) - Optimized for underwater conditions</li>
                        <li style="padding: 0.5rem 0; border-bottom: 1px solid rgba(76,161,175,0.3);"><strong style="color: #4ca1af;">🎯 Output:</strong> Bounding boxes with confidence scores and species classification</li>
                        <li style="padding: 0.5rem 0; border-bottom: 1px solid rgba(76,161,175,0.3);"><strong style="color: #4ca1af;">⚡ Performance:</strong> Real-time detection capability with advanced analytics</li>
                        <li style="padding: 0.5rem 0;"><strong style="color: #4ca1af;">🌊 Specialization:</strong> Marine environment optimized for underwater research</li>
                    </ul>
                </div>
                
                <div style="text-align: center; margin-top: 1.5rem; padding: 1rem; background: linear-gradient(90deg, rgba(30,60,114,0.3), rgba(76,161,175,0.3), rgba(30,60,114,0.3)); border-radius: 10px;">
                    <span style="color: #4ca1af; font-weight: 600; font-size: 1.1em;">🐬 Built for Marine Research & Conservation 🌊</span>
                </div>
            </div>
            """)
        
        # Main Interface with Tabs
        with gr.Tabs() as main_tabs:
            # Home Tab
            with gr.Tab("🏠 Home"):
                with gr.Row():
                    # Left Panel - Welcome Content (same scale as Detection Controls)
                    with gr.Column(scale=1, elem_classes="control-panel"):
                        gr.HTML("""
                        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(30,60,114,0.1), rgba(76,161,175,0.1)); border-radius: 15px; margin: 0.5rem 0;">
                            <h2 style="color: #4ca1af; margin-bottom: 1rem;">🌊 Welcome to Underwater Detection</h2>
                            <p style="color: #e0e6ed; font-size: 1em; line-height: 1.6;">
                                Experience cutting-edge AI technology for marine life identification. Our system uses advanced YOLOv8 models 
                                trained specifically for underwater environments to detect and classify marine species with high accuracy.
                            </p>
                            <div style="margin: 1.5rem 0;">
                                <div style="display: grid; grid-template-columns: 1fr; gap: 0.8rem; margin: 1rem 0;">
                                    <div style="background: rgba(30,40,50,0.8); padding: 1rem; border-radius: 12px; border: 2px solid rgba(76,161,175,0.3);">
                                        <div style="font-size: 2em; margin-bottom: 0.3rem;">🤖</div>
                                        <h3 style="color: #4ca1af; margin-bottom: 0.3rem; font-size: 1.1em;">AI-Powered</h3>
                                        <p style="color: #a8c8ec; font-size: 0.9em;">Advanced neural networks for precise detection</p>
                                    </div>
                                    <div style="background: rgba(30,40,50,0.8); padding: 1rem; border-radius: 12px; border: 2px solid rgba(76,161,175,0.3);">
                                        <div style="font-size: 2em; margin-bottom: 0.3rem;">🌊</div>
                                        <h3 style="color: #4ca1af; margin-bottom: 0.3rem; font-size: 1.1em;">Marine Specialized</h3>
                                        <p style="color: #a8c8ec; font-size: 0.9em;">Optimized for underwater environments</p>
                                    </div>
                                    <div style="background: rgba(30,40,50,0.8); padding: 1rem; border-radius: 12px; border: 2px solid rgba(76,161,175,0.3);">
                                        <div style="font-size: 2em; margin-bottom: 0.3rem;">📊</div>
                                        <h3 style="color: #4ca1af; margin-bottom: 0.3rem; font-size: 1.1em;">Advanced Analytics</h3>
                                        <p style="color: #a8c8ec; font-size: 0.9em;">Comprehensive reporting and insights</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """)
                        
                        # Go to Detect Button
                        go_to_detect_btn = gr.Button(
                            "🚀 Go to Detection System",
                            variant="primary",
                            elem_classes="primary-button",
                            size="lg"
                        )
                    
                    # Right Panel - Species List (same scale as Detection Results)
                    with gr.Column(scale=2, elem_classes="detection-panel"):
                        gr.HTML("""
                        <div style="padding: 1rem;">
                            <h3 style="color: #4ca1af; margin-bottom: 1.5rem; text-align: center;">🐠 Supported Marine Species</h3>
                            <div style="display: flex; flex-direction: column; gap: 1rem;">
                                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.8rem; background: rgba(255,107,107,0.1); border-radius: 8px; border-left: 4px solid #FF6B6B;">
                                    <span style="font-size: 1.5em;">🐠</span>
                                    <span style="color: #FF6B6B; font-weight: 600;">Fish</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.8rem; background: rgba(78,205,196,0.1); border-radius: 8px; border-left: 4px solid #4ECDC4;">
                                    <span style="font-size: 1.5em;">🪼</span>
                                    <span style="color: #4ECDC4; font-weight: 600;">Jellyfish</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.8rem; background: rgba(69,183,209,0.1); border-radius: 8px; border-left: 4px solid #45B7D1;">
                                    <span style="font-size: 1.5em;">🐧</span>
                                    <span style="color: #45B7D1; font-weight: 600;">Penguin</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.8rem; background: rgba(150,206,180,0.1); border-radius: 8px; border-left: 4px solid #96CEB4;">
                                    <span style="font-size: 1.5em;">🦅</span>
                                    <span style="color: #96CEB4; font-weight: 600;">Puffin</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.8rem; background: rgba(254,202,87,0.1); border-radius: 8px; border-left: 4px solid #FECA57;">
                                    <span style="font-size: 1.5em;">🦈</span>
                                    <span style="color: #FECA57; font-weight: 600;">Shark</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.8rem; background: rgba(255,159,243,0.1); border-radius: 8px; border-left: 4px solid #FF9FF3;">
                                    <span style="font-size: 1.5em;">⭐</span>
                                    <span style="color: #FF9FF3; font-weight: 600;">Starfish</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.8rem; background: rgba(84,160,255,0.1); border-radius: 8px; border-left: 4px solid #54A0FF;">
                                    <span style="font-size: 1.5em;">🐙</span>
                                    <span style="color: #54A0FF; font-weight: 600;">Stingray</span>
                                </div>
                            </div>
                        </div>
                        """)
            
            # Detection Tab
            with gr.Tab("🔍 Detection System"):
                with gr.Row():
                    # Left column: Upload + controls
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="background: linear-gradient(45deg, #1976d2, #4285f4); color: white; padding: 0.8rem; border-radius: 8px 8px 0 0; text-align: center; font-weight: 600; margin-bottom: 0;">
                            🖼️ Upload Underwater Image
                        </div>
                        """)
                        image_input = gr.Image(
                            type="pil",
                            label="",
                            width=700,
                            height=700,
                            show_label=False,
                            container=False,
                            show_download_button=False,
                            show_share_button=False,
                            elem_classes=["full-cover-image"]
                        )
                        # Controls below image
                        gr.HTML("""
                        <div style="margin-top: 1rem;"></div>
                        """)
                        model_dropdown = gr.Dropdown(
                            choices=available_models,
                            value=available_models[0] if available_models else None,
                            label="Select Model",
                            info="Choose the best performing model for your needs"
                        )
                        conf_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.25,
                            step=0.05,
                            label="Confidence Threshold",
                            info="adjust detection sensitivity"
                        )
                        detect_btn = gr.Button(
                            "Submit",
                            variant="primary"
                        )
                        
                    # Right column: Detection result and outputs
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="background: linear-gradient(45deg, #4285f4, #1976d2); color: white; padding: 0.8rem; border-radius: 8px 8px 0 0; text-align: center; font-weight: 600; margin-bottom: 0;">
                            🎯 Detection Results
                        </div>
                        """)
                        output_image = gr.Image(
                            type="pil",
                            label="",
                            width=700,
                            height=700,
                            show_label=False,
                            container=False,
                            show_download_button=False,
                            show_share_button=False,
                            elem_classes=["full-cover-image"]
                        )
                        # Detected objects below result image
                        output_summary = gr.Markdown("Detected objects will appear here.")
                        status_text = gr.Textbox(label="Status", value="", interactive=False, show_label=False)

        detect_btn.click(
            detect_objects,
            inputs=[image_input, model_dropdown, conf_slider],
            outputs=[output_image, output_summary, status_text]
        )
    
    return demo

if __name__ == "__main__":
    print("🐬 Starting Underwater Detection System...")
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=1041,
        share=False,
        show_error=True,
        quiet=False
    )
