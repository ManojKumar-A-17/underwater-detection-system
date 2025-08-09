# 🐬 Underwater Object Detection with YOLOv8 + Gradio

A web-based application for detecting underwater objects using YOLOv8 models. This project provides an interactive interface for real-time object detection in underwater images.

## 🌊 Features

- **Real-time Detection**: Upload images and get instant detection results
- **Multi-Model Comparison**: Compare YOLOv8n and YOLOv8s models side-by-side
- **Automatic Best Model Selection**: Automatically selects the best performing model
- **Performance Metrics**: Shows mAP50 scores for each model
- **Web Interface**: User-friendly Gradio web interface
- **Underwater Optimized**: Designed for underwater object detection tasks

## 🎯 Supported Object Classes

The application can detect 7 specific underwater objects:
- **Fish** 🐠
- **Jellyfish** 🪼
- **Penguin** 🐧
- **Puffin** 🦅
- **Shark** 🦈
- **Starfish** ⭐
- **Stingray** 🐙

*Note: This uses a custom-trained YOLOv8 model specifically for underwater detection.*

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd underwater-detection-gradio
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train and update model (if needed)**
   ```bash
   python train_and_update.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to the URL shown in the terminal (usually `http://127.0.0.1:7860`)

## 📖 Usage

1. **Upload Image**: Click the upload area to select an underwater image
2. **Select Model**: Choose between YOLOv8n (faster) or YOLOv8s (more accurate)
3. **Get Results**: View the detection results with bounding boxes and labels

## 🛠️ Technical Details

### Models Used
- **YOLOv8n**: Faster inference, good for real-time applications
- **YOLOv8s**: Higher accuracy, better for precision-critical tasks
- **Automatic Selection**: App automatically chooses the best performing model
- **7 Classes**: fish, jellyfish, penguin, puffin, shark, starfish, stingray
- **Performance Comparison**: Compare mAP50 scores between models

### Dependencies
- `gradio`: Web interface framework
- `ultralytics`: YOLOv8 implementation
- `opencv-python`: Computer vision operations
- `Pillow`: Image processing
- `torch`: PyTorch deep learning framework

## 📁 Project Structure

```
underwater-detection-gradio/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── venv/              # Virtual environment (created locally)
└── examples/          # Example images (optional)
```

## 🔧 Customization

### Adding Custom Models
To use your own trained YOLOv8 models:

1. Place your `.pt` model file in the project directory
2. Update the model loading in `app.py`:
   ```python
   model_custom = YOLO("your_model.pt")
   ```

### Modifying Object Classes
The application uses the default COCO dataset classes. To customize for specific underwater objects, train a custom YOLOv8 model on your dataset.

## 🌐 Deployment

### Local Deployment
```bash
python app.py
```

### Cloud Deployment
The application can be deployed on platforms like:
- **Hugging Face Spaces**
- **Heroku**
- **Google Colab**
- **AWS/GCP/Azure**

## 📊 Performance

- **YOLOv8n**: ~45 FPS on RTX 3080
- **YOLOv8s**: ~30 FPS on RTX 3080
- **Accuracy**: mAP@0.5: 0.86 (YOLOv8s)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 implementation
- [Gradio](https://gradio.app/) for the web interface framework
- [COCO Dataset](https://cocodataset.org/) for pre-trained model weights

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

## Results

Some predicted images are added.


**Made with ❤️ for underwater research and marine conservation** 
