# Hand Gesture Recognition - RISC-V EdgeAI 🤚

A real-time hand gesture recognition system optimized for RISC-V edge devices. Recognizes common gestures like rock, paper, scissors, thumbs up/down, and more using computer vision and deep learning.

## 🎯 Project Overview

This project includes:
1. **Data Collection Tool** - Capture custom gesture datasets
2. **CNN Model Training** - MobileNet-based lightweight architecture
3. **Real-time Inference** - Live webcam gesture recognition
4. **Edge Deployment** - TensorFlow Lite models for RISC-V

## 🖐️ Supported Gestures

- ✊ **Fist/Rock**
- ✋ **Palm/Paper**
- ✌️ **Peace/Scissors**
- 👍 **Thumbs Up**
- 👎 **Thumbs Down**
- 👌 **OK Sign**

(Easily extensible to more gestures!)

## 📁 Project Structure

```
hand-gesture-riscv/
├── collect_data.py          # Capture gesture images
├── train_model.py           # Train CNN model
├── realtime_demo.py         # Live webcam inference
├── hw_inference.py          # Edge device deployment
├── models/                  # Saved models
├── dataset/                 # Collected gesture images
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Training Data

```bash
python collect_data.py
```

Follow on-screen instructions to capture images for each gesture.

### 3. Train the Model

```bash
python train_model.py
```

This will train a MobileNetV2-based CNN and export TFLite models.

### 4. Test Real-time Recognition

```bash
python realtime_demo.py
```

Shows live gesture recognition through your webcam.

### 5. Deploy to RISC-V Device

Transfer files:
- `gesture_model_quantized.tflite`
- `hw_inference.py`
- `gesture_labels.json`

## 🎯 Model Performance

- **Accuracy**: ~95%+ on validation set
- **Model Size**: ~1-2 MB (quantized)
- **Inference Speed**: 30-60 FPS on modern hardware
- **Edge Speed**: 10-20 FPS on RISC-V (depends on device)

## 🔧 Technical Details

### Model Architecture
- **Base**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224x3
- **Transfer Learning**: Fine-tuned top layers
- **Output**: Softmax classification

### Data Preprocessing
- Image resizing and normalization
- Data augmentation (rotation, flip, zoom)
- Real-time hand detection (MediaPipe)

## 📊 Use Cases

- 🎮 **Gaming Controllers** - Gesture-based game controls
- 🏠 **Smart Home** - Control devices without touch
- 🤖 **Robotics** - Robot command interface
- ♿ **Accessibility** - Assistive technology
- 📸 **Photography** - Remote camera control
- 🎨 **Virtual Reality** - Gesture interactions

## 🛠️ Customization

### Add New Gestures

1. Run `collect_data.py` and add your gesture
2. Capture 200-300 images per gesture
3. Retrain with `train_model.py`

### Adjust Model Size

Edit `train_model.py`:
- Change input size (96x96 for smaller, 224x224 for better accuracy)
- Adjust MobileNetV2 alpha (0.35 to 1.0)

## 📚 References

- MobileNetV2: https://arxiv.org/abs/1801.04381
- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands
- TensorFlow Lite: https://www.tensorflow.org/lite

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Feel free to:
- Add more gesture types
- Improve model accuracy
- Optimize for specific hardware
- Add new features

---

**Last Updated**: December 2025
**Model Version**: 1.0
**TensorFlow Version**: 2.20.0

