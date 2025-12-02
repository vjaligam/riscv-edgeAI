# RISC-V Edge AI Projects 🚀

A collection of machine learning projects optimized for RISC-V edge devices. These projects demonstrate practical EdgeAI applications with complete training pipelines, optimized models, and deployment scripts.

## 📁 Projects

### 1. 📏 [Children Height Prediction](./README_height_prediction.md)
Predict children's height based on age and gender using WHO growth standards.

- **Model Type**: Simple Neural Network
- **Model Size**: 8.69 KB (quantized)
- **Accuracy**: ±4.74 cm MAE
- **Inference Speed**: ~0.23 ms per prediction
- **Use Cases**: Healthcare, growth tracking, pediatrics

**Key Features:**
- ✅ Ultra-lightweight model
- ✅ TFLite optimized for RISC-V
- ✅ Interactive visualization
- ✅ WHO standards data

### 2. 🖐️ [Hand Gesture Recognition](./hand-gesture-recognition/)
Real-time hand gesture recognition using computer vision and deep learning.

- **Model Type**: Custom CNN
- **Model Size**: ~2 MB (quantized)
- **Gestures**: 6 classes (Fist, Palm, Peace, Thumbs Up/Down, OK)
- **Inference Speed**: 30-60 FPS (desktop), 5-20 FPS (RISC-V)
- **Use Cases**: Smart home, gaming, robotics, accessibility

**Key Features:**
- ✅ Data collection tool (webcam)
- ✅ Training pipeline with augmentation
- ✅ Real-time demo
- ✅ Edge deployment ready

---

## 🎯 Quick Start

### Height Prediction
```bash
# Visualize data
python children_age.py

# Train model
python train_height_model.py

# Test inference
python hw_inference.py
```

### Hand Gesture Recognition
```bash
cd hand-gesture-recognition/

# Collect data
python collect_data.py

# Train model
python train_model.py

# Live demo
python realtime_demo.py
```

---

## 📊 Project Comparison

| Feature | Height Prediction | Gesture Recognition |
|---------|------------------|---------------------|
| **Complexity** | Simple | Medium |
| **Model Size** | 8.69 KB | ~2 MB |
| **Input** | Age + Gender | Camera (224x224) |
| **Output** | Height (cm) | Gesture class |
| **Training Time** | 2-5 min | 5-10 min |
| **Accuracy** | 95%+ | 90-98% |
| **Hardware** | Any RISC-V | RISC-V with vision |
| **RAM Required** | < 1 MB | < 10 MB |

---

## 🛠️ Technology Stack

- **Framework**: TensorFlow 2.20+
- **Optimization**: TensorFlow Lite
- **Target**: RISC-V Edge Devices
- **Languages**: Python
- **Libraries**: NumPy, OpenCV, Scikit-learn

---

## 🚀 Deployment

### For RISC-V Devices

**Height Prediction:**
```bash
# Transfer files
- height_model_quantized.tflite (8.69 KB)
- scaler.pkl
- hw_inference.py

# Run
python hw_inference.py
```

**Gesture Recognition:**
```bash
# Transfer files
- gesture_model_quantized.tflite (~2 MB)
- gesture_labels.json
- hw_inference.py

# Run
python hw_inference.py
```

---

## 📈 Performance Benchmarks

### Height Prediction
- **Inference Time**: 0.23 ms
- **Throughput**: 4,400 predictions/sec
- **Memory**: < 1 MB RAM
- **Power**: Ultra-low

### Gesture Recognition
- **Inference Time**: 10-30 ms (RISC-V)
- **Throughput**: 30-60 FPS (desktop)
- **Memory**: < 10 MB RAM
- **Power**: Low

---

## 🎓 Learning Path

1. **Start with**: Height Prediction (simpler)
2. **Progress to**: Gesture Recognition (more complex)
3. **Explore**: Custom ML projects

---

## 📚 Documentation

- [Height Prediction Details](./README_height_prediction.md)
- [Gesture Recognition Guide](./hand-gesture-recognition/README.md)
- [GitHub Setup Guide](./GITHUB_SETUP.md)

---

## 🤝 Contributing

We welcome contributions! Feel free to:
- Add new ML projects
- Improve existing models
- Optimize for specific RISC-V hardware
- Enhance documentation

---

## 📄 License

MIT License - See [LICENSE](./LICENSE) file

---

## 🌟 Project Structure

```
riscv-edgeAI/
├── README.md                           # This file
├── README_height_prediction.md         # Height prediction docs
│
├── children_age.py                     # Height: Data visualization
├── train_height_model.py               # Height: Training
├── hw_inference.py                     # Height: Inference
├── height_model_quantized.tflite       # Height: Trained model
├── scaler.pkl                          # Height: Preprocessor
├── model_info.json                     # Height: Model specs
├── training_history.png                # Height: Training plots
├── predictions.png                     # Height: Accuracy plots
│
└── hand-gesture-recognition/           # Gesture recognition project
    ├── README.md                       # Gesture: Documentation
    ├── QUICKSTART.md                   # Gesture: Quick guide
    ├── collect_data.py                 # Gesture: Data collection
    ├── train_model.py                  # Gesture: Training
    ├── realtime_demo.py                # Gesture: Live demo
    ├── hw_inference.py                 # Gesture: Edge inference
    ├── create_dummy_data.py            # Gesture: Test data
    ├── requirements.txt                # Gesture: Dependencies
    ├── models/                         # Gesture: Trained models
    └── dataset/                        # Gesture: Training data
```

---

## 🎯 Use Cases

### Height Prediction
- 👶 Pediatric healthcare
- 📊 Growth tracking apps
- 🏥 Medical diagnosis tools
- 📱 Parenting apps

### Gesture Recognition
- 🏠 Smart home control
- 🎮 Gaming interfaces
- 🤖 Robotics control
- ♿ Accessibility tools
- 📸 Photography controls

---

## 🔬 Future Projects (Coming Soon)

- 🔊 **Keyword Spotting** - Wake word detection
- 🏃 **Activity Recognition** - Fitness tracking
- 🔍 **Object Detection** - YOLO-Nano for RISC-V
- ❤️ **Health Monitoring** - Vital signs tracking
- 🌡️ **Environmental Sensing** - Air quality prediction

---

## 📧 Contact & Support

- **GitHub**: [https://github.com/vjaligam/riscv-edgeAI](https://github.com/vjaligam/riscv-edgeAI)
- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Open GitHub Discussions for questions

---

## 🙏 Acknowledgments

- WHO for growth standards data
- TensorFlow team for TFLite
- RISC-V community
- OpenCV contributors

---

**Last Updated**: December 2025  
**Repository**: https://github.com/vjaligam/riscv-edgeAI  
**License**: MIT

---

⭐ **Star this repo if you find it useful!** ⭐
