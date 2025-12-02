# Quick Start Guide 🚀

Get started with Hand Gesture Recognition in 5 minutes!

## Step 1: Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow 2.20+
- OpenCV (for camera)
- NumPy, Matplotlib
- Scikit-learn, Seaborn

## Step 2: Collect Training Data (5-10 min)

```bash
python collect_data.py
```

**What this does:**
- Opens your webcam
- Guides you through capturing 6 gestures
- Saves ~300 images per gesture
- Total: ~1,800 images

**Tips:**
- Use good lighting
- Clear background
- Vary hand position, angle, distance
- Press SPACE to start auto-capture
- Press N to move to next gesture

## Step 3: Train the Model (5-10 min)

```bash
python train_model.py
```

**What this does:**
- Loads your collected images
- Trains MobileNetV2-based CNN
- Exports Keras model
- Exports TFLite models (standard & quantized)
- Generates accuracy plots

**Expected output:**
- Validation accuracy: 90-98%
- Model files in `models/` folder
- Training plots saved

## Step 4: Test Real-time Recognition (immediately)

```bash
python realtime_demo.py
```

**What this does:**
- Loads trained model
- Opens webcam
- Shows live gesture recognition
- Displays confidence scores

**Controls:**
- Q: Quit
- S: Screenshot
- R: Reset prediction buffer

## Step 5: Deploy to RISC-V Edge Device

### Files to Transfer:
```
models/gesture_model_quantized.tflite
models/gesture_labels.json
hw_inference.py
```

### On Your RISC-V Device:

```python
from hw_inference import EdgeGestureRecognizer
import numpy as np

# Initialize
recognizer = EdgeGestureRecognizer()
recognizer.load_model()

# Predict (image should be 224x224x3 numpy array)
gesture, confidence, predictions = recognizer.predict(image)
print(f"Detected: {gesture} ({confidence*100:.1f}%)")
```

## Troubleshooting

### Camera not opening?
- Check if another app is using the camera
- Try `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
- On Linux, you may need: `sudo apt-get install python3-opencv`

### Low accuracy?
- Collect more training data (300+ per gesture)
- Ensure varied positions/angles/lighting
- Check if gestures are clearly different
- Try training longer (increase EPOCHS in train_model.py)

### Out of memory during training?
- Reduce batch size: `BATCH_SIZE = 16` in train_model.py
- Reduce image size: `IMG_SIZE = 160` in all files
- Close other applications

### Slow inference?
- Use quantized model (already default)
- Reduce image size
- On edge device, use TFLite runtime instead of full TensorFlow

## Customization

### Add More Gestures

1. Edit `collect_data.py`:
```python
GESTURES = [
    'fist',
    'palm',
    'peace',
    'thumbs_up',
    'thumbs_down',
    'ok_sign',
    'your_new_gesture'  # Add here
]
```

2. Run `collect_data.py` again
3. Train with `train_model.py`

### Change Model Size

In `train_model.py`:
```python
# Smaller, faster (less accurate)
IMG_SIZE = 160
alpha = 0.5  # In MobileNetV2 loading

# Larger, slower (more accurate)
IMG_SIZE = 224
alpha = 1.0
```

## Performance Benchmarks

### Development Machine (CPU):
- Training: 5-10 minutes
- Inference: 30-60 FPS
- Model size: ~2 MB (quantized)

### RISC-V Edge Device (varies by hardware):
- Inference: 5-20 FPS (depending on CPU)
- RAM usage: < 10 MB
- Model size: ~2 MB

## Next Steps

✅ Collect data
✅ Train model  
✅ Test with webcam  
✅ Deploy to edge device  

**Advanced:**
- Add hand landmark detection (MediaPipe)
- Implement gesture sequences
- Add temporal smoothing
- Create gesture-based controls
- Build IoT applications

## Support

- Read the full README.md
- Check GitHub Issues
- Review TensorFlow Lite docs

Happy Gesture Recognition! 🖐️

