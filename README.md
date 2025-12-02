# Children Height Prediction Model - RISC-V EdgeAI

A machine learning project that predicts children's height based on age and gender, trained and optimized for edge device deployment on RISC-V hardware.

## 📋 Project Overview

This project includes:
1. **Data Visualization** - Interactive plots showing height vs age relationships
2. **Neural Network Model** - Trained on WHO growth standards data
3. **Edge Deployment** - TensorFlow Lite models optimized for RISC-V devices
4. **Hardware Inference** - Lightweight inference scripts for edge devices

## 🎯 Model Performance

- **Test Loss (MSE)**: 37.46
- **Test MAE**: 4.74 cm
- **Inference Speed**: ~0.23 ms per prediction (~4,400 inferences/sec on CPU)
- **Model Size**: 
  - Keras: 64.66 KB
  - TFLite: 7.60 KB
  - Quantized TFLite: 8.69 KB (best for edge devices)

## 📁 Files Description

### Python Scripts

| File | Description |
|------|-------------|
| `children_age.py` | Data visualization script with WHO growth standards |
| `train_height_model.py` | Complete training pipeline for the neural network |
| `hw_inference.py` | Hardware inference script for edge deployment |

### Model Files

| File | Description | Use Case |
|------|-------------|----------|
| `height_model.keras` | Full Keras model | Training/Development |
| `height_model_saved/` | SavedModel format | TensorFlow Serving |
| `height_model.tflite` | TFLite model | Edge devices |
| `height_model_quantized.tflite` | Quantized TFLite | **Recommended for RISC-V** |
| `scaler.pkl` | Input preprocessor | Required for inference |
| `model_info.json` | Model specifications | Documentation |

### Visualizations

| File | Description |
|------|-------------|
| `training_history.png` | Training/validation loss curves |
| `predictions.png` | Model predictions vs actual heights |

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow scikit-learn matplotlib numpy
```

### 1. Visualize Data

```bash
python children_age.py
```

This displays:
- Height data table for ages 0-18
- Interactive graph comparing boys vs girls growth

### 2. Train the Model

```bash
python train_height_model.py
```

This will:
- Generate synthetic training data (2000 samples)
- Train a neural network (4 layers, ~2,800 parameters)
- Evaluate model performance
- Export models in multiple formats
- Generate visualization plots

### 3. Test Inference

```bash
# Demo mode (default)
python hw_inference.py

# Interactive mode
python hw_inference.py --interactive
```

## 🔧 Model Architecture

```
Input Layer:       2 features (age, gender)
Hidden Layer 1:    64 neurons (ReLU)
Dropout:           20%
Hidden Layer 2:    32 neurons (ReLU)
Hidden Layer 3:    16 neurons (ReLU)
Output Layer:      1 neuron (Height in cm)

Total Parameters:  2,817
```

### Input Format

- **Age**: Float (0-18 years)
- **Gender**: Integer (0=boy, 1=girl)

### Output

- **Height**: Float (cm)

## 📊 Training Details

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Training Data**: 2000 synthetic samples with realistic variations
- **Validation Split**: 20%
- **Epochs**: Up to 100 (with early stopping)
- **Callbacks**: 
  - Early Stopping (patience=10)
  - Learning Rate Reduction (factor=0.5, patience=5)

## 🎯 Deployment on RISC-V EdgeAI

### Files to Transfer to Device

1. `height_model_quantized.tflite` (8.69 KB)
2. `scaler.pkl`
3. `hw_inference.py`
4. `model_info.json` (optional, for reference)

### On the RISC-V Device

1. **Install TensorFlow Lite Runtime**:
   ```bash
   pip install tflite-runtime
   ```

2. **Run Inference**:
   ```python
   from hw_inference import HeightPredictor
   
   predictor = HeightPredictor()
   height = predictor.predict(age=10, gender='boy')
   print(f"Predicted height: {height:.1f} cm")
   ```

### Memory Requirements

- **Model**: ~9 KB
- **Runtime Memory**: < 1 MB
- **Suitable for**: Most microcontrollers with 2+ MB RAM

## 📈 Sample Predictions

| Age | Gender | Predicted Height |
|-----|--------|-----------------|
| 0   | Boy    | 60.1 cm        |
| 5   | Boy    | 110.6 cm       |
| 10  | Boy    | 138.7 cm       |
| 15  | Boy    | 166.7 cm       |
| 18  | Boy    | 184.3 cm       |
| 0   | Girl   | 62.3 cm        |
| 5   | Girl   | 109.6 cm       |
| 10  | Girl   | 141.2 cm       |
| 15  | Girl   | 159.6 cm       |
| 18  | Girl   | 170.7 cm       |

## 🔬 Technical Details

### Data Preprocessing

- **Normalization**: StandardScaler (zero mean, unit variance)
- **Feature Engineering**: Direct age and gender encoding
- **Data Augmentation**: Realistic variations added (σ = 3 + age × 0.3)

### Model Optimization

- **Quantization**: Float16 for reduced model size
- **Pruning**: Dropout regularization during training
- **Optimization**: TFLite converter with default optimizations

### Inference Pipeline

```
Input (age, gender) 
  → StandardScaler 
  → Neural Network 
  → Output (height)
```

## 📝 API Usage Example

```python
from hw_inference import HeightPredictor

# Initialize
predictor = HeightPredictor(
    model_path='height_model_quantized.tflite',
    scaler_path='scaler.pkl'
)

# Single prediction
height = predictor.predict(age=12.5, gender='girl')
print(f"Height: {height:.1f} cm")

# Batch prediction
ages = [7, 8, 9, 10]
genders = ['boy', 'girl', 'boy', 'girl']
heights = predictor.predict_batch(ages, genders)
```

## 🧪 Testing & Validation

The model was validated against WHO growth standards:
- **Accuracy**: ±4.74 cm average error
- **Coverage**: Ages 0-18 years
- **Populations**: Both male and female

## 🛠️ Customization

### Retrain with Custom Data

Modify `train_height_model.py`:

```python
# Replace these arrays with your data
ages = np.array([...])
boys_height = np.array([...])
girls_height = np.array([...])
```

### Adjust Model Complexity

```python
# In create_model() function
keras.layers.Dense(64, activation='relu')  # Change neuron count
```

### Change Quantization

```python
# In save_model_for_deployment()
converter.target_spec.supported_types = [tf.int8]  # More aggressive
```

## 📊 Benchmarking

Benchmark on your device:

```python
from hw_inference import HeightPredictor, benchmark_inference

predictor = HeightPredictor()
benchmark_inference(predictor, num_iterations=1000)
```

## 🐛 Troubleshooting

### Issue: Import Error for TensorFlow

**Solution**: Install TensorFlow or TFLite runtime
```bash
pip install tensorflow
# or for edge devices
pip install tflite-runtime
```

### Issue: Scaler Not Found

**Solution**: Ensure `scaler.pkl` is in the same directory as the model

### Issue: Predictions Seem Off

**Solution**: Check input format - age should be 0-18, gender should be 0/1 or 'boy'/'girl'

## 📚 References

- WHO Child Growth Standards: https://www.who.int/tools/child-growth-standards
- TensorFlow Lite: https://www.tensorflow.org/lite
- RISC-V: https://riscv.org/

## 📄 License

This project is for educational and research purposes.

## 👥 Contributing

Feel free to:
- Add more features (weight prediction, BMI calculation)
- Improve model accuracy with more data
- Optimize for specific hardware platforms
- Add support for other edge devices

## 📧 Contact

For questions about deployment on RISC-V EdgeAI hardware, refer to your device documentation.

---

**Last Updated**: December 2025  
**Model Version**: 1.0  
**TensorFlow Version**: 2.20.0

