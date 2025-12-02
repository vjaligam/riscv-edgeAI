"""
Machine Learning Model for Children Height Prediction
Train a neural network model and export it for edge device deployment
Suitable for RISC-V EdgeAI hardware testing
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json

# Age data (in years)
ages = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

# Average height data for boys (in cm)
boys_height = np.array([50, 75, 87, 96, 103, 110, 116, 122, 128, 133, 138, 143, 149, 156, 164, 170, 173, 175, 176])

# Average height data for girls (in cm)
girls_height = np.array([49, 74, 86, 95, 102, 109, 115, 121, 127, 133, 138, 144, 151, 157, 160, 162, 163, 163, 163])

def generate_training_data(num_samples=1000):
    """
    Generate synthetic training data with realistic variations
    """
    X = []  # [age, gender]
    y = []  # height
    
    for _ in range(num_samples):
        age = np.random.uniform(0, 18)
        gender = np.random.choice([0, 1])  # 0: boy, 1: girl
        
        # Interpolate base height
        if gender == 0:  # boy
            base_height = np.interp(age, ages, boys_height)
        else:  # girl
            base_height = np.interp(age, ages, girls_height)
        
        # Add realistic variation (standard deviation increases with age)
        variation = np.random.normal(0, 3 + age * 0.3)
        height = base_height + variation
        
        X.append([age, gender])
        y.append(height)
    
    return np.array(X), np.array(y)

def create_model(input_shape):
    """
    Create a neural network model for height prediction
    Optimized for edge device deployment
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(64, activation='relu', name='hidden1'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu', name='hidden2'),
        keras.layers.Dense(16, activation='relu', name='hidden3'),
        keras.layers.Dense(1, name='output')  # Single output: height
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model():
    """
    Train the height prediction model
    """
    print("="*60)
    print("TRAINING HEIGHT PREDICTION MODEL FOR EDGE DEPLOYMENT")
    print("="*60)
    
    # Generate training data
    print("\n1. Generating training data...")
    X, y = generate_training_data(num_samples=2000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize features
    print("2. Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    print("3. Creating neural network model...")
    model = create_model(input_shape=(2,))
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\n4. Training model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Evaluate model
    print("\n5. Evaluating model...")
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest Loss (MSE): {test_loss:.2f}")
    print(f"Test MAE: {test_mae:.2f} cm")
    
    # Make predictions on test set
    y_pred = model.predict(X_test_scaled, verbose=0)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot predictions vs actual
    plot_predictions(y_test, y_pred)
    
    # Save model and scaler
    save_model_for_deployment(model, scaler, history)
    
    return model, scaler, history

def plot_training_history(history):
    """
    Plot training and validation loss
    """
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (cm)')
    plt.title('Model Training MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\n✓ Training history plot saved as 'training_history.png'")
    plt.show()

def plot_predictions(y_true, y_pred):
    """
    Plot predicted vs actual heights
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Height (cm)', fontsize=12)
    plt.ylabel('Predicted Height (cm)', fontsize=12)
    plt.title('Model Predictions vs Actual Heights', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    print("✓ Predictions plot saved as 'predictions.png'")
    plt.show()

def save_model_for_deployment(model, scaler, history):
    """
    Save model in multiple formats for hardware deployment
    """
    print("\n6. Saving model for deployment...")
    
    # Save full Keras model (native format)
    model.save('height_model.keras')
    print("   ✓ Keras model saved: height_model.keras")
    
    # Save model in SavedModel format (for TFLite conversion)
    model.export('height_model_saved')
    print("   ✓ SavedModel format saved: height_model_saved/")
    
    # Convert to TensorFlow Lite (for edge devices)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('height_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("   ✓ TensorFlow Lite model saved: height_model.tflite")
    
    # Save quantized model (smaller size for edge devices)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()
    
    with open('height_model_quantized.tflite', 'wb') as f:
        f.write(tflite_quant_model)
    print("   ✓ Quantized TFLite model saved: height_model_quantized.tflite")
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("   ✓ Scaler saved: scaler.pkl")
    
    # Save model info
    model_info = {
        'input_shape': [2],
        'input_features': ['age', 'gender'],
        'gender_encoding': {'boy': 0, 'girl': 1},
        'output': 'height_cm',
        'final_loss': float(history.history['loss'][-1]),
        'final_mae': float(history.history['mae'][-1]),
        'architecture': {
            'layers': [
                {'type': 'Dense', 'units': 64, 'activation': 'relu'},
                {'type': 'Dropout', 'rate': 0.2},
                {'type': 'Dense', 'units': 32, 'activation': 'relu'},
                {'type': 'Dense', 'units': 16, 'activation': 'relu'},
                {'type': 'Dense', 'units': 1, 'activation': 'linear'}
            ]
        }
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    print("   ✓ Model info saved: model_info.json")
    
    # Print file sizes
    import os
    print("\nModel File Sizes:")
    print(f"   - Keras (.keras): {os.path.getsize('height_model.keras') / 1024:.2f} KB")
    print(f"   - TFLite: {os.path.getsize('height_model.tflite') / 1024:.2f} KB")
    print(f"   - Quantized TFLite: {os.path.getsize('height_model_quantized.tflite') / 1024:.2f} KB")

def test_model_inference(model, scaler):
    """
    Test the model with sample inputs
    """
    print("\n" + "="*60)
    print("TESTING MODEL INFERENCE")
    print("="*60)
    
    test_cases = [
        (5, 0, "5-year-old boy"),
        (5, 1, "5-year-old girl"),
        (12, 0, "12-year-old boy"),
        (12, 1, "12-year-old girl"),
        (16, 0, "16-year-old boy"),
        (16, 1, "16-year-old girl"),
    ]
    
    print("\nSample Predictions:")
    print("-" * 60)
    
    for age, gender, description in test_cases:
        # Prepare input
        X_input = np.array([[age, gender]])
        X_scaled = scaler.transform(X_input)
        
        # Predict
        height = model.predict(X_scaled, verbose=0)[0][0]
        
        # Get expected height
        if gender == 0:
            expected = np.interp(age, ages, boys_height)
        else:
            expected = np.interp(age, ages, girls_height)
        
        print(f"{description:20s} -> Predicted: {height:.1f} cm | Expected: {expected:.1f} cm")

def test_tflite_model():
    """
    Test TensorFlow Lite model for hardware deployment
    """
    print("\n" + "="*60)
    print("TESTING TFLITE MODEL (Edge Device Simulation)")
    print("="*60)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="height_model.tflite")
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nTFLite Model Details:")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output type: {output_details[0]['dtype']}")
    
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Test inference
    print("\nTFLite Inference Test:")
    print("-" * 60)
    
    test_cases = [
        (10, 0, "10-year-old boy"),
        (10, 1, "10-year-old girl"),
        (15, 0, "15-year-old boy"),
        (15, 1, "15-year-old girl"),
    ]
    
    for age, gender, description in test_cases:
        # Prepare input
        X_input = np.array([[age, gender]], dtype=np.float32)
        X_scaled = scaler.transform(X_input).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], X_scaled)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        height = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        print(f"{description:20s} -> Predicted: {height:.1f} cm")
    
    print("\n✓ TFLite model is ready for hardware deployment!")

if __name__ == "__main__":
    # Train the model
    model, scaler, history = train_model()
    
    # Test the trained model
    test_model_inference(model, scaler)
    
    # Test TFLite model (hardware simulation)
    test_tflite_model()
    
    print("\n" + "="*60)
    print("MODEL TRAINING AND EXPORT COMPLETE!")
    print("="*60)
    print("\nNext Steps for Hardware Deployment:")
    print("1. Transfer 'height_model_quantized.tflite' to your RISC-V device")
    print("2. Transfer 'scaler.pkl' for input preprocessing")
    print("3. Use TensorFlow Lite runtime on the edge device")
    print("4. Refer to 'model_info.json' for model specifications")
    print("\nFiles ready for deployment:")
    print("  - height_model_quantized.tflite (optimized for edge)")
    print("  - height_model.tflite (standard)")
    print("  - scaler.pkl")
    print("  - model_info.json")
    print("="*60)

