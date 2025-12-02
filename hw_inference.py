"""
Hardware Inference Script for Height Prediction Model
This script demonstrates how to use the TFLite model on edge devices
Optimized for RISC-V EdgeAI deployment
"""

import numpy as np
import tensorflow as tf
import pickle
import json

class HeightPredictor:
    """
    Lightweight inference class for edge device deployment
    """
    def __init__(self, model_path='height_model_quantized.tflite', scaler_path='scaler.pkl'):
        """
        Initialize the height predictor with TFLite model
        
        Parameters:
        -----------
        model_path : str
            Path to the TFLite model file
        scaler_path : str
            Path to the scaler pickle file
        """
        print("Initializing Height Predictor...")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("✓ Model loaded successfully")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Output shape: {self.output_details[0]['shape']}")
    
    def predict(self, age, gender):
        """
        Predict height for given age and gender
        
        Parameters:
        -----------
        age : float
            Age in years (0-18)
        gender : str or int
            'boy'/0 or 'girl'/1
        
        Returns:
        --------
        float : Predicted height in cm
        """
        # Convert gender to numeric if needed
        if isinstance(gender, str):
            gender_code = 0 if gender.lower() == 'boy' else 1
        else:
            gender_code = gender
        
        # Prepare input
        X_input = np.array([[age, gender_code]], dtype=np.float32)
        X_scaled = self.scaler.transform(X_input).astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], X_scaled)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        height = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        
        return float(height)
    
    def predict_batch(self, ages, genders):
        """
        Predict heights for multiple inputs
        
        Parameters:
        -----------
        ages : list
            List of ages in years
        genders : list
            List of genders ('boy'/'girl' or 0/1)
        
        Returns:
        --------
        list : Predicted heights in cm
        """
        predictions = []
        for age, gender in zip(ages, genders):
            height = self.predict(age, gender)
            predictions.append(height)
        return predictions

def benchmark_inference(predictor, num_iterations=100):
    """
    Benchmark inference speed on the device
    """
    import time
    
    print(f"\nRunning inference benchmark ({num_iterations} iterations)...")
    
    # Sample test case
    age, gender = 10.0, 0
    
    # Warmup
    for _ in range(10):
        predictor.predict(age, gender)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        predictor.predict(age, gender)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = (total_time / num_iterations) * 1000  # Convert to ms
    
    print(f"✓ Benchmark complete")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average inference time: {avg_time:.4f} ms")
    print(f"  Throughput: {num_iterations/total_time:.2f} inferences/second")

def demo_inference():
    """
    Demonstrate model inference with various test cases
    """
    print("="*60)
    print("HEIGHT PREDICTION - HARDWARE INFERENCE DEMO")
    print("="*60)
    
    # Initialize predictor
    predictor = HeightPredictor()
    
    # Test cases
    print("\n" + "-"*60)
    print("Sample Predictions:")
    print("-"*60)
    
    test_cases = [
        (0, 'boy', "Newborn boy"),
        (0, 'girl', "Newborn girl"),
        (5, 'boy', "5-year-old boy"),
        (5, 'girl', "5-year-old girl"),
        (10, 'boy', "10-year-old boy"),
        (10, 'girl', "10-year-old girl"),
        (12, 'boy', "12-year-old boy"),
        (12, 'girl', "12-year-old girl"),
        (15, 'boy', "15-year-old boy"),
        (15, 'girl', "15-year-old girl"),
        (18, 'boy', "18-year-old boy"),
        (18, 'girl', "18-year-old girl"),
    ]
    
    for age, gender, description in test_cases:
        height = predictor.predict(age, gender)
        height_inches = height / 2.54
        print(f"{description:25s} -> {height:6.1f} cm ({height_inches:5.1f} inches)")
    
    # Batch prediction demo
    print("\n" + "-"*60)
    print("Batch Prediction Demo:")
    print("-"*60)
    
    ages = [7, 8, 9, 11, 13, 14]
    genders = ['boy', 'girl', 'boy', 'girl', 'boy', 'girl']
    
    heights = predictor.predict_batch(ages, genders)
    
    for age, gender, height in zip(ages, genders, heights):
        print(f"Age {age} ({gender:4s}) -> {height:6.1f} cm")
    
    # Benchmark
    benchmark_inference(predictor, num_iterations=1000)
    
    print("\n" + "="*60)
    print("INFERENCE DEMO COMPLETE")
    print("="*60)

def interactive_prediction():
    """
    Interactive mode for testing predictions
    """
    print("\n" + "="*60)
    print("INTERACTIVE HEIGHT PREDICTION")
    print("="*60)
    
    predictor = HeightPredictor()
    
    print("\nEnter child information to predict height")
    print("(Type 'quit' to exit)")
    
    while True:
        print("\n" + "-"*60)
        try:
            age_input = input("Enter age (0-18 years): ").strip()
            if age_input.lower() == 'quit':
                break
            
            age = float(age_input)
            
            if age < 0 or age > 18:
                print("⚠ Age must be between 0 and 18 years")
                continue
            
            gender = input("Enter gender (boy/girl): ").strip().lower()
            
            if gender not in ['boy', 'girl']:
                print("⚠ Gender must be 'boy' or 'girl'")
                continue
            
            # Make prediction
            height = predictor.predict(age, gender)
            height_inches = height / 2.54
            height_feet = int(height_inches // 12)
            height_remaining_inches = height_inches % 12
            
            print(f"\n📏 Predicted Height:")
            print(f"   {height:.1f} cm")
            print(f"   {height_inches:.1f} inches")
            print(f"   {height_feet}' {height_remaining_inches:.1f}\"")
            
        except ValueError:
            print("⚠ Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
    
    print("\nThank you for using Height Predictor!")

if __name__ == "__main__":
    import sys
    
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_prediction()
    else:
        # Run demo by default
        demo_inference()
    
    print("\nTo run in interactive mode: python hw_inference.py --interactive")

