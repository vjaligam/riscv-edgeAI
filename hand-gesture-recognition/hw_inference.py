"""
Hardware Inference Script for Hand Gesture Recognition
Lightweight inference for RISC-V edge devices using TensorFlow Lite
"""

import numpy as np
import json
import os
import time
try:
    import tensorflow as tf
except ImportError:
    import tflite_runtime.interpreter as tflite
    tf = None

# Configuration
MODEL_PATH = 'models/gesture_model_quantized.tflite'
LABELS_PATH = 'models/gesture_labels.json'
IMG_SIZE = 224

class EdgeGestureRecognizer:
    """Lightweight gesture recognizer for edge devices"""
    
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        """Initialize the edge recognizer"""
        self.model_path = model_path
        self.labels_path = labels_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        
        print("="*60)
        print("EDGE DEVICE GESTURE RECOGNITION")
        print("="*60)
    
    def load_model(self):
        """Load TFLite model"""
        print("\nLoading model...")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Error: Model not found at '{self.model_path}'")
            return False
        
        try:
            # Load TFLite model
            if tf:
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            else:
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
            
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ Model loaded: {self.model_path}")
            print(f"   Input shape: {self.input_details[0]['shape']}")
            print(f"   Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
        
        # Load labels
        if not os.path.exists(self.labels_path):
            print(f"❌ Error: Labels not found at '{self.labels_path}'")
            return False
        
        with open(self.labels_path, 'r') as f:
            data = json.load(f)
            self.labels = data['labels']
        
        print(f"✅ Loaded {len(self.labels)} gesture classes")
        
        return True
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Ensure image is the right size
        if image.shape[:2] != (IMG_SIZE, IMG_SIZE):
            # Note: In production, you'd use a proper resize function
            # For now, assume image is already correct size
            raise ValueError(f"Image must be {IMG_SIZE}x{IMG_SIZE}, got {image.shape[:2]}")
        
        # Normalize to [0, 1]
        normalized = image.astype('float32') / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data.astype(np.float32)
    
    def predict(self, image):
        """
        Predict gesture from image
        
        Parameters:
        -----------
        image : numpy.ndarray
            RGB image of shape (224, 224, 3)
        
        Returns:
        --------
        tuple : (gesture_name, confidence, all_predictions)
        """
        # Preprocess
        input_data = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Get best prediction
        confidence = float(np.max(predictions))
        gesture_idx = int(np.argmax(predictions))
        gesture_name = self.labels[gesture_idx]
        
        return gesture_name, confidence, predictions
    
    def predict_from_file(self, image_path):
        """Predict gesture from image file"""
        try:
            # Load image (you'll need to add PIL or opencv)
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            image = np.array(img)
        except ImportError:
            print("❌ PIL not available. Install: pip install Pillow")
            return None, 0.0, None
        except Exception as e:
            print(f"❌ Error loading image: {str(e)}")
            return None, 0.0, None
        
        return self.predict(image)
    
    def benchmark(self, num_iterations=100):
        """Benchmark inference speed"""
        print(f"\nRunning benchmark ({num_iterations} iterations)...")
        
        # Create dummy input
        dummy_image = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype('float32')
        
        # Warmup
        for _ in range(10):
            self.predict(dummy_image)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            self.predict(dummy_image)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = (total_time / num_iterations) * 1000  # ms
        fps = num_iterations / total_time
        
        print(f"✅ Benchmark complete:")
        print(f"   Total time: {total_time:.4f} seconds")
        print(f"   Average inference time: {avg_time:.2f} ms")
        print(f"   Throughput: {fps:.2f} FPS")
        
        return avg_time, fps

def demo_inference():
    """Demonstrate model inference"""
    print("="*60)
    print("GESTURE RECOGNITION - HARDWARE INFERENCE DEMO")
    print("="*60)
    
    # Initialize recognizer
    recognizer = EdgeGestureRecognizer()
    
    if not recognizer.load_model():
        return
    
    # Benchmark
    recognizer.benchmark(num_iterations=100)
    
    # Test with random images
    print("\n" + "-"*60)
    print("Testing with random images:")
    print("-"*60)
    
    for i in range(5):
        # Create random image
        random_image = np.random.rand(IMG_SIZE, IMG_SIZE, 3)
        
        # Predict
        gesture, confidence, predictions = recognizer.predict(random_image)
        
        print(f"\nTest {i+1}:")
        print(f"  Predicted: {gesture.replace('_', ' ').title()}")
        print(f"  Confidence: {confidence*100:.2f}%")
        print(f"  All predictions:")
        for label, pred in zip(recognizer.labels, predictions):
            print(f"    {label.replace('_', ' ').title():15s}: {pred*100:.2f}%")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nUsage in your application:")
    print("```python")
    print("from hw_inference import EdgeGestureRecognizer")
    print("import numpy as np")
    print("")
    print("# Initialize")
    print("recognizer = EdgeGestureRecognizer()")
    print("recognizer.load_model()")
    print("")
    print("# Predict from numpy array (224x224x3)")
    print("image = np.array(...) # Your image here")
    print("gesture, confidence, predictions = recognizer.predict(image)")
    print("print(f'Gesture: {gesture}, Confidence: {confidence:.2f}')")
    print("")
    print("# Or predict from file")
    print("gesture, confidence, _ = recognizer.predict_from_file('hand.jpg')")
    print("```")

def main():
    """Main function"""
    demo_inference()

if __name__ == "__main__":
    main()

