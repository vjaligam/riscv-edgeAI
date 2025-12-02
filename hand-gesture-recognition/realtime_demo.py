"""
Real-time Hand Gesture Recognition Demo
Live webcam demonstration of trained gesture recognition model
"""

import cv2
import numpy as np
import json
import os
import tensorflow as tf
from collections import deque
import time

# Configuration
MODEL_PATH = 'models/gesture_model.keras'
LABELS_PATH = 'models/gesture_labels.json'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_FRAMES = 10  # Number of frames for prediction smoothing

class GestureRecognizer:
    """Real-time gesture recognition from webcam"""
    
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        """Initialize the recognizer"""
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = []
        self.cap = None
        self.prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)
        
        print("="*60)
        print("REAL-TIME HAND GESTURE RECOGNITION")
        print("="*60)
    
    def load_model(self):
        """Load the trained model"""
        print("\nLoading model...")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Error: Model not found at '{self.model_path}'")
            print("   Please run 'python train_model.py' first")
            return False
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
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
        print(f"   Gestures: {', '.join([l.replace('_', ' ').title() for l in self.labels])}")
        
        return True
    
    def initialize_camera(self):
        """Initialize webcam"""
        print("\nInitializing camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open webcam")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized")
        return True
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Define region of interest (center square)
        roi_size = min(h, w) * 3 // 4
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resize and normalize
        resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype('float32') / 255.0
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data, (x1, y1, x2, y2)
    
    def predict_gesture(self, frame):
        """Predict gesture from frame"""
        input_data, roi_coords = self.preprocess_frame(frame)
        
        # Make prediction
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        # Add to buffer for smoothing
        self.prediction_buffer.append(predictions)
        
        # Average predictions over buffer
        if len(self.prediction_buffer) > 0:
            smoothed_predictions = np.mean(self.prediction_buffer, axis=0)
        else:
            smoothed_predictions = predictions
        
        # Get best prediction
        confidence = np.max(smoothed_predictions)
        gesture_idx = np.argmax(smoothed_predictions)
        gesture_name = self.labels[gesture_idx]
        
        return gesture_name, confidence, roi_coords, smoothed_predictions
    
    def draw_ui(self, frame, gesture, confidence, roi_coords, predictions):
        """Draw UI overlay on frame"""
        x1, y1, x2, y2 = roi_coords
        h, w = frame.shape[:2]
        
        # Determine color based on confidence
        if confidence >= CONFIDENCE_THRESHOLD:
            color = (0, 255, 0)  # Green
            status = "DETECTED"
        else:
            color = (0, 165, 255)  # Orange
            status = "LOW CONFIDENCE"
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw title
        title = "Hand Gesture Recognition"
        cv2.putText(frame, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw gesture name (large)
        gesture_display = gesture.replace('_', ' ').title()
        text_size = cv2.getTextSize(gesture_display, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 150
        
        # Draw background for text
        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, gesture_display, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Draw confidence
        conf_text = f"{status} ({confidence*100:.1f}%)"
        cv2.putText(frame, conf_text, (text_x, text_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw prediction bars
        bar_x = 10
        bar_y = 80
        bar_width = 200
        bar_height = 20
        bar_spacing = 5
        
        cv2.putText(frame, "Predictions:", (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, (label, pred) in enumerate(zip(self.labels, predictions)):
            y = bar_y + i * (bar_height + bar_spacing)
            
            # Background bar
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height),
                         (50, 50, 50), -1)
            
            # Prediction bar
            fill_width = int(bar_width * pred)
            bar_color = color if i == np.argmax(predictions) else (100, 100, 100)
            cv2.rectangle(frame, (bar_x, y), (bar_x + fill_width, y + bar_height),
                         bar_color, -1)
            
            # Label and percentage
            label_text = f"{label.replace('_', ' ').title()[:12]}: {pred*100:.1f}%"
            cv2.putText(frame, label_text, (bar_x + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw instructions
        instructions = [
            "Controls:",
            "Q - Quit",
            "S - Screenshot",
            "R - Reset buffer"
        ]
        
        inst_x = w - 200
        inst_y = 30
        for i, inst in enumerate(instructions):
            cv2.putText(frame, inst, (inst_x, inst_y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main recognition loop"""
        if not self.load_model():
            return
        
        if not self.initialize_camera():
            return
        
        print("\n" + "="*60)
        print("DEMO STARTED")
        print("="*60)
        print("\nInstructions:")
        print("  1. Position your hand in the green/orange box")
        print("  2. Perform gestures and see real-time recognition")
        print("  3. Press 'Q' to quit, 'S' for screenshot\n")
        
        fps_counter = deque(maxlen=30)
        frame_count = 0
        
        try:
            while self.cap.isOpened():
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Predict gesture
                gesture, confidence, roi_coords, predictions = self.predict_gesture(frame)
                
                # Draw UI
                frame = self.draw_ui(frame, gesture, confidence, roi_coords, predictions)
                
                # Calculate and display FPS
                fps = 1.0 / (time.time() - start_time)
                fps_counter.append(fps)
                avg_fps = np.mean(fps_counter)
                
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Gesture Recognition Demo', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Exiting...")
                    break
                
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                
                elif key == ord('r'):
                    # Reset prediction buffer
                    self.prediction_buffer.clear()
                    print("🔄 Prediction buffer reset")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Resources released")

def main():
    """Main function"""
    recognizer = GestureRecognizer()
    recognizer.run()

if __name__ == "__main__":
    main()

