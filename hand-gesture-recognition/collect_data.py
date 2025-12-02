"""
Hand Gesture Data Collection Tool
Captures images from webcam for training gesture recognition model
"""

import cv2
import os
import time
import numpy as np
from datetime import datetime

# Gesture classes
GESTURES = [
    'fist',
    'palm',
    'peace',
    'thumbs_up',
    'thumbs_down',
    'ok_sign'
]

# Configuration
IMG_SIZE = 224
SAMPLES_PER_GESTURE = 300
DATASET_DIR = 'dataset'
DELAY_BETWEEN_CAPTURES = 0.1  # seconds

class GestureDataCollector:
    """Collect hand gesture images from webcam"""
    
    def __init__(self):
        """Initialize the data collector"""
        self.cap = None
        self.current_gesture = None
        self.sample_count = 0
        
        # Create dataset directory
        os.makedirs(DATASET_DIR, exist_ok=True)
        
        print("="*60)
        print("HAND GESTURE DATA COLLECTION TOOL")
        print("="*60)
        print("\nGestures to collect:")
        for i, gesture in enumerate(GESTURES, 1):
            print(f"  {i}. {gesture.replace('_', ' ').title()}")
        print(f"\nTarget: {SAMPLES_PER_GESTURE} samples per gesture")
        print("="*60)
    
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
        
        print("✅ Camera initialized successfully")
        return True
    
    def create_gesture_directory(self, gesture):
        """Create directory for gesture images"""
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        return gesture_dir
    
    def preprocess_frame(self, frame):
        """Extract and preprocess hand region"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Define region of interest (center square)
        roi_size = min(h, w) * 3 // 4
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # Extract ROI
        roi = rgb_frame[y1:y2, x1:x2]
        
        # Resize to target size
        resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        
        return resized, (x1, y1, x2, y2)
    
    def draw_ui(self, frame, roi_coords, status_text, color=(0, 255, 0)):
        """Draw user interface on frame"""
        x1, y1, x2, y2 = roi_coords
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw instructions
        cv2.putText(frame, "Position hand in green box", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw status
        cv2.putText(frame, status_text, 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw sample count
        count_text = f"Samples: {self.sample_count}/{SAMPLES_PER_GESTURE}"
        cv2.putText(frame, count_text, 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw controls
        controls = "SPACE: Capture | Q: Quit | N: Next Gesture"
        cv2.putText(frame, controls, 
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        
        return frame
    
    def collect_gesture(self, gesture):
        """Collect samples for a specific gesture"""
        print(f"\n{'='*60}")
        print(f"Collecting: {gesture.replace('_', ' ').title()}")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("  1. Position your hand in the green box")
        print("  2. Press SPACE to capture (auto-capture enabled)")
        print("  3. Vary hand position, angle, and distance")
        print("  4. Press 'N' when done or after 300 samples\n")
        
        gesture_dir = self.create_gesture_directory(gesture)
        self.current_gesture = gesture
        self.sample_count = 0
        
        # Wait for user to be ready
        print("Press SPACE to start capturing...")
        
        ready = False
        last_capture_time = 0
        auto_capture = False
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Preprocess frame
            processed, roi_coords = self.preprocess_frame(frame)
            
            # Determine status
            if not ready:
                status = "Press SPACE to start"
                color = (0, 255, 255)
            elif auto_capture:
                status = f"AUTO-CAPTURING: {gesture.replace('_', ' ').title()}"
                color = (0, 255, 0)
            else:
                status = f"Capturing: {gesture.replace('_', ' ').title()}"
                color = (0, 255, 0)
            
            # Draw UI
            frame = self.draw_ui(frame, roi_coords, status, color)
            
            # Display frame
            cv2.imshow('Gesture Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n⚠️ Collection cancelled by user")
                return False
            
            elif key == ord('n'):
                print(f"\n✅ Completed {gesture}: {self.sample_count} samples")
                return True
            
            elif key == ord(' '):
                if not ready:
                    ready = True
                    auto_capture = True
                    last_capture_time = time.time()
                    print("▶️ Auto-capture started")
                else:
                    # Manual capture
                    self.save_sample(processed, gesture_dir)
            
            elif key == ord('a'):
                # Toggle auto-capture
                auto_capture = not auto_capture
                print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")
            
            # Auto-capture
            if ready and auto_capture:
                current_time = time.time()
                if current_time - last_capture_time >= DELAY_BETWEEN_CAPTURES:
                    self.save_sample(processed, gesture_dir)
                    last_capture_time = current_time
            
            # Check if enough samples collected
            if self.sample_count >= SAMPLES_PER_GESTURE:
                print(f"\n✅ Target reached: {self.sample_count} samples")
                print("Press 'N' to continue to next gesture")
                auto_capture = False
        
        return True
    
    def save_sample(self, image, gesture_dir):
        """Save captured image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.current_gesture}_{timestamp}.jpg"
        filepath = os.path.join(gesture_dir, filename)
        
        # Convert RGB to BGR for saving
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr_image)
        
        self.sample_count += 1
        
        if self.sample_count % 10 == 0:
            print(f"  Captured: {self.sample_count} samples")
    
    def run(self):
        """Main collection loop"""
        if not self.initialize_camera():
            return
        
        try:
            for gesture in GESTURES:
                success = self.collect_gesture(gesture)
                if not success:
                    break
                
                # Brief pause between gestures
                print("\nPreparing for next gesture...")
                time.sleep(2)
            
            print("\n" + "="*60)
            print("DATA COLLECTION COMPLETE!")
            print("="*60)
            self.print_summary()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Collection interrupted by user")
        
        finally:
            self.cleanup()
    
    def print_summary(self):
        """Print collection summary"""
        print("\nDataset Summary:")
        print("-"*60)
        
        total_samples = 0
        for gesture in GESTURES:
            gesture_dir = os.path.join(DATASET_DIR, gesture)
            if os.path.exists(gesture_dir):
                count = len([f for f in os.listdir(gesture_dir) if f.endswith('.jpg')])
                print(f"  {gesture.replace('_', ' ').title():15s}: {count:4d} samples")
                total_samples += count
        
        print("-"*60)
        print(f"  {'Total':15s}: {total_samples:4d} samples")
        print("\nNext Steps:")
        print("  1. Run 'python train_model.py' to train the model")
        print("  2. Use 'python realtime_demo.py' to test recognition")
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Resources released")

def main():
    """Main function"""
    print("\n🖐️  Hand Gesture Data Collection")
    print("Make sure you have good lighting and a clear background!\n")
    
    collector = GestureDataCollector()
    collector.run()

if __name__ == "__main__":
    main()

