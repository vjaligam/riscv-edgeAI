"""
Create dummy dataset for testing the training pipeline
"""

import os
import numpy as np
from PIL import Image

# Gestures
GESTURES = ['fist', 'palm', 'peace', 'thumbs_up', 'thumbs_down', 'ok_sign']
DATASET_DIR = 'dataset'
SAMPLES_PER_GESTURE = 50  # Small dataset for quick testing
IMG_SIZE = 224

def create_dummy_gesture_image(gesture_idx):
    """Create a simple synthetic gesture image"""
    # Create base image
    img = np.random.randint(200, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # Add some pattern specific to each gesture (for differentiation)
    center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
    
    # Add colored circles/shapes to differentiate gestures
    y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    
    if gesture_idx == 0:  # fist
        mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
        img[mask] = [150, 100, 100]
    elif gesture_idx == 1:  # palm
        mask = (x - center_x)**2 + (y - center_y)**2 <= 60**2
        img[mask] = [100, 150, 100]
    elif gesture_idx == 2:  # peace
        mask1 = (x - center_x + 20)**2 + (y - center_y)**2 <= 20**2
        mask2 = (x - center_x - 20)**2 + (y - center_y)**2 <= 20**2
        img[mask1] = [100, 100, 150]
        img[mask2] = [100, 100, 150]
    elif gesture_idx == 3:  # thumbs_up
        mask = (x - center_x)**2 + (y - center_y - 30)**2 <= 40**2
        img[mask] = [150, 150, 100]
    elif gesture_idx == 4:  # thumbs_down
        mask = (x - center_x)**2 + (y - center_y + 30)**2 <= 40**2
        img[mask] = [150, 100, 150]
    elif gesture_idx == 5:  # ok_sign
        mask_outer = (x - center_x)**2 + (y - center_y)**2 <= 50**2
        mask_inner = (x - center_x)**2 + (y - center_y)**2 <= 25**2
        img[mask_outer] = [100, 150, 150]
        img[mask_inner] = [200, 200, 200]
    
    # Add some noise
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def create_dummy_dataset():
    """Create dummy dataset for testing"""
    print("Creating dummy dataset for testing...")
    print(f"Gestures: {len(GESTURES)}")
    print(f"Samples per gesture: {SAMPLES_PER_GESTURE}")
    
    for gesture_idx, gesture in enumerate(GESTURES):
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        
        print(f"Creating {gesture}...", end=" ")
        
        for i in range(SAMPLES_PER_GESTURE):
            # Create image
            img_array = create_dummy_gesture_image(gesture_idx)
            
            # Convert to PIL Image
            img = Image.fromarray(img_array)
            
            # Save
            filename = f"{gesture}_{i:04d}.jpg"
            filepath = os.path.join(gesture_dir, filename)
            img.save(filepath)
        
        print(f"✓ {SAMPLES_PER_GESTURE} images")
    
    print(f"\n✅ Dummy dataset created in '{DATASET_DIR}/'")
    print(f"Total: {len(GESTURES) * SAMPLES_PER_GESTURE} images")

if __name__ == "__main__":
    create_dummy_dataset()

