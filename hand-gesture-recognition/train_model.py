"""
Hand Gesture Recognition Model Training
Train CNN model using MobileNetV2 with transfer learning
Optimized for RISC-V Edge deployment
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
DATASET_DIR = 'dataset'
MODEL_DIR = 'models'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

class GestureModelTrainer:
    """Train and export gesture recognition model"""
    
    def __init__(self):
        """Initialize the trainer"""
        self.model = None
        self.history = None
        self.class_names = []
        self.num_classes = 0
        
        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        print("="*70)
        print("HAND GESTURE RECOGNITION - MODEL TRAINING")
        print("="*70)
    
    def load_dataset(self):
        """Load and prepare dataset"""
        print("\n1. Loading dataset...")
        
        # Check if dataset exists
        if not os.path.exists(DATASET_DIR):
            print(f"❌ Error: Dataset directory '{DATASET_DIR}' not found!")
            print("   Please run 'python collect_data.py' first")
            return False
        
        # Get class names
        self.class_names = sorted([d for d in os.listdir(DATASET_DIR) 
                                   if os.path.isdir(os.path.join(DATASET_DIR, d))])
        
        if len(self.class_names) == 0:
            print("❌ Error: No gesture classes found!")
            return False
        
        self.num_classes = len(self.class_names)
        
        print(f"✅ Found {self.num_classes} gesture classes:")
        for i, name in enumerate(self.class_names):
            gesture_dir = os.path.join(DATASET_DIR, name)
            num_images = len([f for f in os.listdir(gesture_dir) if f.endswith('.jpg')])
            print(f"   {i+1}. {name.replace('_', ' ').title():15s} - {num_images:4d} images")
        
        return True
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        print("\n2. Creating data generators with augmentation...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # 20% for validation
        )
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Load validation data (no augmentation)
        self.val_generator = train_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"✅ Training samples: {self.train_generator.samples}")
        print(f"✅ Validation samples: {self.val_generator.samples}")
        
        return True
    
    def create_model(self):
        """Create CNN model (training from scratch for testing)"""
        print("\n3. Creating model architecture...")
        print("   (Training from scratch - no pre-trained weights needed)")
        
        # Create custom CNN model
        self.model = keras.Sequential([
            # Input layer
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Print model info
        total_params = self.model.count_params()
        print(f"\nTotal Parameters: {total_params:,}")
        
        return True
    
    def train_model(self):
        """Train the model"""
        print("\n4. Training model...")
        print(f"   Epochs: {EPOCHS}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Learning rate: {LEARNING_RATE}\n")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(MODEL_DIR, 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            self.train_generator,
            epochs=EPOCHS,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("\n5. Evaluating model...")
        val_loss, val_accuracy = self.model.evaluate(self.val_generator, verbose=0)
        print(f"✅ Validation Loss: {val_loss:.4f}")
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        return True
    
    def plot_training_history(self):
        """Plot training history"""
        print("\n6. Generating training plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=150)
        print(f"✅ Saved: {os.path.join(MODEL_DIR, 'training_history.png')}")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Generate and plot confusion matrix"""
        print("\n7. Generating confusion matrix...")
        
        # Get predictions
        self.val_generator.reset()
        predictions = self.model.predict(self.val_generator, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.val_generator.classes
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[name.replace('_', '\n') for name in self.class_names],
                    yticklabels=[name.replace('_', '\n') for name in self.class_names])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150)
        print(f"✅ Saved: {os.path.join(MODEL_DIR, 'confusion_matrix.png')}")
        plt.close()
        
        # Print classification report
        print("\nClassification Report:")
        print("-"*70)
        print(classification_report(y_true, y_pred, 
                                   target_names=self.class_names,
                                   digits=4))
    
    def export_models(self):
        """Export models for deployment"""
        print("\n8. Exporting models...")
        
        # Save Keras model
        keras_path = os.path.join(MODEL_DIR, 'gesture_model.keras')
        self.model.save(keras_path)
        print(f"✅ Saved Keras model: {keras_path}")
        
        # Export to TensorFlow Lite
        print("   Converting to TensorFlow Lite...")
        
        # Standard TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(MODEL_DIR, 'gesture_model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ Saved TFLite model: {tflite_path}")
        
        # Quantized TFLite (optimized for edge)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_quant_model = converter.convert()
        
        tflite_quant_path = os.path.join(MODEL_DIR, 'gesture_model_quantized.tflite')
        with open(tflite_quant_path, 'wb') as f:
            f.write(tflite_quant_model)
        print(f"✅ Saved Quantized TFLite model: {tflite_quant_path}")
        
        # Save class labels
        labels_path = os.path.join(MODEL_DIR, 'gesture_labels.json')
        with open(labels_path, 'w') as f:
            json.dump({'labels': self.class_names}, f, indent=2)
        print(f"✅ Saved labels: {labels_path}")
        
        # Print file sizes
        print("\nModel File Sizes:")
        print(f"   Keras model:      {os.path.getsize(keras_path) / 1024 / 1024:.2f} MB")
        print(f"   TFLite model:     {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
        print(f"   Quantized TFLite: {os.path.getsize(tflite_quant_path) / 1024 / 1024:.2f} MB")
    
    def run(self):
        """Main training pipeline"""
        try:
            # Load dataset
            if not self.load_dataset():
                return
            
            # Create data generators
            if not self.create_data_generators():
                return
            
            # Create model
            if not self.create_model():
                return
            
            # Train model
            if not self.train_model():
                return
            
            # Plot results
            self.plot_training_history()
            self.plot_confusion_matrix()
            
            # Export models
            self.export_models()
            
            print("\n" + "="*70)
            print("TRAINING COMPLETE!")
            print("="*70)
            print("\nNext Steps:")
            print("  1. Run 'python realtime_demo.py' to test live recognition")
            print("  2. Deploy 'gesture_model_quantized.tflite' to RISC-V device")
            print("  3. Use 'hw_inference.py' for edge deployment")
            
        except Exception as e:
            print(f"\n❌ Error during training: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    trainer = GestureModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main()

