import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from PIL import Image
import glob
import gc
import signal
from attention_resunet_3d import attention_resunet_3d

# GPU and TF configuration
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
mixed_precision.set_global_policy('mixed_float16')

# Configure Ctrl+C handling
def signal_handler(sig, frame):
    print('\nCaught Ctrl+C! Cleaning up...')
    tf.keras.backend.clear_session()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Print GPU information
print("\nGPU Information:")
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs available: {len(physical_devices)}")
for gpu in physical_devices:
    print(f"GPU: {gpu}")
    tf.config.experimental.set_memory_growth(gpu, True)

def load_images_3d(image_folder):
    """Load 3D image volumes with debug prints"""
    print(f"\nLoading images from {image_folder}")
    image_paths = sorted([f for f in os.listdir(image_folder) if f.endswith(".tif")])
    print(f"Found {len(image_paths)} .tif files")
    
    for img_path in image_paths:
        full_path = os.path.join(image_folder, img_path)
        print(f"Processing: {img_path}")
        try:
            with Image.open(full_path) as img:
                n_frames = img.n_frames
                print(f"Number of frames: {n_frames}")
                
                batch_size = 16
                for i in range(0, n_frames, batch_size):
                    batch_end = min(i + batch_size, n_frames)
                    print(f"Processing batch {i} to {batch_end}")
                    
                    volume = np.empty((batch_end-i, 1024, 1024), dtype=np.float16)
                    for j in range(i, batch_end):
                        img.seek(j)
                        volume[j-i] = np.array(img, dtype=np.float16)
                    
                    volume = tf.cast(volume, tf.float16) / 255.0
                    yield volume
                    
                    # Force cleanup every few batches
                    if i % 128 == 0:
                        gc.collect()
                        
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

def load_dataset(subfolders, image_dir, batch_size=1):
    """Load dataset maintaining slice order"""
    print(f"\nLoading dataset with batch_size={batch_size}")
    print(f"Processing subfolders: {subfolders}")
    
    # First, load all chunks into memory
    all_chunks = []
    
    for subfolder in subfolders:
        print(f"\nProcessing subfolder: {subfolder}")
        subfolder_path = os.path.join(image_dir, subfolder)
        for volume in load_images_3d(subfolder_path):
            chunk_size = 256
            for i in range(0, volume.shape[0], chunk_size):
                chunk = volume[i:i+chunk_size]
                if chunk.shape[0] == chunk_size:
                    # Store both input and target
                    chunk = tf.expand_dims(chunk, -1)  # Add channel dimension here
                    all_chunks.append((chunk, chunk))
    
    total_steps = len(all_chunks)
    print(f"Total chunks loaded: {total_steps}")
    
    if total_steps == 0:
        raise ValueError("No valid chunks were loaded!")
    
    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(all_chunks)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    
    return dataset, total_steps

# Main execution
if __name__ == "__main__":
    print("Starting training script...")
    
    IMAGE_DIR = "3d-stacks"
    MODEL_PATH = "models/test_3d_model.h5"
    BATCH_SIZE = 2  # Smaller batch size for better memory management
    EPOCHS = 10
    
    # Define datasets
    train_subfolders = ["r01_", "r04_", "r07_"]
    val_subfolders = ["rL4_"]
    
    print("\nLoading datasets...")
    train_dataset, train_steps = load_dataset(train_subfolders, IMAGE_DIR, BATCH_SIZE)
    val_dataset, val_steps = load_dataset(val_subfolders, IMAGE_DIR, BATCH_SIZE)
    
    # Calculate proper steps based on actual data
    STEPS_PER_EPOCH = train_steps // (BATCH_SIZE * strategy.num_replicas_in_sync)
    VALIDATION_STEPS = max(val_steps // (BATCH_SIZE * strategy.num_replicas_in_sync), 1)
    
    print(f"\nTraining configuration:")
    print(f"Total training steps available: {train_steps}")
    print(f"Steps per epoch: {STEPS_PER_EPOCH}")
    print(f"Validation steps: {VALIDATION_STEPS}")
    
    # Create distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
    
    with strategy.scope():
        model = attention_resunet_3d(input_shape=(256, 1024, 1024, 1))
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        )
    ]
    
    print("\nStarting training...")
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            callbacks=callbacks,
            verbose=1
        )
        print("Training completed successfully!")
        model.save(MODEL_PATH)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        tf.keras.backend.clear_session()
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        tf.keras.backend.clear_session()
        sys.exit(1)