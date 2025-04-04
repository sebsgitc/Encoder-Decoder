#!/usr/bin/env python3
import os
import sys

# Set CUDA environment variables BEFORE importing TensorFlow
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu'
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Standard library imports that don't depend on CUDA
import numpy as np
import tifffile as tiff
import argparse
import time
from tqdm import tqdm

# Now import TensorFlow after environment variables are set
import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)
print("Python version:", sys.version)
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))

# Check GPU detection
gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {len(gpus)}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Memory growth enabled for GPUs")

def create_3d_unet(input_shape=(64, 64, 64, 1), filters_base=16):
    """
    Creates a 3D U-Net model for vessel segmentation
    
    Args:
        input_shape: Input dimensions (default: 64x64x64 chunks with 1 channel)
        filters_base: Base number of filters (multiplied in deeper layers)
    
    Returns:
        Compiled keras model
    """
    # Input layer
    inputs = layers.Input(input_shape)
    
    # Contracting path (encoder)
    # Level 1
    conv1 = layers.Conv3D(filters_base, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(filters_base, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    # Level 2
    conv2 = layers.Conv3D(filters_base*2, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(filters_base*2, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    # Level 3
    conv3 = layers.Conv3D(filters_base*4, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(filters_base*4, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    # Bridge
    conv4 = layers.Conv3D(filters_base*8, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv3D(filters_base*8, 3, activation='relu', padding='same')(conv4)
    
    # Expansive path (decoder)
    # Level 3
    up5 = layers.UpSampling3D(size=(2, 2, 2))(conv4)
    concat5 = layers.Concatenate()([up5, conv3])
    conv5 = layers.Conv3D(filters_base*4, 3, activation='relu', padding='same')(concat5)
    conv5 = layers.Conv3D(filters_base*4, 3, activation='relu', padding='same')(conv5)
    
    # Level 2
    up6 = layers.UpSampling3D(size=(2, 2, 2))(conv5)
    concat6 = layers.Concatenate()([up6, conv2])
    conv6 = layers.Conv3D(filters_base*2, 3, activation='relu', padding='same')(concat6)
    conv6 = layers.Conv3D(filters_base*2, 3, activation='relu', padding='same')(conv6)
    
    # Level 1
    up7 = layers.UpSampling3D(size=(2, 2, 2))(conv6)
    concat7 = layers.Concatenate()([up7, conv1])
    conv7 = layers.Conv3D(filters_base, 3, activation='relu', padding='same')(concat7)
    conv7 = layers.Conv3D(filters_base, 3, activation='relu', padding='same')(conv7)
    
    # Output layer
    outputs = layers.Conv3D(1, 1, activation='sigmoid')(conv7)
    
    # Create and compile model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def create_seed_guidance(volume_shape, seed_points, seed_type='vessel', radius=3):
    """
    Create a volume with seed points for guidance
    
    Args:
        volume_shape: Shape of the target volume (z, y, x)
        seed_points: List of (x, y, z) seed point coordinates
        seed_type: 'vessel' or 'background' to indicate seed type
        radius: Radius around each seed point to mark
        
    Returns:
        Numpy array with marked seed points
    """
    # Create empty guidance volume
    guidance = np.zeros(volume_shape, dtype=np.float32)
    
    # Set value based on seed type (1 for vessel, -1 for background)
    value = 1.0 if seed_type == 'vessel' else -1.0
    
    # Mark seed points with specified radius
    for x, y, z in seed_points:
        if (0 <= z < volume_shape[0] and 
            0 <= y < volume_shape[1] and 
            0 <= x < volume_shape[2]):
            
            # Define bounds for the seed region (respecting volume boundaries)
            z_min = max(0, z - radius)
            z_max = min(volume_shape[0], z + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(volume_shape[1], y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(volume_shape[2], x + radius + 1)
            
            # Create spherical region around seed (approximate)
            for zi in range(z_min, z_max):
                for yi in range(y_min, y_max):
                    for xi in range(x_min, x_max):
                        # Check if within radius
                        dist = np.sqrt((zi-z)**2 + (yi-y)**2 + (xi-x)**2)
                        if dist <= radius:
                            guidance[zi, yi, xi] = value
    
    return guidance

def generate_chunk_coordinates(volume_shape, chunk_size=64, overlap=8):
    """
    Generate coordinates for overlapping chunks to process large volumes
    
    Args:
        volume_shape: Shape of the volume (z, y, x)
        chunk_size: Size of each chunk
        overlap: Overlap between adjacent chunks
        
    Returns:
        List of (z_start, z_end, y_start, y_end, x_start, x_end) coordinates
    """
    z_steps = range(0, volume_shape[0], chunk_size - overlap)
    y_steps = range(0, volume_shape[1], chunk_size - overlap)
    x_steps = range(0, volume_shape[2], chunk_size - overlap)
    
    chunk_coords = []
    
    for z in z_steps:
        z_end = min(z + chunk_size, volume_shape[0])
        z_start = z
        
        for y in y_steps:
            y_end = min(y + chunk_size, volume_shape[1])
            y_start = y
            
            for x in x_steps:
                x_end = min(x + chunk_size, volume_shape[2])
                x_start = x
                
                # Only include chunk if it's the full size or at the edge
                chunk_coords.append((z_start, z_end, y_start, y_end, x_start, x_end))
    
    return chunk_coords

def process_chunk(model, chunk, guidance_chunk=None, threshold=0.5):
    """
    Process a single chunk through the model
    
    Args:
        model: Keras model for segmentation
        chunk: Image chunk to process
        guidance_chunk: Optional guidance from seed points
        threshold: Threshold for binary segmentation
        
    Returns:
        Binary segmentation of the chunk
    """
    # Normalize chunk to [0,1]
    chunk_norm = chunk.astype(np.float32) / 255.0
    
    # Add guidance channel if provided
    if guidance_chunk is not None:
        chunk_input = np.stack([chunk_norm, guidance_chunk], axis=-1)
    else:
        chunk_input = chunk_norm[..., np.newaxis]
    
    # Ensure chunk has the right shape [1, D, H, W, C]
    chunk_input = np.expand_dims(chunk_input, axis=0)
    
    # Run inference
    prediction = model.predict(chunk_input, verbose=0)
    
    # Convert to binary mask
    binary_mask = (prediction[0, ..., 0] > threshold).astype(np.uint8) * 255
    
    return binary_mask

def segment_large_volume(model, volume, seed_points=None, chunk_size=64, overlap=8, threshold=0.5):
    """
    Segment a large volume by processing it in chunks
    
    Args:
        model: Keras model for segmentation
        volume: 3D volume to segment
        seed_points: List of (x, y, z) seed point coordinates
        chunk_size: Size of each chunk to process
        overlap: Overlap between chunks
        threshold: Threshold for binary segmentation
        
    Returns:
        Binary segmentation of the entire volume
    """
    # Initialize output segmentation
    segmentation = np.zeros_like(volume, dtype=np.uint8)
    
    # Create guidance volume if seed points are provided
    if seed_points and len(seed_points) > 0:
        print(f"Creating guidance from {len(seed_points)} seed points...")
        guidance = create_seed_guidance(volume.shape, seed_points, 'vessel', radius=3)
    else:
        guidance = None
    
    # Generate chunk coordinates
    chunk_coords = generate_chunk_coordinates(volume.shape, chunk_size, overlap)
    print(f"Processing volume in {len(chunk_coords)} chunks...")
    
    # Process each chunk
    for i, (z_start, z_end, y_start, y_end, x_start, x_end) in enumerate(tqdm(chunk_coords)):
        # Extract chunk
        chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Extract guidance for this chunk if available
        guidance_chunk = None
        if guidance is not None:
            guidance_chunk = guidance[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Skip processing if chunk is too small
        if any(dim < 16 for dim in chunk.shape):
            continue
            
        # Process chunk
        processed_chunk = process_chunk(model, chunk, guidance_chunk, threshold)
        
        # Determine valid region (exclude overlap except at edges)
        z_valid_start = overlap//2 if z_start > 0 else 0
        y_valid_start = overlap//2 if y_start > 0 else 0
        x_valid_start = overlap//2 if x_start > 0 else 0
        
        z_valid_end = chunk.shape[0] - overlap//2 if z_end < volume.shape[0] else chunk.shape[0]
        y_valid_end = chunk.shape[1] - overlap//2 if y_end < volume.shape[1] else chunk.shape[1]
        x_valid_end = chunk.shape[2] - overlap//2 if x_end < volume.shape[2] else chunk.shape[2]
        
        # Extract valid region from processed chunk
        valid_chunk = processed_chunk[
            z_valid_start:z_valid_end,
            y_valid_start:y_valid_end,
            x_valid_start:x_valid_end
        ]
        
        # Insert into final segmentation
        segmentation[
            z_start+z_valid_start:z_start+z_valid_end,
            y_start+y_valid_start:y_start+y_valid_end,
            x_start+x_valid_start:x_start+x_valid_end
        ] = valid_chunk
        
    return segmentation

def prepare_training_data(volume, seed_points, patch_size=64, num_patches=1000):
    """
    Prepare training data from volume and seed points
    
    Args:
        volume: 3D volume data
        seed_points: List of seed points (x, y, z)
        patch_size: Size of extracted patches
        num_patches: Number of patches to generate
        
    Returns:
        inputs, labels for training
    """
    # Create seed guidance for full volume
    guidance = create_seed_guidance(volume.shape, seed_points, 'vessel', radius=3)
    
    # Extract patches around seed points
    inputs = []
    labels = []
    
    # First, extract patches around seed points
    for x, y, z in seed_points:
        # Define bounds for patch extraction (respecting volume boundaries)
        z_min = max(0, z - patch_size//2)
        z_max = min(volume.shape[0], z + patch_size//2)
        y_min = max(0, y - patch_size//2)
        y_max = min(volume.shape[1], y + patch_size//2)
        x_min = max(0, x - patch_size//2)
        x_max = min(volume.shape[2], x + patch_size//2)
        
        # Skip if patch is too small
        if (z_max - z_min < patch_size//2 or 
            y_max - y_min < patch_size//2 or 
            x_max - x_min < patch_size//2):
            continue
            
        # Extract patch
        vol_patch = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        guide_patch = guidance[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Only use complete patches
        if vol_patch.shape[0] < patch_size or vol_patch.shape[1] < patch_size or vol_patch.shape[2] < patch_size:
            continue
            
        # Crop to exact patch size if needed
        vol_patch = vol_patch[:patch_size, :patch_size, :patch_size]
        guide_patch = guide_patch[:patch_size, :patch_size, :patch_size]
        
        # Normalize volume patch
        vol_patch = vol_patch.astype(np.float32) / 255.0
        
        # Create input with guidance
        input_patch = np.stack([vol_patch, guide_patch], axis=-1)
        
        # Create simple label based on guidance (for demonstration)
        # In a real implementation, you'd use actual ground truth segmentations
        # This is a simplified approach that treats seed points as positive examples
        label_patch = (guide_patch > 0).astype(np.float32)
        label_patch = np.expand_dims(label_patch, axis=-1)
        
        inputs.append(input_patch)
        labels.append(label_patch)
    
    # Generate additional random patches
    remaining = num_patches - len(inputs)
    if remaining > 0:
        for _ in range(remaining):
            z = np.random.randint(0, volume.shape[0] - patch_size)
            y = np.random.randint(0, volume.shape[1] - patch_size)
            x = np.random.randint(0, volume.shape[2] - patch_size)
            
            vol_patch = volume[z:z+patch_size, y:y+patch_size, x:x+patch_size]
            guide_patch = guidance[z:z+patch_size, y:y+patch_size, x:x+patch_size]
            
            # Normalize volume patch
            vol_patch = vol_patch.astype(np.float32) / 255.0
            
            # Create input with guidance
            input_patch = np.stack([vol_patch, guide_patch], axis=-1)
            
            # Create label based on guidance
            label_patch = (guide_patch > 0).astype(np.float32)
            label_patch = np.expand_dims(label_patch, axis=-1)
            
            inputs.append(input_patch)
            labels.append(label_patch)
    
    return np.array(inputs), np.array(labels)

def main():
    parser = argparse.ArgumentParser(description="Deep Learning-Based Blood Vessel Segmentation")
    parser.add_argument("--input", default="3d-stacks/r01_/r01_.8bit.tif", help="Input volume path")
    parser.add_argument("--output", default="output/deep_segmentation/result.tif", help="Output segmentation path")
    parser.add_argument("--chunk-size", type=int, default=64, help="Size of chunks for processing")
    parser.add_argument("--overlap", type=int, default=8, help="Overlap between chunks")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    parser.add_argument("--seed-count", type=int, default=8, help="Number of seed points to use (default 8)")
    parser.add_argument("--weights", default=None, help="Path to pre-trained weights")
    parser.add_argument("--train", action="store_true", help="Train the model before segmentation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--save-model", default="models/vessel_segmentation_model.h5", help="Path to save trained model")
    
    args = parser.parse_args()
    
    # Define seed points for vessel (using fewer points than traditional methods)
    # The DNN approach requires significantly fewer seed points (4-8 typically sufficient)
    all_seed_points = [
        (478, 323, 32),  # Upper right lung vessel
        (372, 648, 45),  # Lower left lung vessel
        (920, 600, 72),  # Right peripheral vessel
        (420, 457, 24),  # Central vessel
        (369, 326, 74),  # Upper left pulmonary vessel
        (753, 417, 124), # Right middle lobe vessel
        (755, 607, 174), # Lower right vessel branch
        (305, 195, 274)  # Upper branch
    ]
    
    # Use only the requested number of seed points
    seed_points = all_seed_points[:args.seed_count]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    
    # Load volume
    print(f"Loading volume from {args.input}...")
    volume = tiff.imread(args.input)
    print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
    
    # Create model
    print("Creating 3D U-Net model...")
    input_shape = (args.chunk_size, args.chunk_size, args.chunk_size, 1)
    if len(seed_points) > 0:
        # If using seed points, add guidance channel
        input_shape = (args.chunk_size, args.chunk_size, args.chunk_size, 2)
    
    model = create_3d_unet(input_shape=input_shape)
    
    # Train or load weights
    if args.train:
        print(f"Preparing training data from {len(seed_points)} seed points...")
        inputs, labels = prepare_training_data(
            volume, 
            seed_points, 
            patch_size=args.chunk_size, 
            num_patches=100
        )
        print(f"Training data: {inputs.shape}, labels: {labels.shape}")
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                args.save_model,
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        print(f"Training model for {args.epochs} epochs...")
        history = model.fit(
            inputs, labels,
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks
        )
        
        # Plot training history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.savefig(os.path.splitext(args.save_model)[0] + '_training_history.png')
        
        print(f"Model trained and saved to {args.save_model}")
    elif args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}...")
        model.load_weights(args.weights)
    else:
        print("No weights provided, using default initialization")
        print("NOTE: Without training, this model will produce random results!")
        print("This is intended as a demonstration of the architecture and approach.")
    
    # Print model summary
    model.summary()
    
    # Process volume in chunks
    start_time = time.time()
    print(f"Segmenting volume using {len(seed_points)} seed points...")
    segmentation = segment_large_volume(
        model, 
        volume, 
        seed_points=seed_points,
        chunk_size=args.chunk_size, 
        overlap=args.overlap,
        threshold=args.threshold
    )
    
    # Save segmentation
    print(f"Saving segmentation to {args.output}...")
    tiff.imwrite(args.output, segmentation)
    
    # Print statistics
    elapsed_time = time.time() - start_time
    segmented_voxels = np.sum(segmentation > 0)
    percentage = segmented_voxels / np.prod(segmentation.shape) * 100
    
    print(f"Segmentation complete in {elapsed_time:.2f} seconds")
    print(f"Segmented {segmented_voxels} voxels ({percentage:.4f}% of volume)")
    print(f"Used {len(seed_points)} seed points (vs 16+ for traditional methods)")
    
    # Comparison with traditional methods
    print("\nComparison with traditional methods:")
    print("1. Seed points needed: 4-8 (DNN) vs 16+ (traditional)")
    print("2. Background points: Not required (DNN) vs Sometimes needed (traditional)")
    print("3. Processing approach: Parallel chunks (DNN) vs Sequential region growing (traditional)")
    print("4. Advantages: Handles vessel branching better, less sensitive to parameter tuning")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
