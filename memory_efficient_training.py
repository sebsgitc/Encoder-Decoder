#!/usr/bin/env python3
"""
Memory-efficient training script for vessel segmentation using TensorFlow Dataset API.
This script is specifically designed to work with limited GPU memory by loading and 
processing data in small chunks.
"""
import os
import sys
import numpy as np
import tensorflow as tf
import tifffile as tiff
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from scipy import ndimage

# Import from the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from semi_supervised_vessel_training import (
    create_seed_guidance, create_simplified_model, 
    dice_coefficient, dice_loss, focal_loss, combined_loss, 
    load_seed_points_from_csv, save_training_visualizations
)

# TensorFlow memory settings
# Set before anything else
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # More efficient memory allocator

def prepare_training_patches(volume, seed_points, patch_size=64, max_patches=1000, random_patches=True):
    """
    Extract training patches around seed points, saving them to disk to avoid memory issues
    """
    # Create temporary directory for patches
    temp_dir = "temp_patches"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create seed guidance for the full volume
    print("Creating initial guidance volume...")
    guidance = create_seed_guidance(volume.shape, seed_points, 'vessel', radius=3)
    
    patches_info = []
    patch_count = 0
    
    # Extract patches around seed points
    print(f"Extracting patches around {len(seed_points)} seed points...")
    for i, (x, y, z) in enumerate(tqdm(seed_points)):
        # Create multiple patches with different offsets around this seed point
        offsets = [
            (0,0,0), (5,5,5), (-5,-5,-5), (5,-5,5), (-5,5,-5),
            (10,0,0), (-10,0,0), (0,10,0), (0,-10,0), (0,0,10), (0,0,-10),
        ]
        
        for offset_x, offset_y, offset_z in offsets:
            cx, cy, cz = x + offset_x, y + offset_y, z + offset_z
            
            # Define bounds for patch extraction (respecting volume boundaries)
            z_min = max(0, cz - patch_size//2)
            z_max = min(volume.shape[0], z - patch_size//2 + patch_size)
            y_min = max(0, cy - patch_size//2)
            y_max = min(volume.shape[1], y - patch_size//2 + patch_size)
            x_min = max(0, cx - patch_size//2)
            x_max = min(volume.shape[2], x - patch_size//2 + patch_size)
            
            # Skip if patch is too small
            if (z_max - z_min < patch_size or 
                y_max - y_min < patch_size or 
                x_max - x_min < patch_size):
                continue
                
            # Extract patch
            vol_patch = volume[z_min:z_max, y_min:y_max, x_min:x_max]
            guide_patch = guidance[z_min:z_max, y_min:y_max, x_min:x_max]
            
            # Only use patches that have seed guidance
            if np.sum(guide_patch) == 0:
                continue
                
            # Create label using the guidance
            label_patch = guide_patch > 0
            
            # Save patches to disk as numpy files
            patch_id = f"patch_{patch_count:06d}"
            np.save(os.path.join(temp_dir, f"{patch_id}_volume.npy"), vol_patch)
            np.save(os.path.join(temp_dir, f"{patch_id}_guidance.npy"), guide_patch)
            np.save(os.path.join(temp_dir, f"{patch_id}_label.npy"), label_patch)
            
            # Add to patch info
            patches_info.append(patch_id)
            patch_count += 1
            
            # Break if we've reached the max patches
            if patch_count >= max_patches:
                break
        
        # Break if we've reached the max patches
        if patch_count >= max_patches:
            break
    
    # Generate some random patches for background context
    if random_patches and patch_count < max_patches:
        print(f"Generating {max_patches - patch_count} random patches...")
        for i in range(max_patches - patch_count):
            z = np.random.randint(0, volume.shape[0] - patch_size)
            y = np.random.randint(0, volume.shape[1] - patch_size)
            x = np.random.randint(0, volume.shape[2] - patch_size)
            
            vol_patch = volume[z:z+patch_size, y:y+patch_size, x:x+patch_size]
            guide_patch = guidance[z:z+patch_size, y:y+patch_size, x:x+patch_size]
            
            # Create label using the guidance
            label_patch = guide_patch > 0
            
            # Save patches to disk as numpy files
            patch_id = f"patch_{patch_count:06d}"
            np.save(os.path.join(temp_dir, f"{patch_id}_volume.npy"), vol_patch)
            np.save(os.path.join(temp_dir, f"{patch_id}_guidance.npy"), guide_patch)
            np.save(os.path.join(temp_dir, f"{patch_id}_label.npy"), label_patch)
            
            # Add to patch info
            patches_info.append(patch_id)
            patch_count += 1
    
    print(f"Generated {patch_count} patches in {temp_dir}")
    return patches_info, temp_dir

def create_dataset_from_patches(patches_info, temp_dir, patch_size, batch_size=1, shuffle=True):
    """
    Create a TensorFlow dataset that loads patches from disk as needed
    """
    # Function to load a single patch from disk
    def load_patch(patch_id):
        patch_id = patch_id.numpy().decode('utf-8')
        vol_path = os.path.join(temp_dir, f"{patch_id}_volume.npy")
        guidance_path = os.path.join(temp_dir, f"{patch_id}_guidance.npy")
        label_path = os.path.join(temp_dir, f"{patch_id}_label.npy")
        
        # Load the patches
        vol_patch = np.load(vol_path)
        guidance_patch = np.load(guidance_path)
        label_patch = np.load(label_path)
        
        # Normalize volume
        vol_patch = vol_patch.astype(np.float32) / 255.0
        
        # Create input with guidance
        input_patch = np.stack([vol_patch, guidance_patch], axis=-1)
        
        # Ensure label has channel dimension
        label_patch = np.expand_dims(label_patch.astype(np.float32), axis=-1)
        
        return input_patch, label_patch
    
    # TensorFlow wrapper for the function
    def tf_load_patch(patch_id):
        input_patch, label_patch = tf.py_function(
            load_patch,
            [patch_id],
            [tf.float32, tf.float32]
        )
        # Set shapes explicitly
        input_patch.set_shape((patch_size, patch_size, patch_size, 2))
        label_patch.set_shape((patch_size, patch_size, patch_size, 1))
        return input_patch, label_patch
    
    # Create a dataset of patch IDs
    dataset = tf.data.Dataset.from_tensor_slices(patches_info)
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(patches_info)))
    
    # Map to actual data
    dataset = dataset.map(tf_load_patch, 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Memory-Efficient Vessel Segmentation Training")
    parser.add_argument("--data-dir", default="3d-stacks", help="Directory containing 3D volume data")
    parser.add_argument("--output-dir", default="output/memory_efficient", help="Output directory for results")
    parser.add_argument("--model-dir", default="models", help="Directory to save trained models")
    parser.add_argument("--patch-size", type=int, default=48, help="Size of training patches")
    parser.add_argument("--max-patches", type=int, default=500, help="Maximum number of patches to generate")
    parser.add_argument("--batch-size", type=int, default=0, help="Training batch size (0 for auto-detection)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--seed-points", default=None, help="Path to CSV file with seed points")
    
    args = parser.parse_args()
    
    # Create output directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.model_dir, f"vessel_model_efficient_{timestamp}.h5")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Configure GPUs and set up strategy for multi-GPU training
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Found {len(gpus)} GPUs")
    
    # Set up distribution strategy if multiple GPUs are available
    use_multi_gpu = len(gpus) > 1
    
    if use_multi_gpu:
        try:
            # Set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Create a MirroredStrategy for multi-GPU training
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {strategy.num_replicas_in_sync} GPUs with MirroredStrategy")
            
            # Force TensorFlow to use all GPUs
            if strategy.num_replicas_in_sync < len(gpus):
                print("Warning: Not all GPUs are being used. Recreating strategy...")
                # Clear any existing session
                tf.keras.backend.clear_session()
                
                # Explicitly set visible devices
                tf.config.set_visible_devices(gpus, 'GPU')
                
                # Recreate strategy
                strategy = tf.distribute.MirroredStrategy()
                print(f"Recreated strategy now using {strategy.num_replicas_in_sync} GPUs")
            
        except Exception as e:
            print(f"Error setting up MirroredStrategy: {e}")
            use_multi_gpu = False
            strategy = tf.distribute.get_strategy()  # Default strategy
            
            # Still configure individual GPUs for single-GPU use
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    print(f"Error configuring GPU {gpu.name}: {e}")
    else:
        # Single GPU or CPU
        strategy = tf.distribute.get_strategy()
        if len(gpus) == 1:
            # Configure the single GPU
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except Exception as e:
                print(f"Error configuring GPU {gpus[0].name}: {e}")
    
    # Calculate optimal batch size based on hardware and patch size
    def estimate_optimal_batch_size(patch_size, num_gpus):
        # Base calculation considering patch size
        if patch_size <= 32:
            base_batch = 4
        elif patch_size <= 48:
            base_batch = 2
        else:
            base_batch = 1
            
        # Scale by number of GPUs
        if num_gpus == 0:  # CPU only
            return max(1, base_batch // 2)
        elif num_gpus == 1:
            return base_batch
        else:
            # With multiple GPUs, increase batch size (with some caution)
            return base_batch * max(1, int(num_gpus * 0.8))
    
    # Set batch size
    if args.batch_size <= 0:
        # Auto-detect based on hardware
        batch_size = estimate_optimal_batch_size(
            args.patch_size, 
            strategy.num_replicas_in_sync  # Use actual number of GPUs from strategy
        )
        
        # With multi-GPU, ensure batch size is at least equal to number of GPUs
        if use_multi_gpu and batch_size < strategy.num_replicas_in_sync:
            batch_size = strategy.num_replicas_in_sync
    else:
        # User specified batch size
        batch_size = args.batch_size
    
    print(f"Using batch size: {batch_size} with {strategy.num_replicas_in_sync} GPU(s)")
    
    # Load volume
    volume_path = os.path.join(args.data_dir, "r01_", "r01_.8bit.tif")
    print(f"Loading volume from {volume_path}")
    volume = tiff.imread(volume_path)
    print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
    
    # Load seed points
    if args.seed_points and os.path.exists(args.seed_points):
        seed_points = load_seed_points_from_csv(args.seed_points)
    else:
        # Check for default seed points file
        default_seed_path = os.path.join("seed_points", "r01_seed_points.csv")
        if os.path.exists(default_seed_path):
            print(f"Using default seed points from {default_seed_path}")
            seed_points = load_seed_points_from_csv(default_seed_path)
        else:
            print(f"No seed points file found, using a small set of default points")
            seed_points = [
                (478, 323, 32),
                (372, 648, 45),
                (920, 600, 72),
                (420, 457, 24),
                (369, 326, 74)
            ]
    
    # Generate and save training patches
    print(f"Preparing up to {args.max_patches} training patches of size {args.patch_size}...")
    patches_info, temp_dir = prepare_training_patches(
        volume, 
        seed_points, 
        patch_size=args.patch_size, 
        max_patches=args.max_patches
    )
    
    # Split into training and validation sets
    np.random.shuffle(patches_info)
    val_size = int(len(patches_info) * 0.2)
    train_patches = patches_info[val_size:]
    val_patches = patches_info[:val_size]
    
    print(f"Using {len(train_patches)} patches for training and {len(val_patches)} for validation")
    
    # Create TensorFlow datasets with the optimal batch size
    train_dataset = create_dataset_from_patches(
        train_patches, 
        temp_dir, 
        args.patch_size, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_dataset = create_dataset_from_patches(
        val_patches, 
        temp_dir, 
        args.patch_size, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Create a memory-efficient model within the strategy scope if using multiple GPUs
    print("Creating memory-efficient model...")
    input_shape = (args.patch_size, args.patch_size, args.patch_size, 2)
    
    if use_multi_gpu:
        with strategy.scope():
            model = create_simplified_model(input_shape=input_shape, filters_base=16)
    else:
        model = create_simplified_model(input_shape=input_shape, filters_base=16)
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor='val_dice_coefficient',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_dice_coefficient',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            monitor='val_dice_coefficient',
            mode='max'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_dir, f"logs_{timestamp}")
        )
    ]
    
    # Train the model
    print(f"Starting training with batch size {batch_size} on {strategy.num_replicas_in_sync} GPU(s)...")
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=2
        )
        
        # Save training visualizations
        save_training_visualizations(
            history, 
            os.path.join(args.output_dir, f"training_{timestamp}")
        )
        
        print(f"Model trained successfully and saved to {model_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("\nTrying with a smaller model and reduced complexity...")
        
        # Clear session
        tf.keras.backend.clear_session()
        
        # Create an even simpler model - within strategy scope if using multiple GPUs
        if use_multi_gpu:
            with strategy.scope():
                model = create_simplified_model(input_shape=input_shape, filters_base=8)
        else:
            model = create_simplified_model(input_shape=input_shape, filters_base=8)
        
        # Use a minimal batch size, but still respect multi-GPU requirements
        fallback_batch_size = 1
        if use_multi_gpu:
            fallback_batch_size = strategy.num_replicas_in_sync
        
        # Reduce dataset complexity
        train_dataset = create_dataset_from_patches(
            train_patches[:min(200, len(train_patches))], 
            temp_dir, 
            args.patch_size, 
            batch_size=fallback_batch_size,
            shuffle=True
        )
        
        val_dataset = create_dataset_from_patches(
            val_patches[:min(50, len(val_patches))], 
            temp_dir, 
            args.patch_size, 
            batch_size=fallback_batch_size,
            shuffle=False
        )
        
        # Try training again
        try:
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=2
            )
            
            # Save training visualizations
            save_training_visualizations(
                history, 
                os.path.join(args.output_dir, f"training_fallback_{timestamp}")
            )
            
            print(f"Fallback model trained successfully and saved to {model_path}")
            
        except Exception as e2:
            print(f"Error during fallback training: {e2}")
            print("Please try reducing patch size or simplifying the model further.")
    
    # Clean up temporary files
    print(f"Cleaning up temporary files in {temp_dir}...")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("Done!")

if __name__ == "__main__":
    main()
