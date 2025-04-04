#!/usr/bin/env python3
"""
Single GPU inference script for vessel segmentation model.
This is a simplified version that runs on a single GPU to avoid multi-GPU conflicts.
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
import glob
import datetime
import pandas as pd
from scipy import ndimage

# Import the core functions from the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from semi_supervised_vessel_training import (
    create_seed_guidance, create_3d_unet, process_chunk, expand_seed_points, 
    save_volume_slices, dice_coefficient, dice_loss, focal_loss, combined_loss
)

def load_seed_points_from_csv(csv_path):
    """
    Load seed points from a CSV file
    
    The CSV file should have columns: 
    - Column 1: ID (ignored)
    - Column 2: X
    - Column 3: Y
    - Column 4: Slice (Z)
    
    Returns a list of (x, y, z) tuples
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        if not all(col in df.columns for col in ['X', 'Y', 'Slice']):
            print(f"Warning: CSV file {csv_path} does not have the expected columns (X, Y, Slice)")
            return []
        
        # Convert to list of tuples (x, y, z)
        seed_points = [(x, y, z) for x, y, z in zip(df['X'], df['Y'], df['Slice'])]
        
        print(f"Loaded {len(seed_points)} seed points from {csv_path}")
        return seed_points
    
    except Exception as e:
        print(f"Error loading seed points from {csv_path}: {e}")
        return []

def segment_volume(model, volume, seed_points=None, chunk_size=64, overlap=8, threshold=0.15):
    """
    Segment a volume using the provided model, optimized for a single GPU
    """
    # Initialize output segmentation
    segmentation = np.zeros_like(volume, dtype=np.uint8)
    
    # Create guidance volume if seed points are provided
    if seed_points and len(seed_points) > 0:
        print(f"Creating guidance from {len(seed_points)} seed points...")
        guidance = create_seed_guidance(volume.shape, seed_points, 'vessel', radius=3)
    else:
        guidance = None
    
    # Make sure chunk_size is divisible by 16
    if chunk_size % 16 != 0:
        chunk_size = ((chunk_size // 16) + 1) * 16
        print(f"Adjusted chunk size to {chunk_size} to ensure compatibility with dilated convolutions")
    
    # Generate chunk coordinates
    chunk_coords = []
    for z in range(0, volume.shape[0], chunk_size - overlap):
        z_end = min(z + chunk_size, volume.shape[0])
        z_start = max(0, z_end - chunk_size)
        for y in range(0, volume.shape[1], chunk_size - overlap):
            y_end = min(y + chunk_size, volume.shape[1])
            y_start = max(0, y_end - chunk_size)
            for x in range(0, volume.shape[2], chunk_size - overlap):
                x_end = min(x + chunk_size, volume.shape[2])
                x_start = max(0, x_end - chunk_size)
                chunk_coords.append((z_start, z_end, y_start, y_end, x_start, x_end))
    
    print(f"Processing volume in {len(chunk_coords)} chunks...")
    
    # Process each chunk
    for i, (z_start, z_end, y_start, y_end, x_start, x_end) in enumerate(tqdm(chunk_coords)):
        # Extract chunk
        chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Skip processing if chunk is too small
        if any(dim < 16 for dim in chunk.shape):
            continue
            
        # Extract guidance for this chunk if available
        guidance_chunk = None
        if guidance is not None:
            guidance_chunk = guidance[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Process chunk
        try:
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
            
        except Exception as e:
            print(f"Error processing chunk at z={z_start}-{z_end}, y={y_start}-{y_end}, x={x_start}-{x_end}: {e}")
            continue
    
    return segmentation

def main():
    parser = argparse.ArgumentParser(description="Single GPU Vessel Segmentation Inference")
    parser.add_argument("--data-dir", default="3d-stacks", help="Directory containing 3D volume data")
    parser.add_argument("--output-dir", default="output/single_gpu", help="Output directory for results")
    parser.add_argument("--model-path", required=True, help="Path to saved model file")
    parser.add_argument("--chunk-size", type=int, default=64, help="Size of chunks for processing")
    parser.add_argument("--overlap", type=int, default=8, help="Overlap between chunks")
    parser.add_argument("--threshold", type=float, default=0.15, help="Segmentation threshold")
    parser.add_argument("--volume", default="r01_", help="Volume name to process")
    parser.add_argument("--seed-points", default=None, help="Path to CSV file with seed points")
    parser.add_argument("--expand-seeds", action="store_true", help="Expand seed points for better coverage")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            # Only use the first GPU and limit memory growth
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"Using GPU: {physical_devices[0].name}")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU found. Using CPU.")
    
    # Load model with proper custom objects
    print(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'focal_loss': focal_loss,
            'combined_loss': combined_loss
        }
    )
    
    # Load volume
    volume_path = os.path.join(args.data_dir, args.volume, f"{args.volume}.8bit.tif")
    print(f"Loading volume from {volume_path}")
    volume = tiff.imread(volume_path)
    print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
    
    # Load seed points
    if args.seed_points and os.path.exists(args.seed_points):
        seed_points = load_seed_points_from_csv(args.seed_points)
    else:
        # Check if a default seed points file exists for this volume
        default_seed_path = os.path.join("seed_points", f"{args.volume}seed_points.csv")
        if os.path.exists(default_seed_path):
            print(f"Using default seed points from {default_seed_path}")
            seed_points = load_seed_points_from_csv(default_seed_path)
        else:
            # Fall back to generating seed points
            print(f"No seed points file found. Generating default seed points.")
            seed_points = [
                (volume.shape[2]//2, volume.shape[1]//2, volume.shape[0]//4),
                (volume.shape[2]//2, volume.shape[1]//2, volume.shape[0]//2),
                (volume.shape[2]//2, volume.shape[1]//2, 3*volume.shape[0]//4),
                (volume.shape[2]//4, volume.shape[1]//2, volume.shape[0]//2),
                (3*volume.shape[2]//4, volume.shape[1]//2, volume.shape[0]//2),
                (volume.shape[2]//2, volume.shape[1]//4, volume.shape[0]//2),
                (volume.shape[2]//2, 3*volume.shape[1]//4, volume.shape[0]//2),
            ]
    
    # Expand seed points if requested
    if args.expand_seeds:
        original_count = len(seed_points)
        seed_points = expand_seed_points(seed_points, volume.shape)
        print(f"Expanded {original_count} seed points to {len(seed_points)} points")
    
    # Start segmentation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting segmentation with {len(seed_points)} seed points")
    start_time = time.time()
    
    # Run segmentation
    segmentation = segment_volume(
        model, 
        volume, 
        seed_points=seed_points,
        chunk_size=args.chunk_size, 
        overlap=args.overlap,
        threshold=args.threshold
    )
    
    # Apply post-processing
    print("Applying post-processing to enhance vessel connectivity")
    
    # Clean up small isolated components
    labeled, num_features = ndimage.label(segmentation)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes > 50  # Only keep components with at least 50 voxels
    mask_sizes[0] = 0  # Background stays background
    segmentation = mask_sizes[labeled]
    
    # Apply morphological closing to connect nearby segments
    segmentation = ndimage.binary_closing(segmentation, structure=np.ones((3,3,3))).astype(np.uint8) * 255
    
    # Save segmentation
    seg_output_path = os.path.join(args.output_dir, f"segmentation_{args.volume}_{timestamp}.tif")
    print(f"Saving segmentation to {seg_output_path}")
    tiff.imwrite(seg_output_path, segmentation)
    
    # Save visualization
    save_volume_slices(
        volume, 
        segmentation, 
        os.path.join(args.output_dir, f"segmentation_{args.volume}_{timestamp}")
    )
    
    # Print statistics
    elapsed_time = time.time() - start_time
    segmented_voxels = np.sum(segmentation > 0)
    percentage = segmented_voxels / np.prod(segmentation.shape) * 100
    
    print(f"Segmentation complete in {elapsed_time:.2f} seconds")
    print(f"Segmented {segmented_voxels} voxels ({percentage:.4f}% of volume)")
    print(f"Used {len(seed_points)} seed points")
    
    # Save the stats to a file
    with open(os.path.join(args.output_dir, f"stats_{args.volume}_{timestamp}.txt"), "w") as f:
        f.write(f"Segmentation Statistics:\n")
        f.write(f"- Volume: {args.volume}\n")
        f.write(f"- Processing time: {elapsed_time:.2f} seconds\n")
        f.write(f"- Segmented voxels: {segmented_voxels} ({percentage:.4f}% of volume)\n")
        f.write(f"- Seed points used: {len(seed_points)}\n")
        f.write(f"- Chunk size: {args.chunk_size}\n")
        f.write(f"- Threshold: {args.threshold}\n")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
