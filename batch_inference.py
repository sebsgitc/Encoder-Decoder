#!/usr/bin/env python3
"""
Optimized batch inference for vessel segmentation model.
Allows processing multiple chunks in batches for faster inference.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import ndimage
import pandas as pd  # Add this import at the top

# Import the core functions from the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from semi_supervised_vessel_training import create_seed_guidance, create_3d_unet, expand_seed_points, save_volume_slices

def prepare_chunk_batch(chunks, guidance_chunks=None):
    """
    Prepare a batch of chunks for inference
    """
    # Normalize chunks
    normalized_chunks = [chunk.astype(np.float32) / 255.0 for chunk in chunks]
    
    # Prepare input batches
    input_batch = []
    for i, chunk_norm in enumerate(normalized_chunks):
        if guidance_chunks is not None and i < len(guidance_chunks):
            chunk_input = np.stack([chunk_norm, guidance_chunks[i]], axis=-1)
        else:
            # Create diffuse guidance
            diffuse_guidance = np.zeros_like(chunk_norm, dtype=np.float32)
            center = chunk_norm.shape[0] // 2
            radius = chunk_norm.shape[0] // 4
            z, y, x = np.ogrid[:chunk_norm.shape[0], :chunk_norm.shape[1], :chunk_norm.shape[2]]
            dist = np.sqrt((z - center)**2 + (y - center)**2 + (x - center)**2)
            mask = dist <= radius
            diffuse_guidance[mask] = 0.1
            chunk_input = np.stack([chunk_norm, diffuse_guidance], axis=-1)
        
        # Ensure dimensions are multiples of 16
        original_shape = chunk_input.shape
        padded_sizes = []
        for j in range(3):
            if original_shape[j] % 16 != 0:
                new_size = ((original_shape[j] // 16) + 1) * 16
            else:
                new_size = original_shape[j]
            padded_sizes.append(new_size)
        
        # Create padded array
        padded_input = np.zeros((padded_sizes[0], padded_sizes[1], padded_sizes[2], 2), dtype=chunk_input.dtype)
        padded_input[:original_shape[0], :original_shape[1], :original_shape[2], :] = chunk_input
        input_batch.append(padded_input)
    
    # Stack the batch
    return np.stack(input_batch), [chunk.shape for chunk in chunks]

def process_batch_prediction(batch_prediction, original_shapes, threshold=0.15):
    """
    Process batch prediction results
    """
    processed_chunks = []
    
    for i in range(batch_prediction.shape[0]):
        prediction = batch_prediction[i, :original_shapes[i][0], :original_shapes[i][1], :original_shapes[i][2], 0]
        
        # Use multithreshold approach to improve vessel detection
        binary_mask = np.zeros_like(prediction, dtype=np.uint8)
        
        # First pass: detect strong vessel signals
        strong_vessels = (prediction > threshold * 2).astype(np.uint8)
        
        # Second pass: detect weaker vessel signals that connect to strong ones
        weak_vessels = (prediction > threshold).astype(np.uint8)
        
        # Create connectivity-based mask (only keep weak signals connected to strong ones)
        if np.sum(strong_vessels) > 0:
            # Use binary_fill_holes to connect nearby vessel segments
            combined = np.logical_or(strong_vessels, weak_vessels).astype(np.uint8)
            
            # Extract connected components
            labeled, num_features = ndimage.label(combined)
            
            # Only keep components that have at least one strong vessel voxel
            for j in range(1, num_features + 1):
                component = (labeled == j)
                if not np.any(np.logical_and(component, strong_vessels)):
                    # This component has no strong vessel voxel - remove it
                    combined[component] = 0
            
            # Fill small gaps for better connectivity
            combined = ndimage.binary_closing(combined, structure=np.ones((3,3,3))).astype(np.uint8)
            
            # Only keep larger components to reduce noise
            labeled, num_features = ndimage.label(combined)
            if num_features > 0:
                sizes = ndimage.sum(combined, labeled, range(1, num_features+1))
                
                # Keep components with more than minimum size
                min_size = 20
                mask = np.zeros_like(labeled, dtype=bool)
                for j, size in enumerate(sizes):
                    if size > min_size:
                        mask[labeled == j+1] = True
                        
                binary_mask = mask.astype(np.uint8) * 255
        else:
            # If no strong vessels detected, use weak vessels as fallback with stricter filtering
            if np.sum(weak_vessels) > 50:
                binary_mask = weak_vessels * 255
        
        processed_chunks.append(binary_mask)
    
    return processed_chunks

def segment_large_volume_batch(model, volume, seed_points=None, chunk_size=64, overlap=8, batch_size=4, threshold=0.15):
    """
    Segment a large volume by processing multiple chunks in batches
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
    
    print(f"Processing volume in {len(chunk_coords)} chunks with batch size {batch_size}")
    
    # Process chunks in batches
    for batch_start in tqdm(range(0, len(chunk_coords), batch_size)):
        batch_end = min(batch_start + batch_size, len(chunk_coords))
        batch_coords = chunk_coords[batch_start:batch_end]
        
        # Extract chunks for this batch
        batch_chunks = []
        batch_guidance = []
        for z_start, z_end, y_start, y_end, x_start, x_end in batch_coords:
            chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
            # Skip if any dimension is too small
            if any(dim < 16 for dim in chunk.shape):
                continue
            batch_chunks.append(chunk)
            
            if guidance is not None:
                guidance_chunk = guidance[z_start:z_end, y_start:y_end, x_start:x_end]
                batch_guidance.append(guidance_chunk)
        
        if not batch_chunks:
            continue
        
        # Prepare batch for inference
        input_batch, original_shapes = prepare_chunk_batch(batch_chunks, batch_guidance if guidance is not None else None)
        
        # Run batch inference
        try:
            batch_prediction = model.predict(input_batch, verbose=0)
            
            # Process batch predictions
            processed_chunks = process_batch_prediction(batch_prediction, original_shapes, threshold)
            
            # Insert processed chunks into segmentation
            for i, (z_start, z_end, y_start, y_end, x_start, x_end) in enumerate(batch_coords):
                if i >= len(processed_chunks):
                    continue
                
                processed_chunk = processed_chunks[i]
                chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Ensure the processed chunk matches the original chunk dimensions
                if processed_chunk.shape != chunk.shape:
                    temp_chunk = np.zeros_like(chunk, dtype=processed_chunk.dtype)
                    z_dim = min(processed_chunk.shape[0], chunk.shape[0])
                    y_dim = min(processed_chunk.shape[1], chunk.shape[1])
                    x_dim = min(processed_chunk.shape[2], chunk.shape[2])
                    temp_chunk[:z_dim, :y_dim, :x_dim] = processed_chunk[:z_dim, :y_dim, :x_dim]
                    processed_chunk = temp_chunk
                
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
            print(f"Error processing batch starting at index {batch_start}: {e}")
    
    return segmentation

# Import our custom functions
from semi_supervised_vessel_training import dice_coefficient, dice_loss, focal_loss, combined_loss

# Add the seed point loading function (same as in the main script)
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fast Batch Inference for Vessel Segmentation")
    parser.add_argument("--data-dir", default="3d-stacks", help="Directory containing 3D volume data")
    parser.add_argument("--output-dir", default="output/batch_inference", help="Output directory for results")
    parser.add_argument("--model-path", required=True, help="Path to saved model file")
    parser.add_argument("--chunk-size", type=int, default=64, help="Size of chunks for processing")
    parser.add_argument("--overlap", type=int, default=8, help="Overlap between chunks")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--threshold", type=float, default=0.15, help="Segmentation threshold")
    parser.add_argument("--volume", default="r01_", help="Volume name to process")
    parser.add_argument("--seed-points", default=None, help="Path to CSV file with seed points")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure GPUs to allow memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    
    print(f"Available GPUs: {len(physical_devices)}")
    
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
    
    # Define seed points based on volume name
    if args.seed_points and os.path.exists(args.seed_points):
        seed_points = load_seed_points_from_csv(args.seed_points)
    else:
        # Check if a default seed points file exists for this volume
        default_seed_path = os.path.join("seed_points", f"{args.volume}seed_points.csv")
        if os.path.exists(default_seed_path):
            print(f"Using default seed points from {default_seed_path}")
            seed_points = load_seed_points_from_csv(default_seed_path)
        else:
            # Fall back to the predefined seed points
            seed_points = []
            if args.volume == "r01_":
                # Use predefined seed points for r01_
                seed_points = [
                    (478, 323, 32),   # Upper right lung vessel
                    (372, 648, 45),   # Lower left lung vessel
                    (920, 600, 72),   # Right peripheral vessel
                    (420, 457, 24),   # Central vessel
                    (369, 326, 74),   # Upper left pulmonary vessel
                    (753, 417, 124),  # Right middle lobe vessel
                    (755, 607, 174),  # Lower right vessel branch
                    (305, 195, 274),  # Upper branch
                    (574, 476, 324),
                    (380, 625, 374),
                    (313, 660, 424),
                    (100, 512, 610),
                    (512, 20, 730),
                    (512, 200, 820),
                    (512, 400, 940)
                ]
            else:
                print(f"No predefined seed points for volume {args.volume}, using default seed points")
                # Use default set of seed points spread throughout the volume
                seed_points = [
                    (volume.shape[2]//2, volume.shape[1]//2, volume.shape[0]//4),
                    (volume.shape[2]//2, volume.shape[1]//2, volume.shape[0]//2),
                    (volume.shape[2]//2, volume.shape[1]//2, 3*volume.shape[0]//4),
                    (volume.shape[2]//4, volume.shape[1]//2, volume.shape[0]//2),
                    (3*volume.shape[2]//4, volume.shape[1]//2, volume.shape[0]//2),
                    (volume.shape[2]//2, volume.shape[1]//4, volume.shape[0]//2),
                    (volume.shape[2]//2, 3*volume.shape[1]//4, volume.shape[0]//2),
                ]
    
    if not seed_points:
        print("No valid seed points found. Using default generated seed points.")
        seed_points = [
            (volume.shape[2]//2, volume.shape[1]//2, volume.shape[0]//4),
            (volume.shape[2]//2, volume.shape[1]//2, volume.shape[0]//2),
            (volume.shape[2]//2, volume.shape[1]//2, 3*volume.shape[0]//4),
            (volume.shape[2]//4, volume.shape[1]//2, volume.shape[0]//2),
            (3*volume.shape[2]//4, volume.shape[1]//2, volume.shape[0]//2),
            (volume.shape[2]//2, volume.shape[1]//4, volume.shape[0]//2),
            (volume.shape[2]//2, 3*volume.shape[1]//4, volume.shape[0]//2),
        ]
    
    # Expand seed points for better coverage
    expanded_points = expand_seed_points(seed_points, volume.shape)
    
    # Start segmentation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting batch inference with {len(expanded_points)} expanded seed points")
    start_time = time.time()
    
    # Run batch segmentation
    segmentation = segment_large_volume_batch(
        model, 
        volume, 
        seed_points=expanded_points,
        chunk_size=args.chunk_size, 
        overlap=args.overlap,
        batch_size=args.batch_size,
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
    print(f"Used {len(seed_points)} seed points (expanded to {len(expanded_points)})")
    
    # Save the stats to a file
    with open(os.path.join(args.output_dir, f"stats_{args.volume}_{timestamp}.txt"), "w") as f:
        f.write(f"Segmentation Statistics:\n")
        f.write(f"- Volume: {args.volume}\n")
        f.write(f"- Processing time: {elapsed_time:.2f} seconds\n")
        f.write(f"- Segmented voxels: {segmented_voxels} ({percentage:.4f}% of volume)\n")
        f.write(f"- Seed points used: {len(seed_points)} (expanded to {len(expanded_points)})\n")
        f.write(f"- Chunk size: {args.chunk_size}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Threshold: {args.threshold}\n")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
