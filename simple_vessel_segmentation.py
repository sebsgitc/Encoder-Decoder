#!/usr/bin/env python3
"""
Simple and reliable blood vessel segmentation using neighborhood region growing
"""
import os
import numpy as np
import SimpleITK as sitk
import tifffile as tiff
from tqdm import tqdm
import time
import argparse

def load_volume(filepath):
    """Load volume from file"""
    print(f"Loading volume from {filepath}")
    volume = tiff.imread(filepath)
    print(f"Loaded volume with shape {volume.shape}, dtype {volume.dtype}")
    return volume

def save_volume(volume, filepath):
    """Save volume to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Saving volume to {filepath}")
    tiff.imwrite(filepath, volume)
    print(f"Volume saved successfully")

def vessel_region_growing(volume, seed_points, intensity_range=30, threshold=30, max_iterations=100):
    """
    Basic region growing vessel segmentation with direct voxel examination
    This is a simpler approach that doesn't rely on complex algorithms
    """
    print("Starting vessel segmentation with direct region growing")
    
    # Time tracking
    start_time = time.time()
    
    # Create segmentation mask - use uint8 instead of bool for better compatibility
    segmentation = np.zeros_like(volume, dtype=np.uint8)
    
    # For each seed point
    for i, (x, y, z) in enumerate(seed_points):
        print(f"Processing seed {i+1}/{len(seed_points)}: ({x}, {y}, {z})")
        
        # Check if seed is within bounds
        if not (0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]):
            print(f"  Seed out of bounds, skipping")
            continue
        
        # Invert the volume (assuming vessels are dark)
        inverted_volume = 255 - volume
        
        # Get seed intensity
        seed_intensity = inverted_volume[z, y, x]
        print(f"  Seed intensity (inverted): {seed_intensity}")
        
        # Define intensity thresholds
        lower_threshold = max(0, seed_intensity - intensity_range)
        upper_threshold = min(255, seed_intensity + intensity_range)
        print(f"  Using intensity range: [{lower_threshold}, {upper_threshold}]")
        
        try:
            # Convert to SimpleITK
            sitk_volume = sitk.GetImageFromArray(inverted_volume)
            
            # Apply smoothing
            smoothed = sitk.CurvatureAnisotropicDiffusion(sitk_volume, 
                                                        timeStep=0.0625,
                                                        conductanceParameter=1.0,
                                                        numberOfIterations=5)
            
            # Create seed as SimpleITK point
            seed = [int(z), int(y), int(x)]  # z, y, x for SimpleITK
            
            # Region growing
            print(f"  Running connected threshold...")
            connected = sitk.ConnectedThreshold(
                smoothed,
                seedList=[seed],
                lower=lower_threshold,
                upper=upper_threshold,
                replaceValue=1
            )
            
            # Get result as numpy array - use uint8 instead of bool
            seed_segmentation = sitk.GetArrayFromImage(connected).astype(np.uint8)
            
            # Calculate size of segmentation
            segmented_voxels = np.sum(seed_segmentation)
            print(f"  Segmented {segmented_voxels} voxels")
            
            # If the segmentation is too small, try again with more permissive parameters
            if segmented_voxels < threshold:
                print(f"  Segmentation too small, trying with wider range...")
                lower_threshold = max(0, seed_intensity - intensity_range * 1.5)
                upper_threshold = min(255, seed_intensity + intensity_range * 1.5)
                print(f"  Using wider intensity range: [{lower_threshold}, {upper_threshold}]")
                
                connected = sitk.ConnectedThreshold(
                    smoothed,
                    seedList=[seed],
                    lower=lower_threshold,
                    upper=upper_threshold,
                    replaceValue=1
                )
                
                seed_segmentation = sitk.GetArrayFromImage(connected).astype(np.uint8)
                segmented_voxels = np.sum(seed_segmentation)
                print(f"  Segmented {segmented_voxels} voxels with wider range")
                
            # Add to overall segmentation - use logical_or with uint8 output
            segmentation = np.logical_or(segmentation, seed_segmentation).astype(np.uint8)
            
        except Exception as e:
            print(f"  Error processing seed: {str(e)}")
    
    # Convert to uint8 for saving
    segmentation = segmentation * 255
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    total_voxels = segmentation.size
    segmented_voxels = np.sum(segmentation > 0)
    percentage = segmented_voxels / total_voxels * 100
    
    print(f"Segmentation complete")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Total voxels: {total_voxels}")
    print(f"Segmented voxels: {segmented_voxels}")
    print(f"Percentage segmented: {percentage:.6f}%")
    
    return segmentation

def main():
    parser = argparse.ArgumentParser(description="Simple vessel segmentation")
    parser.add_argument("--input", default="3d-stacks/r01_/r01_.8bit.tif", help="Input file path")
    parser.add_argument("--output", default="output/simple_segmentation/vessels.tif", help="Output file path")
    parser.add_argument("--range", type=int, default=30, help="Intensity range around seed")
    args = parser.parse_args()
    
    # Define seed points
    seed_points = [
        (478, 323, 32), (372, 648, 45), (920, 600, 72),
        (420, 457, 24), (369, 326, 74), (753, 417, 124),
        (755, 607, 174), (887, 507, 224), (305, 195, 274),
        (574, 476, 324), (380, 625, 374), (313, 660, 424),
        (100, 512, 610), (512, 20, 730), (512, 200, 820), (512, 400, 940)
    ]
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load volume
    volume = load_volume(args.input)
    
    # Perform segmentation
    segmentation = vessel_region_growing(volume, seed_points, intensity_range=args.range)
    
    # Save result
    save_volume(segmentation, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()
