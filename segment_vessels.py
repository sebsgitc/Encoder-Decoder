#!/usr/bin/env python3
"""
Simple and reliable blood vessel segmentation using thresholding and region growing
"""
import os
import numpy as np
import SimpleITK as sitk
import tifffile as tiff
import argparse
import time

def load_volume(filepath):
    """Load 3D volume from file"""
    print(f"Loading volume from {filepath}")
    volume = tiff.imread(filepath)
    print(f"Loaded volume with shape {volume.shape}, dtype {volume.dtype}")
    return volume

def save_volume(volume, filepath):
    """Save 3D volume to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Saving volume to {filepath}")
    # Ensure volume is uint8 type for saving
    volume_uint8 = volume.astype(np.uint8)
    tiff.imwrite(filepath, volume_uint8)
    print(f"Volume saved successfully")

def segment_vessels(volume, seed_points, lower_threshold=None, upper_threshold=None):
    """Segment blood vessels using region growing from seed points"""
    print("\nSegmenting blood vessels...")
    start_time = time.time()
    
    # Ensure we're working with uint8 data
    if volume.dtype != np.uint8:
        print(f"Converting volume from {volume.dtype} to uint8")
        volume = volume.astype(np.uint8)
    
    # Determine thresholds if not provided
    if lower_threshold is None or upper_threshold is None:
        # Compute dynamic thresholds based on seed point intensities
        seed_values = []
        for x, y, z in seed_points:
            try:
                if 0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]:
                    val = volume[z, y, x]
                    seed_values.append(val)
            except IndexError:
                print(f"Warning: Seed point ({x},{y},{z}) is out of bounds")
        
        if seed_values:
            mean_val = np.mean(seed_values)
            std_val = np.std(seed_values)
            
            if lower_threshold is None:
                lower_threshold = max(0, int(mean_val - 1.5 * std_val))
            if upper_threshold is None:
                upper_threshold = min(255, int(mean_val + 1.5 * std_val))
    
    print(f"Using threshold range: [{lower_threshold}, {upper_threshold}]")
    
    # Invert volume if vessels are dark (lower intensity than surroundings)
    vessels_are_dark = True  # Assuming vessels are dark in CT scans
    if vessels_are_dark:
        print("Inverting volume (vessels are dark)")
        volume = np.subtract(255, volume, dtype=np.uint8)
        # Adjust thresholds accordingly
        old_lower = lower_threshold
        lower_threshold = 255 - upper_threshold
        upper_threshold = 255 - old_lower
    
    # Convert to SimpleITK for processing
    print("Converting to SimpleITK Image...")
    sitk_volume = sitk.GetImageFromArray(volume)
    
    # Apply median filter to reduce noise (without using bool type)
    print("Applying median filter...")
    try:
        smoothed = sitk.Median(sitk_volume, [3, 3, 3])
    except Exception as e:
        print(f"Error in median filter: {e}")
        print("Using original volume instead")
        smoothed = sitk_volume
    
    # Initialize segmentation as empty SimpleITK image
    print("Initializing segmentation...")
    segmentation = sitk.Image(smoothed.GetSize(), sitk.sitkUInt8)
    segmentation.CopyInformation(smoothed)  # Copy metadata
    
    # Process each seed point
    print(f"Processing {len(seed_points)} seed points...")
    valid_seeds = 0
    
    for i, (x, y, z) in enumerate(seed_points):
        print(f"Processing seed {i+1}/{len(seed_points)}: ({x},{y},{z})")
        
        # Skip if out of bounds
        if not (0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]):
            print(f"  Seed out of bounds, skipping")
            continue
        
        # Create seed for SimpleITK (z,y,x order)
        seed = [int(z), int(y), int(x)]
        
        # Check if seed is within threshold range
        try:
            seed_val = sitk.GetArrayFromImage(smoothed)[z, y, x]
            print(f"  Seed value: {seed_val}")
            
            if not (lower_threshold <= seed_val <= upper_threshold):
                print(f"  Seed value outside threshold range")
                # Adjust the thresholds for this specific seed
                local_lower = max(0, seed_val - 30)
                local_upper = min(255, seed_val + 30)
                print(f"  Using local thresholds: [{local_lower}, {local_upper}]")
            else:
                local_lower = lower_threshold
                local_upper = upper_threshold
            
            # Region growing from this seed
            print(f"  Running ConnectedThreshold...")
            seed_result = sitk.ConnectedThreshold(
                smoothed,
                seedList=[seed],
                lower=float(local_lower),
                upper=float(local_upper),
                replaceValue=1
            )
            
            # Check if anything was segmented
            seed_result_array = sitk.GetArrayFromImage(seed_result)
            segmented_voxels = np.sum(seed_result_array)
            
            if segmented_voxels > 0:
                print(f"  Segmented {segmented_voxels} voxels")
                valid_seeds += 1
                
                # Combine with overall segmentation using OR operation
                segmentation = sitk.Or(segmentation, seed_result)
            else:
                print(f"  No voxels segmented from this seed")
                
                # Try with wider range
                print(f"  Trying with wider range...")
                wider_lower = max(0, seed_val - 50)
                wider_upper = min(255, seed_val + 50)
                print(f"  Using wider thresholds: [{wider_lower}, {wider_upper}]")
                
                seed_result = sitk.ConnectedThreshold(
                    smoothed,
                    seedList=[seed],
                    lower=float(wider_lower),
                    upper=float(wider_upper),
                    replaceValue=1
                )
                
                # Check again
                seed_result_array = sitk.GetArrayFromImage(seed_result)
                segmented_voxels = np.sum(seed_result_array)
                
                if segmented_voxels > 0:
                    print(f"  Segmented {segmented_voxels} voxels with wider range")
                    valid_seeds += 1
                    segmentation = sitk.Or(segmentation, seed_result)
                else:
                    print(f"  Still no voxels segmented, skipping seed")
        
        except Exception as e:
            print(f"  Error processing seed: {e}")
    
    # Post-processing (only if we have valid segmentation)
    if valid_seeds > 0:
        print(f"Post-processing segmentation from {valid_seeds} valid seeds...")
        
        try:
            # Close small holes
            print("Filling holes...")
            closed = sitk.BinaryMorphologicalClosing(segmentation, [2, 2, 2])
            
            # Keep only largest components
            print("Identifying connected components...")
            components = sitk.ConnectedComponent(closed)
            relabeled = sitk.RelabelComponent(components)
            
            # Keep largest components (binary mask)
            print("Keeping largest components...")
            num_keep = min(10, sitk.GetNumberOfComponentsF(relabeled))
            print(f"Keeping {num_keep} largest components")
            
            final_seg = sitk.BinaryThreshold(
                relabeled, 
                lowerThreshold=1,
                upperThreshold=num_keep,
                insideValue=255,
                outsideValue=0
            )
            
            # Convert back to numpy array
            result_np = sitk.GetArrayFromImage(final_seg)
        
        except Exception as e:
            print(f"Error in post-processing: {e}")
            print("Using unprocessed segmentation")
            result_np = sitk.GetArrayFromImage(segmentation)
            result_np = result_np * 255  # Scale to 0-255 range
    else:
        print("No valid segmentation from any seed. Creating empty result.")
        result_np = np.zeros_like(volume, dtype=np.uint8)
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    total_voxels = np.size(result_np)
    segmented_voxels = np.sum(result_np > 0)
    percentage = segmented_voxels / total_voxels * 100
    
    print(f"Segmentation complete in {elapsed_time:.2f} seconds")
    print(f"Segmented {segmented_voxels} voxels ({percentage:.6f}% of volume)")
    
    return result_np

def main():
    parser = argparse.ArgumentParser(description="Blood vessel segmentation")
    parser.add_argument("--input", default="3d-stacks/r01_/r01_.8bit.tif", help="Input volume path")
    parser.add_argument("--output", default="output/simple_segmentation/r01_.8bit.tif", help="Output segmentation path")
    parser.add_argument("--lower", type=int, default=None, help="Lower threshold for segmentation")
    parser.add_argument("--upper", type=int, default=None, help="Upper threshold for segmentation")
    
    args = parser.parse_args()
    
    # Define seed points
    seed_points_r01 = [
        (478, 323, 32), (372, 648, 45), (920, 600, 72),
        (420, 457, 24), (369, 326, 74), (753, 417, 124),
        (755, 607, 174), (887, 507, 224), (305, 195, 274),
        (574, 476, 324), (380, 625, 374), (313, 660, 424),
        (100, 512, 610), (512, 20, 730), (512, 200, 820), (512, 400, 940)
    ]
    
    # Load volume
    volume = load_volume(args.input)
    
    # Segment blood vessels
    segmentation = segment_vessels(
        volume, 
        seed_points_r01,
        lower_threshold=args.lower,
        upper_threshold=args.upper
    )
    
    # Save segmentation
    save_volume(segmentation, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()
