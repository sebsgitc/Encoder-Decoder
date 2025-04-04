#!/usr/bin/env python3
"""
Prepare training data for vessel segmentation by creating patches from the volume
and using the existing segmentation method to generate ground truth labels.
"""
import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import argparse
import SimpleITK as sitk
import random

def load_volume(filepath):
    """Load 3D volume from file"""
    print(f"Loading volume from {filepath}")
    volume = tiff.imread(filepath)
    print(f"Loaded volume with shape {volume.shape}, dtype {volume.dtype}")
    return volume

def save_patches(patches, labels, output_dir):
    """Save extracted patches as training data"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    print(f"Saving {len(patches)} patches to {output_dir}")
    for i, (patch, label) in enumerate(zip(patches, labels)):
        input_path = os.path.join(output_dir, "inputs", f"patch_{i:05d}.npy")
        label_path = os.path.join(output_dir, "labels", f"patch_{i:05d}.npy")
        
        # Save as numpy arrays for efficient loading during training
        np.save(input_path, patch)
        np.save(label_path, label)
    
    print(f"Saved {len(patches)} training samples")
    
    # Save patch info
    with open(os.path.join(output_dir, "patch_info.txt"), 'w') as f:
        f.write(f"Total patches: {len(patches)}\n")
        f.write(f"Patch shape: {patches[0].shape}\n")
        f.write(f"Label shape: {labels[0].shape}\n")

def create_patches_with_gt(volume, seed_points, patch_size=64, num_patches=1000, output_dir="datasets/vessel_patches"):
    """
    Create training patches from volume.
    Uses seed points to ensure generated patches contain vessels.
    For each patch, also creates a ground truth segmentation using region growing.
    """
    print(f"Creating {num_patches} training patches with size {patch_size}^3")
    
    # Data containers
    patches = []
    labels = []
    
    # Initialize SimpleITK for ground truth generation
    volume_sitk = sitk.GetImageFromArray(volume)
    
    # Half of patches centered around seed points to ensure vessel coverage
    num_seed_patches = num_patches // 2
    random.shuffle(seed_points)  # Randomize seed points
    
    print("Generating patches centered on seed points...")
    for i, (x, y, z) in enumerate(tqdm(seed_points[:num_seed_patches])):
        # Skip if out of bounds
        if not (patch_size//2 <= z < volume.shape[0] - patch_size//2 and 
                patch_size//2 <= y < volume.shape[1] - patch_size//2 and 
                patch_size//2 <= x < volume.shape[2] - patch_size//2):
            continue
            
        # Extract patch centered on seed point
        patch = volume[z-patch_size//2:z+patch_size//2, 
                       y-patch_size//2:y+patch_size//2, 
                       x-patch_size//2:x+patch_size//2]
        
        # Create ground truth using SimpleITK region growing
        # First invert the volume (vessels are dark)
        inverted_patch = 255 - patch
        patch_sitk = sitk.GetImageFromArray(inverted_patch)
        
        # Apply smoothing
        smoothed = sitk.CurvatureAnisotropicDiffusion(patch_sitk, 
                                                     timeStep=0.0625,
                                                     conductanceParameter=1.0,
                                                     numberOfIterations=3)
        
        # Create seed in center of patch
        seed = [patch_size//2, patch_size//2, patch_size//2]  # z,y,x for SimpleITK
        
        # Get seed value for thresholds
        seed_val = sitk.GetArrayFromImage(smoothed)[seed[0], seed[1], seed[2]]
        lower = max(0, seed_val - 30)
        upper = min(255, seed_val + 30)
        
        # Run region growing
        try:
            segmentation = sitk.ConnectedThreshold(
                smoothed,
                seedList=[seed],
                lower=lower,
                upper=upper,
                replaceValue=1
            )
            
            # Convert to numpy
            label = sitk.GetArrayFromImage(segmentation).astype(np.uint8)
            
            # Only keep patches with some segmentation
            if np.sum(label) > 10:  # Minimum 10 vessel voxels
                patches.append(patch)
                labels.append(label)
        except Exception as e:
            print(f"Error processing seed at ({x},{y},{z}): {e}")
    
    print(f"Generated {len(patches)} seed-based patches")
    
    # Rest of patches randomly sampled from volume
    remaining_patches = num_patches - len(patches)
    print(f"Generating {remaining_patches} random patches...")
    
    while len(patches) < num_patches:
        # Random position (with padding to ensure full patch fits)
        z = random.randint(patch_size//2, volume.shape[0] - patch_size//2 - 1)
        y = random.randint(patch_size//2, volume.shape[1] - patch_size//2 - 1)
        x = random.randint(patch_size//2, volume.shape[2] - patch_size//2 - 1)
        
        # Extract patch
        patch = volume[z-patch_size//2:z+patch_size//2, 
                       y-patch_size//2:y+patch_size//2, 
                       x-patch_size//2:x+patch_size//2]
        
        # Create ground truth using similar approach
        inverted_patch = 255 - patch
        patch_sitk = sitk.GetImageFromArray(inverted_patch)
        
        # Apply smoothing
        smoothed = sitk.CurvatureAnisotropicDiffusion(patch_sitk, 
                                                     timeStep=0.0625,
                                                     conductanceParameter=1.0,
                                                     numberOfIterations=3)
        
        # Try to automatically find a good seed point (area with low value in original image = vessel)
        min_idx = np.unravel_index(np.argmin(patch), patch.shape)
        seed = [min_idx[0], min_idx[1], min_idx[2]]  # Already in z,y,x order
        
        # Get seed value for thresholds
        try:
            seed_val = sitk.GetArrayFromImage(smoothed)[seed[0], seed[1], seed[2]]
            lower = max(0, seed_val - 30)
            upper = min(255, seed_val + 30)
            
            # Run region growing
            segmentation = sitk.ConnectedThreshold(
                smoothed,
                seedList=[seed],
                lower=lower,
                upper=upper,
                replaceValue=1
            )
            
            # Convert to numpy
            label = sitk.GetArrayFromImage(segmentation).astype(np.uint8)
            
            # Only include if there's some vessel segmentation (not too much, not too little)
            vessel_percentage = np.sum(label) / np.size(label) * 100
            if 0.1 <= vessel_percentage <= 30:  # Between 0.1% and 30% vessel coverage
                patches.append(patch)
                labels.append(label)
        except Exception as e:
            # Just skip problematic patches
            pass
            
    print(f"Final dataset: {len(patches)} patches")
    
    # Balance the dataset to ensure a good mix of vessel/non-vessel samples
    print("Evaluating patch vessel percentages...")
    vessel_percentages = [np.sum(label)/np.size(label)*100 for label in labels]
    print(f"Average vessel percentage: {np.mean(vessel_percentages):.2f}%")
    print(f"Min vessel percentage: {np.min(vessel_percentages):.2f}%")
    print(f"Max vessel percentage: {np.max(vessel_percentages):.2f}%")
    
    # Save all patches
    save_patches(patches, labels, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for vessel segmentation")
    parser.add_argument("--input", default="3d-stacks/r01_/r01_.8bit.tif", 
                        help="Input volume path")
    parser.add_argument("--output-dir", default="datasets/vessel_patches", 
                        help="Output directory for training patches")
    parser.add_argument("--patch-size", type=int, default=64, 
                        help="Size of patches (cubic patches of this dimension)")
    parser.add_argument("--num-patches", type=int, default=1000, 
                        help="Number of patches to generate")
    
    args = parser.parse_args()
    
    # Define seed points for r01_ dataset
    seed_points_r01 = [
        (478, 323, 32), (372, 648, 45), (920, 600, 72),
        (420, 457, 24), (369, 326, 74), (753, 417, 124),
        (755, 607, 174), (887, 507, 224), (305, 195, 274),
        (574, 476, 324), (380, 625, 374), (313, 660, 424),
        (100, 512, 610), (512, 20, 730), (512, 200, 820), (512, 400, 940)
    ]
    
    # Load volume
    volume = load_volume(args.input)
    
    # Create training patches
    create_patches_with_gt(
        volume, 
        seed_points_r01, 
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        output_dir=args.output_dir
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
