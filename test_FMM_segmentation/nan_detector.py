"""Utility for detecting and handling NaN values in datasets."""

import os
import numpy as np
import tensorflow as tf
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from configuration import *

def check_dataset_for_nans(dataset, num_batches=10):
    """
    Analyze a dataset for NaN values to help diagnose issues.
    
    Args:
        dataset: TensorFlow dataset
        num_batches: Number of batches to check
        
    Returns:
        Dict containing statistics about NaNs found
    """
    nan_count = 0
    batch_count = 0
    total_elements = 0
    nan_locations = []
    
    print(f"Analyzing dataset for NaN values (checking {num_batches} batches)...")
    
    for batch in tqdm(dataset.take(num_batches)):
        batch_count += 1
        if len(batch) == 2:  # Regular dataset (images, masks)
            images, masks = batch
        else:  # Dataset with boundary weights (images, masks, weights)
            images, masks = batch[0], batch[1]
        
        # Check for NaNs in images
        has_nans = tf.math.reduce_any(tf.math.is_nan(images))
        has_infs = tf.math.reduce_any(tf.math.is_inf(images))
        
        if has_nans or has_infs:
            # Count NaN/Inf values
            nan_image_count = tf.reduce_sum(tf.cast(tf.math.is_nan(images), tf.int32)).numpy()
            inf_image_count = tf.reduce_sum(tf.cast(tf.math.is_inf(images), tf.int32)).numpy()
            nan_count += nan_image_count + inf_image_count
            
            print(f"Batch {batch_count}: Found {nan_image_count} NaNs and {inf_image_count} Infs")
            
            # Log locations of problematic batches
            nan_locations.append({
                'batch_idx': batch_count,
                'nan_count': nan_image_count,
                'inf_count': inf_image_count
            })
            
            # Visualize the first image with NaNs (helpful for debugging)
            for i in range(images.shape[0]):
                img = images[i].numpy()
                if np.isnan(img).any() or np.isinf(img).any():
                    # Create a safe version for visualization
                    safe_img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
                    nan_mask = np.isnan(img).squeeze()
                    inf_mask = np.isinf(img).squeeze()
                    
                    # Save visualization
                    os.makedirs("debug/nan_analysis", exist_ok=True)
                    filename = f"debug/nan_analysis/nan_image_batch{batch_count}_item{i}.png"
                    
                    # Create visualization
                    plt.figure(figsize=(15, 5))
                    plt.subplot(131)
                    plt.imshow(safe_img.squeeze(), cmap='gray')
                    plt.title(f"Image (NaNs replaced with 0)")
                    plt.colorbar()
                    plt.axis('off')
                    
                    plt.subplot(132)
                    plt.imshow(nan_mask, cmap='hot')
                    plt.title(f"NaN Mask ({np.sum(nan_mask)} pixels)")
                    plt.axis('off')
                    
                    plt.subplot(133)
                    plt.imshow(inf_mask, cmap='cool')
                    plt.title(f"Inf Mask ({np.sum(inf_mask)} pixels)")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(filename)
                    plt.close()
                    
                    print(f"Saved NaN visualization to {filename}")
                    break  # Just visualize the first problematic image
        
        # Update total element count
        total_elements += images.shape[0] * np.prod(images.shape[1:])
    
    # Calculate statistics
    nan_percentage = (nan_count / total_elements * 100) if total_elements > 0 else 0
    
    stats = {
        'nan_count': nan_count,
        'batch_count': batch_count,
        'total_elements': total_elements,
        'nan_percentage': nan_percentage,
        'nan_locations': nan_locations
    }
    
    # Print summary
    print("\nNaN Analysis Results:")
    print(f"Total batches checked: {batch_count}")
    print(f"Total NaN/Inf values found: {nan_count} out of {total_elements} elements ({nan_percentage:.6f}%)")
    print(f"Problematic batches: {len(nan_locations)} out of {batch_count}")
    
    if len(nan_locations) > 0:
        print("\nProblematic batches details:")
        for loc in nan_locations:
            print(f"  Batch {loc['batch_idx']}: {loc['nan_count']} NaNs, {loc['inf_count']} Infs")
    
    return stats

def run_dataset_nan_diagnostics(img_paths, mask_paths, slice_indices):
    """
    Run diagnostics directly on selected image/mask files to find NaN patterns.
    
    Args:
        img_paths: List of image paths
        mask_paths: List of mask paths
        slice_indices: List of slice indices to check
    """
    print(f"\nRunning NaN diagnostics on {len(img_paths)} files, {len(slice_indices)} slices each...")
    
    # Initialize counters
    total_nans = 0
    total_slices = 0
    nan_statistics = {}
    
    # Create output directory
    os.makedirs("debug/nan_diagnostics", exist_ok=True)
    
    # Analyze each file
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        print(f"\nAnalyzing file {i+1}/{len(img_paths)}: {os.path.basename(img_path)}")
        
        nan_counts = []
        problematic_slices = []
        
        try:
            # Load the full image file
            with tifffile.TiffFile(img_path) as tif:
                # Check each slice
                for slice_idx in tqdm(slice_indices):
                    if slice_idx < tif.series[0].shape[0]:
                        try:
                            img_slice = tif.asarray(key=slice_idx)
                            nan_count = np.sum(np.isnan(img_slice))
                            inf_count = np.sum(np.isinf(img_slice))
                            
                            if nan_count > 0 or inf_count > 0:
                                total_nans += nan_count + inf_count
                                nan_counts.append(nan_count + inf_count)
                                problematic_slices.append(slice_idx)
                                
                                # Save diagnostic info for the first few problematic slices
                                if len(problematic_slices) <= 5:
                                    # Create safe version for visualization
                                    safe_img = np.nan_to_num(img_slice, nan=0.0, posinf=1.0, neginf=0.0)
                                    
                                    # Normalize for visualization
                                    if safe_img.max() > safe_img.min():
                                        safe_img = (safe_img - safe_img.min()) / (safe_img.max() - safe_img.min())
                                    
                                    # Create masks
                                    nan_mask = np.isnan(img_slice)
                                    inf_mask = np.isinf(img_slice)
                                    
                                    # Create visualization
                                    plt.figure(figsize=(15, 5))
                                    plt.subplot(131)
                                    plt.imshow(safe_img, cmap='gray')
                                    plt.title(f"Image Slice {slice_idx}")
                                    plt.colorbar()
                                    plt.axis('off')
                                    
                                    plt.subplot(132)
                                    plt.imshow(nan_mask, cmap='hot')
                                    plt.title(f"NaN Locations ({nan_count} pixels)")
                                    plt.axis('off')
                                    
                                    plt.subplot(133)
                                    plt.imshow(inf_mask, cmap='cool')
                                    plt.title(f"Inf Locations ({inf_count} pixels)")
                                    plt.axis('off')
                                    
                                    plt.tight_layout()
                                    plt.savefig(f"debug/nan_diagnostics/file{i}_slice{slice_idx}_nan_analysis.png")
                                    plt.close()
                            
                            total_slices += 1
                        except Exception as slice_err:
                            print(f"  Error accessing slice {slice_idx}: {slice_err}")
        except Exception as file_err:
            print(f"  Error processing file {img_path}: {file_err}")
        
        # Store statistics for this file
        nan_statistics[os.path.basename(img_path)] = {
            'total_nan_slices': len(problematic_slices),
            'total_slices_checked': len(slice_indices),
            'percent_problematic': len(problematic_slices) / len(slice_indices) * 100 if slice_indices else 0,
            'problematic_slices': problematic_slices,
            'nan_counts': nan_counts
        }
        
        # Print summary for this file
        stats = nan_statistics[os.path.basename(img_path)]
        print(f"  Results: {stats['total_nan_slices']} problematic slices out of {stats['total_slices_checked']} ({stats['percent_problematic']:.2f}%)")
        if stats['total_nan_slices'] > 0:
            print(f"  First few problematic slices: {stats['problematic_slices'][:5]}")
    
    # Print overall summary
    print("\nOverall NaN Analysis Results:")
    print(f"Total NaN/Inf values found: {total_nans}")
    print(f"Total slices checked: {total_slices}")
    print(f"Total problematic files: {sum(1 for stats in nan_statistics.values() if stats['total_nan_slices'] > 0)} out of {len(img_paths)}")
    
    # Create summary visualization
    plt.figure(figsize=(12, 6))
    file_names = list(nan_statistics.keys())
    percentages = [stats['percent_problematic'] for stats in nan_statistics.values()]
    
    plt.bar(range(len(file_names)), percentages, color='crimson')
    plt.xlabel('File')
    plt.ylabel('Percentage of Problematic Slices')
    plt.title('NaN Distribution Across Files')
    plt.xticks(range(len(file_names)), [f"File {i}" for i in range(len(file_names))], rotation=45)
    plt.tight_layout()
    plt.savefig("debug/nan_diagnostics/overall_nan_distribution.png")
    plt.close()
    
    return nan_statistics

if __name__ == "__main__":
    print("Running standalone NaN detection...")
    
    # Find all image pairs
    from data_loader import find_data_pairs
    pairs = find_data_pairs()
    
    if pairs:
        # Extract paths for diagnostics
        img_paths = [p[0] for p in pairs]
        mask_paths = [p[1] for p in pairs]
        
        # Use a range of slice indices for each volume
        slice_indices = list(range(0, 300, 10))  # Check every 10th slice up to 300
        
        # Run diagnostics
        diagnostics_results = run_dataset_nan_diagnostics(img_paths, mask_paths, slice_indices)
        
        print("\nNaN detection complete. Check debug/nan_diagnostics/ for results.")
    else:
        print("No image pairs found. Please check your directory settings.")
