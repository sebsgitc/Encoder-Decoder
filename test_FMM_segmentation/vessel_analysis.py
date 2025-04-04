"""Utility for analyzing vessel percentages in the dataset."""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import glob
from tqdm import tqdm
from configuration import DATA_DIR, RESULTS_DIR

def calculate_vessel_percentage(mask):
    """
    Calculate the percentage of vessel pixels in a mask.
    
    Args:
        mask: Numpy array of mask data
        
    Returns:
        Float: Percentage of vessel pixels in the mask (0-100)
    """
    total_pixels = mask.size
    vessel_pixels = np.sum(mask > 0)
    vessel_percentage = (vessel_pixels / total_pixels) * 100
    return vessel_percentage

def analyze_vessel_percentages(mask_dir=RESULTS_DIR, pattern="vessel_segmentation_*.tif"):
    """
    Analyze vessel percentages across all masks in a directory.
    
    Args:
        mask_dir: Directory containing mask files
        pattern: Pattern to match mask files
        
    Returns:
        Dictionary of statistics
    """
    mask_files = glob.glob(os.path.join(mask_dir, pattern))
    
    if not mask_files:
        print(f"No mask files found matching pattern: {pattern}")
        return {}
    
    print(f"Found {len(mask_files)} mask files")
    
    # Store percentages by volume and slice
    volume_percentages = {}
    slice_percentages = []
    
    for mask_file in tqdm(mask_files, desc="Analyzing masks"):
        volume_name = os.path.basename(mask_file).split("vessel_segmentation_")[1].split(".")[0]
        
        try:
            mask_data = tifffile.imread(mask_file)
            
            # Calculate overall volume percentage
            volume_pct = calculate_vessel_percentage(mask_data)
            volume_percentages[volume_name] = volume_pct
            
            # Calculate per-slice percentages
            for slice_idx in range(mask_data.shape[0]):
                slice_data = mask_data[slice_idx]
                slice_pct = calculate_vessel_percentage(slice_data)
                slice_percentages.append({
                    'volume': volume_name,
                    'slice': slice_idx,
                    'percentage': slice_pct
                })
                
        except Exception as e:
            print(f"Error processing {mask_file}: {str(e)}")
    
    # Calculate statistics
    all_percentages = [info['percentage'] for info in slice_percentages]
    non_zero_percentages = [p for p in all_percentages if p > 0]
    
    stats = {
        'mean_percentage': np.mean(all_percentages) if all_percentages else 0,
        'median_percentage': np.median(all_percentages) if all_percentages else 0,
        'mean_non_zero_percentage': np.mean(non_zero_percentages) if non_zero_percentages else 0,
        'max_percentage': np.max(all_percentages) if all_percentages else 0,
        'volume_percentages': volume_percentages,
        'empty_slice_count': all_percentages.count(0),
        'total_slice_count': len(all_percentages),
        'empty_slice_percentage': all_percentages.count(0) / len(all_percentages) * 100 if all_percentages else 0
    }
    
    # Print statistics
    print("\nVessel Percentage Statistics:")
    print(f"  Mean vessel percentage (all slices): {stats['mean_percentage']:.4f}%")
    print(f"  Mean vessel percentage (non-empty slices): {stats['mean_non_zero_percentage']:.4f}%")
    print(f"  Median vessel percentage: {stats['median_percentage']:.4f}%")
    print(f"  Maximum vessel percentage: {stats['max_percentage']:.4f}%")
    print(f"  Empty slices: {stats['empty_slice_count']} out of {stats['total_slice_count']} ({stats['empty_slice_percentage']:.2f}%)")
    print("\nVessel Percentages by Volume:")
    for volume, pct in volume_percentages.items():
        print(f"  {volume}: {pct:.4f}%")
    
    # Create visualizations
    visualize_vessel_statistics(all_percentages, volume_percentages)
    
    return stats

def visualize_vessel_statistics(slice_percentages, volume_percentages):
    """
    Create visualizations of vessel percentage statistics.
    
    Args:
        slice_percentages: List of percentages for all slices
        volume_percentages: Dictionary mapping volume names to percentages
    """
    os.makedirs("vessel_analysis", exist_ok=True)
    
    # Histogram of slice percentages
    plt.figure(figsize=(12, 6))
    plt.hist(slice_percentages, bins=50, alpha=0.7, color='blue')
    plt.title('Distribution of Vessel Percentages Across Slices')
    plt.xlabel('Vessel Percentage (%)')
    plt.ylabel('Number of Slices')
    plt.grid(alpha=0.3)
    plt.savefig("vessel_analysis/vessel_percentage_histogram.png")
    plt.close()
    
    # Bar chart of volume percentages
    plt.figure(figsize=(14, 6))
    volumes = list(volume_percentages.keys())
    percentages = list(volume_percentages.values())
    plt.bar(volumes, percentages, color='teal')
    plt.title('Vessel Percentages by Volume')
    plt.xlabel('Volume')
    plt.ylabel('Vessel Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("vessel_analysis/volume_percentages.png")
    plt.close()
    
    # Calculate the recommended target percentage (2x mean)
    mean_percentage = np.mean(slice_percentages) if slice_percentages else 0
    target_percentage = mean_percentage * 2.0
    
    # Non-zero slices histogram with reference line
    non_zero_percentages = [p for p in slice_percentages if p > 0]
    plt.figure(figsize=(12, 6))
    plt.hist(non_zero_percentages, bins=50, alpha=0.7, color='green')
    plt.axvline(x=target_percentage, color='r', linestyle='--', label=f'Target (2x Mean): {target_percentage:.4f}%')
    plt.title('Distribution of Vessel Percentages (Non-Empty Slices)')
    plt.xlabel('Vessel Percentage (%)')
    plt.ylabel('Number of Slices')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("vessel_analysis/non_empty_vessel_percentage_histogram.png")
    plt.close()
    
    print(f"\nVisualization saved to vessel_analysis/ directory")
    print(f"Recommended target vessel percentage (2x mean): {target_percentage:.4f}%")

if __name__ == "__main__":
    analyze_vessel_percentages()
