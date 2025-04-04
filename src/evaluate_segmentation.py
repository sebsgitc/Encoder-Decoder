#!/usr/bin/env python3
"""
Script to evaluate blood vessel segmentation quality
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tifffile as tiff
from glob import glob
import argparse

def create_overlay_colormap():
    """Create a custom colormap for overlaying segmentation on grayscale images"""
    # Red overlay for segmentation
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]
    return LinearSegmentedColormap.from_list('segmentation_overlay', colors)

def load_images(raw_path, seg_path):
    """Load raw image and segmentation"""
    raw_img = tiff.imread(raw_path)
    seg_img = tiff.imread(seg_path)
    
    print(f"Raw image shape: {raw_img.shape}, dtype: {raw_img.dtype}")
    print(f"Segmentation shape: {seg_img.shape}, dtype: {seg_img.dtype}")
    
    # Ensure segmentation is binary - convert to uint8 instead of bool for compatibility
    if seg_img.dtype != np.uint8:
        seg_img = (seg_img > 0).astype(np.uint8)
        
    return raw_img, seg_img

def plot_segmentation_slices(raw_img, seg_img, output_dir, slice_step=None, max_slices=10):
    """Plot original and segmented slices with overlay"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find slices with segmentation
    has_seg = np.any(seg_img, axis=(1,2))
    seg_slice_indices = np.where(has_seg)[0]
    
    if len(seg_slice_indices) == 0:
        print("No slices with segmentation found!")
        return
    
    print(f"Found {len(seg_slice_indices)} slices with segmentation")
    
    # Select subset of slices if necessary
    if slice_step is None and len(seg_slice_indices) > max_slices:
        slice_step = len(seg_slice_indices) // max_slices
        
    if slice_step and slice_step > 1:
        seg_slice_indices = seg_slice_indices[::slice_step]
        print(f"Showing every {slice_step}th slice ({len(seg_slice_indices)} slices)")
    
    # Create overlay colormap
    overlay_cmap = create_overlay_colormap()
    
    # Plot each slice
    for i, z in enumerate(seg_slice_indices):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original slice
        axes[0].imshow(raw_img[z], cmap='gray')
        axes[0].set_title(f"Original (Slice {z})")
        axes[0].axis('off')
        
        # Segmentation
        axes[1].imshow(seg_img[z], cmap='hot')
        axes[1].set_title(f"Segmentation (Slice {z})")
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(raw_img[z], cmap='gray')
        axes[2].imshow(seg_img[z], cmap=overlay_cmap)
        axes[2].set_title(f"Overlay (Slice {z})")
        axes[2].axis('off')
        
        # Add info on segmentation coverage
        coverage = np.sum(seg_img[z]) / np.prod(seg_img[z].shape) * 100
        fig.suptitle(f"Slice {z}: Segmentation Coverage = {coverage:.2f}%")
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"slice_{z:04d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        if (i+1) % 5 == 0 or i+1 == len(seg_slice_indices):
            print(f"Saved {i+1}/{len(seg_slice_indices)} slice visualizations")
    
    print(f"All visualizations saved to {output_dir}")

def calculate_metrics(seg_img):
    """Calculate quantitative metrics for segmentation"""
    # Calculate volume statistics
    total_voxels = np.prod(seg_img.shape)
    segmented_voxels = np.sum(seg_img)
    volume_percentage = segmented_voxels / total_voxels * 100
    
    # Calculate per-slice statistics
    slice_coverage = np.sum(seg_img, axis=(1,2)) / (seg_img.shape[1] * seg_img.shape[2]) * 100
    non_empty_slices = np.sum(slice_coverage > 0)
    
    # Print statistics
    print("\nSegmentation Metrics:")
    print(f"Total voxels: {total_voxels}")
    print(f"Segmented voxels: {segmented_voxels}")
    print(f"Volume percentage: {volume_percentage:.4f}%")
    print(f"Non-empty slices: {non_empty_slices}/{seg_img.shape[0]} ({non_empty_slices/seg_img.shape[0]*100:.2f}%)")
    print(f"Maximum slice coverage: {np.max(slice_coverage):.2f}%")
    print(f"Average coverage for non-empty slices: {np.mean(slice_coverage[slice_coverage > 0]):.2f}%")
    
    return {
        "total_voxels": total_voxels,
        "segmented_voxels": segmented_voxels,
        "volume_percentage": volume_percentage,
        "non_empty_slices": non_empty_slices,
        "max_slice_coverage": np.max(slice_coverage),
        "avg_slice_coverage": np.mean(slice_coverage[slice_coverage > 0])
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate blood vessel segmentation quality")
    parser.add_argument("--raw", type=str, default="3d-stacks/r01_/r01_.8bit.tif", help="Path to raw image")
    parser.add_argument("--seg", type=str, default="output/segmentation_2d_stack/r01_.8bit.tif", help="Path to segmentation")
    parser.add_argument("--output", type=str, default="output/evaluation", help="Output directory for visualizations")
    parser.add_argument("--max-slices", type=int, default=20, help="Maximum number of slices to visualize")
    
    args = parser.parse_args()
    
    print(f"Loading raw image from: {args.raw}")
    print(f"Loading segmentation from: {args.seg}")
    
    # Load images
    raw_img, seg_img = load_images(args.raw, args.seg)
    
    # Calculate metrics
    metrics = calculate_metrics(seg_img)
    
    # Plot slices
    plot_segmentation_slices(raw_img, seg_img, args.output, max_slices=args.max_slices)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
