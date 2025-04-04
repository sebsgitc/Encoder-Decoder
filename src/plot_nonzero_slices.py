import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from glob import glob

def plot_nonzero_segmentation(raw_pattern="r01_*.tif", output_dir="output/visualisation_nonzero"):
    """
    Saves visualizations of slices that have non-zero segmentation values.
    If more than 100 such slices exist, plots every 10th of those slices.
    """
    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "3d-stacks", "r01_")
    seg_dir = os.path.join(base_dir, "output", "segmentation_2d_stack")
    output_dir = os.path.join(base_dir, output_dir)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    
    # Load image stacks
    raw_files = sorted(glob(os.path.join(raw_dir, raw_pattern)))
    seg_files = sorted(glob(os.path.join(seg_dir, raw_pattern)))
    
    if not raw_files or not seg_files:
        raise ValueError(f"No image files found.\nRaw dir: {raw_dir}\nSeg dir: {seg_dir}")
    
    print(f"Loading {len(raw_files)} image stacks...")
    
    # Load 3D stacks
    raw_volume = tiff.imread(raw_files[0])
    seg_volume = tiff.imread(seg_files[0])
    
    print(f"Raw volume shape: {raw_volume.shape}")
    print(f"Segmentation volume shape: {seg_volume.shape}")
    
    # Find slices with non-zero segmentation
    nonzero_slices = []
    for i in range(seg_volume.shape[0]):
        if np.sum(seg_volume[i]) > 0:
            nonzero_slices.append(i)
    
    print(f"\nFound {len(nonzero_slices)} slices with non-zero segmentation")
    
    # Determine which slices to plot
    if len(nonzero_slices) > 100:
        print("More than 100 non-zero slices, plotting every 10th slice")
        plot_indices = nonzero_slices[::10]
    else:
        plot_indices = nonzero_slices
    
    print(f"Will plot {len(plot_indices)} slices")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")
    
    # Create figures for selected slices
    for depth in plot_indices:
        fig = plt.figure(figsize=(15, 6))
        grid = plt.GridSpec(1, 3)
        
        # Create axes for images
        ax1 = fig.add_subplot(grid[0])
        ax2 = fig.add_subplot(grid[1])
        ax3 = fig.add_subplot(grid[2])
        
        # Display slices
        ax1.imshow(raw_volume[depth], cmap='gray')
        ax2.imshow(seg_volume[depth], cmap='gray')
        ax3.imshow(raw_volume[depth], cmap='gray', alpha=0.5)
        masked = np.ma.masked_where(seg_volume[depth] == 0, raw_volume[depth])
        ax3.imshow(masked, cmap='hot', alpha=0.5)
        
        # Configure axes
        ax1.set_title('Original Volume')
        ax2.set_title('Segmentation Mask')
        ax3.set_title('Overlay')
        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add slice information to the plot
        pixel_sum = np.sum(seg_volume[depth])
        fig.suptitle(f'Slice {depth}/{raw_volume.shape[0]} (Number of pixels: {pixel_sum / 255})')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(output_dir, f'nonzero_slice_{depth:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print progress every 10 slices
        if len(plot_indices) >= 10 and plot_indices.index(depth) % 10 == 0:
            print(f"Saved slice {plot_indices.index(depth) + 1}/{len(plot_indices)}")
    
    print("\nAll visualizations saved successfully!")
    print(f"Files are located in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    try:
        plot_nonzero_segmentation()
    except Exception as e:
        print(f"Error: {str(e)}")