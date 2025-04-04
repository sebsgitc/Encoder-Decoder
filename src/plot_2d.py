import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from glob import glob
from matplotlib.widgets import Slider, Button

def plot_2d_segmentation_comparison(raw_pattern="r01_*.tif", output_dir="output/visualisation", save_interval=10):
    """
    Saves comparison visualizations as PNG files instead of interactive display.
    save_interval: Save every Nth slice to avoid generating too many files
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")
    
    # Calculate total number of slices and estimated file size
    total_slices = raw_volume.shape[0]
    slices_to_save = total_slices // save_interval
    estimated_size_mb = (slices_to_save * 15 * 6 * 3) / (1024 * 1024)  # Rough estimate: 15x6 inches, 3 plots, ~100KB per plot
    print(f"Will save {slices_to_save} slices (every {save_interval}th slice)")
    print(f"Estimated total size: {estimated_size_mb:.1f} MB")
    
    # Create figures for each Nth slice
    for depth in range(0, total_slices, save_interval):
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
        
        # Add slice number to the plot
        fig.suptitle(f'Slice {depth}/{total_slices}')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(output_dir, f'comparison_slice_{depth:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Important: close figure to free memory
        
        # Print progress
        if depth % 50 == 0:
            print(f"Saved slice {depth}/{total_slices}")
    
    print("\nAll visualizations saved successfully!")
    print(f"Files are located in: {os.path.abspath(output_dir)}")

    # Create a compressed archive of all images
    import shutil
    archive_path = os.path.join(os.path.dirname(output_dir), 'visualizations.tar.gz')
    shutil.make_archive(
        os.path.splitext(archive_path)[0],
        'gztar',
        output_dir
    )
    print(f"\nCreated compressed archive: {archive_path}")
    archive_size = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"Archive size: {archive_size:.1f} MB")

if __name__ == "__main__":
    try:
        plot_2d_segmentation_comparison()
    except Exception as e:
        print(f"Error: {str(e)}")