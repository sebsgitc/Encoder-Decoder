"""Script to create slice visualizations for segmentation results in the results folder."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import argparse
from datetime import datetime
import glob
import nibabel as nib
from tqdm import tqdm

# Define the sample datasets to use
SAMPLE_DATASETS = [
    {
        "name": "r01_", 
        "raw_file": "r01_/r01_.8bit.tif",
        "mask_file": "vessel_segmentation_r01_*.tif"  # Using wildcard to find the latest
    },
    {
        "name": "rL1_", 
        "raw_file": "rL1_/rL1_.8bit.tif",
        "mask_file": "vessel_segmentation_rL1_*.tif"
    },
    {
        "name": "r33_", 
        "raw_file": "r33_/r33_.8bit.tif",
        "mask_file": "vessel_segmentation_r33_*.tif"
    },
    {
        "name": "r34_", 
        "raw_file": "r34_/r34_.8bit.tif",
        "mask_file": "vessel_segmentation_r34_*.tif"
    },
    {
        "name": "rL4_", 
        "raw_file": "rL4_/rL4_.8bit.tif",
        "mask_file": "vessel_segmentation_rL4_*.tif"
    }
]

def create_logger():
    """Create a simple logger that prints to console."""
    import logging
    logger = logging.getLogger("results_visualizer")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    
    return logger

def load_volume(file_path):
    """Load a 3D volume from file."""
    try:
        # Check file extension
        if file_path.endswith(('.nii', '.nii.gz')):
            # Load as NIfTI
            nifti_img = nib.load(file_path)
            volume = nifti_img.get_fdata()
            print(f"Loaded NIfTI volume with shape: {volume.shape}")
            return volume
        else:
            # Try loading as TIFF stack
            volume = io.imread(file_path)
            print(f"Loaded TIFF volume with shape: {volume.shape}")
            return volume
    except Exception as e:
        print(f"Error loading volume: {str(e)}")
        return None

def create_slice_visualization(raw_vol, mask_vol, output_path):
    """Create and save visualization of middle slices (512)."""
    # Fixed middle slice at 512
    mid_slice = 512
    
    # Create figure with three subplots for orthogonal views
    plt.figure(figsize=(15, 5))
    
    # Z-axis slice (axial)
    plt.subplot(131)
    # Make sure we use the 512 index, but handle if volume is smaller
    z_slice = min(mid_slice, raw_vol.shape[0]-1) if raw_vol.shape[0] > mid_slice else raw_vol.shape[0]//2
    plt.imshow(raw_vol[z_slice, :, :], cmap='gray')
    
    # For binary data that's 0/255, we need different contour levels
    if np.max(mask_vol) > 1:
        plt.contourf(mask_vol[z_slice, :, :], levels=[128, 256], colors=['r'], alpha=0.3)
    else:
        plt.contourf(mask_vol[z_slice, :, :], levels=[0.5, 1.0], colors=['r'], alpha=0.3)
    plt.title(f'Z-slice {z_slice}')
    
    # Y-axis slice (coronal)
    plt.subplot(132)
    y_slice = min(mid_slice, raw_vol.shape[1]-1) if raw_vol.shape[1] > mid_slice else raw_vol.shape[1]//2
    plt.imshow(raw_vol[:, y_slice, :], cmap='gray')
    
    if np.max(mask_vol) > 1:
        plt.contourf(mask_vol[:, y_slice, :], levels=[128, 256], colors=['r'], alpha=0.3)
    else:
        plt.contourf(mask_vol[:, y_slice, :], levels=[0.5, 1.0], colors=['r'], alpha=0.3)
    plt.title(f'Y-slice {y_slice}')
    
    # X-axis slice (sagittal)
    plt.subplot(133)
    x_slice = min(mid_slice, raw_vol.shape[2]-1) if raw_vol.shape[2] > mid_slice else raw_vol.shape[2]//2
    plt.imshow(raw_vol[:, :, x_slice], cmap='gray')
    
    if np.max(mask_vol) > 1:
        plt.contourf(mask_vol[:, :, x_slice], levels=[128, 256], colors=['r'], alpha=0.3)
    else:
        plt.contourf(mask_vol[:, :, x_slice], levels=[0.5, 1.0], colors=['r'], alpha=0.3)
    plt.title(f'X-slice {x_slice}')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def find_latest_mask(mask_pattern, results_dir):
    """Find the latest mask file matching the pattern."""
    # Get full path with wildcard
    full_pattern = os.path.join(results_dir, mask_pattern)
    
    # Find all matching files
    matching_files = glob.glob(full_pattern)
    
    if not matching_files:
        return None
    
    # Sort by modification time (newest first)
    latest_file = sorted(matching_files, key=os.path.getmtime, reverse=True)[0]
    return latest_file

def visualize_specific_datasets(output_dir):
    """Process only the specific datasets in SAMPLE_DATASETS."""
    logger = create_logger()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Base directories
    raw_dir = os.path.join('/home2/ad4631sv-s/TestSebbe/Encoder-Decoder', '3d-stacks')
    results_dir = os.path.join('/home2/ad4631sv-s/TestSebbe/Encoder-Decoder', 'results')
    
    # Track successful and failed processing
    successful = []
    failed = []
    
    # Process each dataset
    for dataset in tqdm(SAMPLE_DATASETS, desc="Processing datasets"):
        try:
            name = dataset["name"]
            logger.info(f"Processing {name}...")
            
            # Get raw file path
            raw_file = os.path.join(raw_dir, dataset['raw_file'])
            
            # Find the latest mask file
            mask_file = find_latest_mask(dataset['mask_file'], results_dir)
            
            if not os.path.exists(raw_file) or mask_file is None:
                logger.error(f"Files not found for {name}")
                if not os.path.exists(raw_file):
                    logger.error(f"  Raw file not found: {raw_file}")
                if mask_file is None:
                    logger.error(f"  Mask file not found: {dataset['mask_file']}")
                failed.append(name)
                continue
            
            # Load volumes
            logger.info(f"  Loading raw file: {raw_file}")
            raw_vol = load_volume(raw_file)
            
            logger.info(f"  Loading mask file: {mask_file}")
            mask_vol = load_volume(mask_file)
            
            if raw_vol is None or mask_vol is None:
                logger.error(f"Failed to load volumes for {name}")
                failed.append(name)
                continue
            
            # Check if dimensions match
            if raw_vol.shape != mask_vol.shape:
                logger.warning(f"Dimension mismatch for {name}: raw {raw_vol.shape}, mask {mask_vol.shape}")
            
            # Create output path
            vis_path = os.path.join(output_dir, f"{name}_middle_slices.png")
            
            # Create and save visualization
            create_slice_visualization(raw_vol, mask_vol, vis_path)
            logger.info(f"Saved visualization to {vis_path}")
            
            successful.append(name)
            
        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            failed.append(name)
    
    # Generate summary
    summary_path = os.path.join(output_dir, "visualization_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Visualization Summary\n")
        f.write("=====================\n\n")
        f.write(f"Total datasets processed: {len(SAMPLE_DATASETS)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        f.write("Successful datasets:\n")
        for name in successful:
            f.write(f"- {name}\n")
        
        f.write("\nFailed datasets:\n")
        for name in failed:
            f.write(f"- {name}\n")
    
    logger.info(f"Visualization complete. Summary saved to {summary_path}")
    
    return successful, failed

def main():
    """Main function to run visualization."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create slice visualizations for specific datasets')
    parser.add_argument('--output', type=str, default='',
                       help='Output directory for visualizations (default: generated based on timestamp)')
    args = parser.parse_args()
    
    # Create output directory with timestamp if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('/home2/ad4631sv-s/TestSebbe/Encoder-Decoder', f"middle_slice_visualizations_{timestamp}")
    else:
        output_dir = os.path.join('/home2/ad4631sv-s/TestSebbe/Encoder-Decoder', args.output)
    
    # Run visualization for specific datasets
    visualize_specific_datasets(output_dir)

if __name__ == "__main__":
    main()
