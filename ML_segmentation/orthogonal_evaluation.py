"""Script to evaluate a model using orthogonal slices (X, Y, Z) and combine results."""

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import nibabel as nib
from skimage import io, transform
import argparse
from datetime import datetime

from model import get_model
from configuration import *
from utils import create_logger
import matplotlib.pyplot as plt
from scipy import ndimage

# Function to load a 3D volume
def load_volume(file_path):
    """Load a 3D volume from file."""
    try:
        # Try loading as TIFF stack
        volume = io.imread(file_path)
        print(f"Loaded volume with shape: {volume.shape}")
        return volume
    except Exception as e:
        print(f"Error loading volume: {str(e)}")
        return None

# Function to preprocess a volume slice for model input
def preprocess_slice(slice_img, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Preprocess a slice for model input."""
    # Normalize to [0, 1]
    slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-8)
    
    # Resize if necessary
    if slice_img.shape[:2] != target_size:
        slice_img = transform.resize(slice_img, target_size, preserve_range=True)
    
    # Add batch and channel dimensions
    slice_tensor = np.expand_dims(np.expand_dims(slice_img, axis=0), axis=-1)
    
    return slice_tensor.astype(np.float32)

# Function to predict on a volume along a specific axis
def predict_along_axis(model, volume, axis=0):
    """Predict segmentation along a specific axis."""
    # Get dimensions
    dims = volume.shape
    
    # Initialize output volume
    if axis == 0:  # Z-axis (default)
        output = np.zeros(dims, dtype=np.float32)
        n_slices = dims[0]
    elif axis == 1:  # Y-axis
        output = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)
        n_slices = dims[1]
    elif axis == 2:  # X-axis
        output = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)
        n_slices = dims[2]
    
    # Predict slice by slice
    for i in tqdm(range(n_slices), desc=f"Predicting along axis {axis}"):
        # Extract slice
        if axis == 0:  # Z-axis
            slice_img = volume[i, :, :]
        elif axis == 1:  # Y-axis
            slice_img = volume[:, i, :]
        elif axis == 2:  # X-axis
            slice_img = volume[:, :, i]
        
        # Preprocess slice
        slice_tensor = preprocess_slice(slice_img)
        
        # Predict
        pred = model.predict(slice_tensor, verbose=0)
        pred_resized = transform.resize(pred[0, :, :, 0], slice_img.shape, preserve_range=True)
        
        # Store prediction
        if axis == 0:  # Z-axis
            output[i, :, :] = pred_resized
        elif axis == 1:  # Y-axis
            output[:, i, :] = pred_resized
        elif axis == 2:  # X-axis
            output[:, :, i] = pred_resized
    
    return output

# Function to combine predictions from different axes
def combine_predictions(pred_z, pred_y, pred_x, method='average'):
    """Combine predictions from different axes."""
    if method == 'average':
        # Simple averaging
        combined = (pred_z + pred_y + pred_x) / 3.0
    elif method == 'max':
        # Maximum value
        combined = np.maximum(np.maximum(pred_z, pred_y), pred_x)
    elif method == 'weighted':
        # Weighted average (customize weights as needed)
        combined = 0.4 * pred_z + 0.3 * pred_y + 0.3 * pred_x
    
    # Threshold to get binary segmentation with values 0 and 1
    # (we'll scale to 0-255 in the save function)
    binary = (combined > 0.5).astype(np.uint8)
    
    return combined, binary

# Function to visualize and save 3D rendering
def visualize_3d(volume, segmentation, output_path):
    """Create and save 3D visualization of segmentation."""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get coordinates of vessel voxels
        z, y, x = np.where(segmentation > 0.5)
        
        # Downsample if too many points
        max_points = 10000
        if len(z) > max_points:
            idx = np.random.choice(len(z), max_points, replace=False)
            z, y, x = z[idx], y[idx], x[idx]
        
        # Original image intensities for coloring
        c = [volume[z[i], y[i], x[i]] for i in range(len(z))]
        
        # Plot scatter with original intensities as colormap
        scatter = ax.scatter(x, y, z, c=c, alpha=0.7, s=2, cmap='viridis')
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Vessel Segmentation')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Intensity')
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"3D visualization saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error creating 3D visualization: {str(e)}")
        return False

def save_as_tiff_8bit(volume_data, output_path):
    """Save volume data as an 8-bit TIFF stack."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to 8-bit
        data_min = volume_data.min()
        data_max = volume_data.max()
        
        # Scale data to 0-255 range
        if data_max > data_min:  # Avoid division by zero
            scaled_data = ((volume_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            scaled_data = np.zeros_like(volume_data, dtype=np.uint8)
            
        # Save as TIFF
        io.imsave(output_path, scaled_data)
        return output_path
    except Exception as e:
        print(f"Error saving 8-bit TIFF file: {str(e)}")
        return None

def save_binary_nifti(volume_data, output_path):
    """Save binary segmentation as compressed NIfTI file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert binary data to 0 and 255 (instead of 0 and 1)
        binary_data = (volume_data > 0).astype(np.uint8) * 255
        
        # Log statistics for debugging
        unique_values = np.unique(binary_data)
        print(f"Binary data values: {unique_values}, dtype: {binary_data.dtype}")
        
        # Transpose the volume to match Fiji's expected orientation
        aligned_data = np.transpose(binary_data, (2, 1, 0))
        
        # Create NIfTI image with compression
        nifti_img = nib.Nifti1Image(aligned_data, np.eye(4))
        nifti_img.header.set_data_dtype(np.uint8)
        
        # Save with compression
        nib.save(nifti_img, output_path)
        print(f"NIfTI file saved with values {np.unique(aligned_data)}")
        return output_path
    except Exception as e:
        print(f"Error saving binary NIfTI file: {str(e)}")
        return None

def create_output_structure(base_dir, timestamp, single_file=None):
    """Create structured output directories."""
    if single_file:
        # For single file evaluation
        output_dir = os.path.join(base_dir, f"orthogonal_{single_file}{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    else:
        # For all files evaluation
        main_output_dir = os.path.join(base_dir, f"orthogonal_{timestamp}")
        os.makedirs(main_output_dir, exist_ok=True)
        
        # Create subdirectories
        binary_dir = os.path.join(main_output_dir, f"orthogonal_binary_{timestamp}")
        combined_dir = os.path.join(main_output_dir, f"orthogonal_combined_{timestamp}")
        original_dir = os.path.join(main_output_dir, f"orthogonal_original_{timestamp}")
        slices_dir = os.path.join(main_output_dir, f"orthogonal_slices_{timestamp}")
        
        # Create all directories
        for directory in [binary_dir, combined_dir, original_dir, slices_dir]:
            os.makedirs(directory, exist_ok=True)
        
        return {
            'main': main_output_dir,
            'binary': binary_dir,
            'combined': combined_dir,
            'original': original_dir,
            'slices': slices_dir
        }

def process_single_volume(file_name, model, output_dir, logger):
    """Process a single volume file."""
    # Create full file path
    raw_file = f"{file_name}/{file_name}.8bit.tif"
    file_path = os.path.join(DATA_DIR, raw_file)
    
    logger.info(f"Processing file: {file_name}")
    
    # Load volume
    logger.info(f"Loading volume from {file_path}...")
    volume = load_volume(file_path)
    
    if volume is None:
        logger.error(f"Failed to load volume: {file_path}")
        return False
    
    # Predict along each axis
    logger.info("Predicting along Z-axis...")
    pred_z = predict_along_axis(model, volume, axis=0)
    
    logger.info("Predicting along Y-axis...")
    pred_y = predict_along_axis(model, volume, axis=1)
    
    logger.info("Predicting along X-axis...")
    pred_x = predict_along_axis(model, volume, axis=2)
    
    # Combine predictions
    logger.info("Combining predictions...")
    combined, binary = combine_predictions(pred_z, pred_y, pred_x, method='weighted')
    
    # Save combined prediction as 8-bit TIFF
    combined_path = os.path.join(output_dir, f"{file_name}combined.tif")
    save_as_tiff_8bit(combined, combined_path)
    logger.info(f"Saved combined prediction to {combined_path}")
    
    # Save binary prediction as compressed NIfTI
    binary_path = os.path.join(output_dir, f"{file_name}binary.nii.gz")
    save_binary_nifti(binary, binary_path)
    logger.info(f"Saved binary prediction to {binary_path}")
    
    # Create 3D visualization
    logger.info("Creating 3D visualization...")
    vis_path = os.path.join(output_dir, f"{file_name}3d_viz.png")
    visualize_3d(volume, binary, vis_path)
    
    # Save orthogonal slice visualizations
    logger.info("Saving orthogonal slice visualizations...")
    mid_z = volume.shape[0] // 2
    mid_y = volume.shape[1] // 2
    mid_x = volume.shape[2] // 2
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(volume[mid_z, :, :], cmap='gray')
    plt.contourf(binary[mid_z, :, :], levels=[0.5, 1.0], colors=['r'], alpha=0.3)
    plt.title(f'Z-slice {mid_z}')
    
    plt.subplot(132)
    plt.imshow(volume[:, mid_y, :], cmap='gray')
    plt.contourf(binary[:, mid_y, :], levels=[0.5, 1.0], colors=['r'], alpha=0.3)
    plt.title(f'Y-slice {mid_y}')
    
    plt.subplot(133)
    plt.imshow(volume[:, :, mid_x], cmap='gray')
    plt.contourf(binary[:, :, mid_x], levels=[0.5, 1.0], colors=['r'], alpha=0.3)
    plt.title(f'X-slice {mid_x}')
    
    slices_path = os.path.join(output_dir, f"{file_name}orthogonal_slices.png")
    plt.savefig(slices_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved orthogonal slices to {slices_path}")
    
    # Save orientation information
    with open(os.path.join(output_dir, "orientation_info.txt"), "w") as f:
        f.write("Orientation Information\n")
        f.write("=====================\n\n")
        f.write("Output data dimensions: {}\n\n".format(binary.shape))
        f.write("Volume saved as TIFF stack\n")
    
    return True

def process_all_volumes(model, output_dirs, logger):
    """Process all volume files in the data directory."""
    # Get all subdirectories in data_dir that end with '_'
    all_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.endswith('_')]
    
    successful = []
    failed = []
    
    for folder in all_folders:
        file_name = folder  # The folder name is also the file name prefix
        
        try:
            # Create full file path
            raw_file = f"{file_name}/{file_name}.8bit.tif"
            file_path = os.path.join(DATA_DIR, raw_file)
            
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}, skipping...")
                failed.append(file_name)
                continue
            
            logger.info(f"Processing file: {file_name}")
            
            # Load volume
            logger.info(f"Loading volume from {file_path}...")
            volume = load_volume(file_path)
            
            if volume is None:
                logger.error(f"Failed to load volume: {file_path}")
                failed.append(file_name)
                continue
            
            # Predict along each axis
            logger.info("Predicting along Z-axis...")
            pred_z = predict_along_axis(model, volume, axis=0)
            
            logger.info("Predicting along Y-axis...")
            pred_y = predict_along_axis(model, volume, axis=1)
            
            logger.info("Predicting along X-axis...")
            pred_x = predict_along_axis(model, volume, axis=2)
            
            # Combine predictions
            logger.info("Combining predictions...")
            combined, binary = combine_predictions(pred_z, pred_y, pred_x, method='weighted')
            
            # Save binary prediction as compressed NIfTI
            binary_path = os.path.join(output_dirs['binary'], f"{file_name}binary.nii.gz")
            save_binary_nifti(binary, binary_path)
            logger.info(f"Saved binary prediction to {binary_path}")
            
            # Save combined prediction as 8-bit TIFF
            combined_path = os.path.join(output_dirs['combined'], f"{file_name}combined.tif")
            save_as_tiff_8bit(combined, combined_path)
            logger.info(f"Saved combined prediction to {combined_path}")
            
            # Create 3D visualization
            vis_path = os.path.join(output_dirs['main'], f"{file_name}3d_viz.png")
            visualize_3d(volume, binary, vis_path)
            
            # Save orthogonal slice visualizations
            mid_z = volume.shape[0] // 2
            mid_y = volume.shape[1] // 2
            mid_x = volume.shape[2] // 2
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(volume[mid_z, :, :], cmap='gray')
            plt.contourf(binary[mid_z, :, :], levels=[0.5, 1.0], colors=['r'], alpha=0.3)
            plt.title(f'Z-slice {mid_z}')
            
            plt.subplot(132)
            plt.imshow(volume[:, mid_y, :], cmap='gray')
            plt.contourf(binary[:, mid_y, :], levels=[0.5, 1.0], colors=['r'], alpha=0.3)
            plt.title(f'Y-slice {mid_y}')
            
            plt.subplot(133)
            plt.imshow(volume[:, :, mid_x], cmap='gray')
            plt.contourf(binary[:, :, mid_x], levels=[0.5, 1.0], colors=['r'], alpha=0.3)
            plt.title(f'X-slice {mid_x}')
            
            slices_path = os.path.join(output_dirs['slices'], f"{file_name}orthogonal_slices.png")
            plt.savefig(slices_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved orthogonal slices to {slices_path}")
            
            successful.append(file_name)
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
            failed.append(file_name)
    
    # Save summary
    summary_path = os.path.join(output_dirs['main'], "processing_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Orthogonal Evaluation Summary\n")
        f.write("===========================\n\n")
        f.write(f"Total files processed: {len(successful) + len(failed)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        f.write("Successful files:\n")
        for file_name in successful:
            f.write(f"- {file_name}\n")
        
        f.write("\nFailed files:\n")
        for file_name in failed:
            f.write(f"- {file_name}\n")
    
    logger.info(f"Processing summary saved to {summary_path}")
    logger.info(f"Successfully processed {len(successful)} files, failed to process {len(failed)} files")
    
    return (successful, failed)

def main():
    """Main function to run orthogonal evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model using orthogonal slices')
    parser.add_argument('--file', type=str, help='Specific file name to process (e.g., "r01_")')
    parser.add_argument('--all', action='store_true', help='Process all available files')
    parser.add_argument('--model', type=str, default='checkpoints/model_best.weights.h5', help='Path to model weights')
    args = parser.parse_args()
    
    # Check if at least one option is provided
    if not args.file and not args.all:
        print("ERROR: Please specify either --file or --all")
        parser.print_help()
        sys.exit(1)
    
    # Create logger
    logger = create_logger()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model
    logger.info("Loading model...")
    model = get_model(in_channels=1, num_classes=2)
    model.load_weights(args.model)
    
    if args.file:
        # Process single file
        file_name = args.file
        output_dir = create_output_structure(EVALUATION_OUTPUT_DIR, timestamp, single_file=file_name)
        logger.info(f"Results will be saved to: {output_dir}")
        
        success = process_single_volume(file_name, model, output_dir, logger)
        if success:
            logger.info(f"Orthogonal evaluation completed successfully for {file_name}")
        else:
            logger.error(f"Orthogonal evaluation failed for {file_name}")
    
    elif args.all:
        # Process all files
        output_dirs = create_output_structure(EVALUATION_OUTPUT_DIR, timestamp)
        logger.info(f"Results will be saved to: {output_dirs['main']}")
        
        successful, failed = process_all_volumes(model, output_dirs, logger)
        logger.info("Orthogonal evaluation of all volumes completed")

if __name__ == "__main__":
    main()