"""Script to evaluate a model using orthogonal slices (X, Y, Z) and combine results."""

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import nibabel as nib
from skimage import io, transform

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
    
    # Threshold to get binary segmentation
    binary = (combined > 0.5).astype(np.float32)
    
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

def save_aligned_nifti(volume_data, output_path):
    """Save volume data with axis orientation matching the input."""
    # Transpose the volume to match Fiji's expected orientation
    aligned_data = np.transpose(volume_data, (2, 1, 0))
    nifti_img = nib.Nifti1Image(aligned_data, np.eye(4))
    nib.save(nifti_img, output_path)
    return output_path

def main():
    """Main function to run orthogonal evaluation."""
    # Create logger
    logger = create_logger()
    
    # Define file to evaluate
    file_name = "r07_"
    raw_file = f"{file_name}/{file_name}.8bit.tif"

    # Create full file path
    file_path = os.path.join(DATA_DIR, raw_file)
    
    # Create output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(EVALUATION_OUTPUT_DIR, f"orthogonal_{file_name}{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Load model
    logger.info("Loading model...")
    model = get_model(in_channels=1, num_classes=2)
    model.load_weights('checkpoints/model_best.weights.h5')
    
    # Load volume
    logger.info(f"Loading volume from {file_path}...")
    volume = load_volume(file_path)
    
    if volume is None:
        logger.error("Failed to load volume")
        return
    
    # Store original orientation for reference
    original_volume = volume.copy()
    
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
    
    # Save combined prediction
    combined_path = os.path.join(output_dir, f"{file_name}combined.nii.gz")
    binary_path = os.path.join(output_dir, f"{file_name}binary.nii.gz")
    
    # Save aligned NIfTI files
    logger.info("Saving aligned outputs...")
    save_aligned_nifti(combined, combined_path)
    save_aligned_nifti(binary, binary_path)
    
    # Also save original volume in same orientation for reference
    original_path = os.path.join(output_dir, f"{file_name}original.nii.gz")
    save_aligned_nifti(original_volume, original_path)
    
    logger.info(f"Saved combined prediction to {combined_path}")
    logger.info(f"Saved binary prediction to {binary_path}")
    logger.info(f"Saved original volume to {original_path}")
    
    # Create 3D visualization with aligned data
    logger.info("Creating 3D visualization...")
    vis_path = os.path.join(output_dir, f"{file_name}3d_viz.png")
    visualize_3d(volume, binary, vis_path)
    
    # Save additional visualizations of orthogonal slices with filled contours
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
    
    plt.savefig(os.path.join(output_dir, f"{file_name}orthogonal_slices.png"), dpi=300, bbox_inches='tight')
    
    # Save a text file explaining the orientation
    with open(os.path.join(output_dir, "orientation_info.txt"), "w") as f:
        f.write("Orientation Information\n")
        f.write("=====================\n\n")
        f.write("Original data dimensions: {}\n".format(original_volume.shape))
        f.write("Output data dimensions: {}\n\n".format(np.transpose(binary, (2, 1, 0)).shape))
        f.write("Axis convention used:\n")
        f.write("- Original: [Z, Y, X] (Python/NumPy convention)\n")
        f.write("- Output: [X, Y, Z] (Fiji convention)\n\n")
        f.write("The output files have been transposed to match Fiji's expected orientation.\n")
    
    logger.info("Orthogonal evaluation completed successfully")

if __name__ == "__main__":
    main()