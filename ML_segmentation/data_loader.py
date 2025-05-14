"""Data loading and preprocessing utilities for binary vessel segmentation using TensorFlow."""

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
import tifffile
from configuration import *

DEBUG_DATA_LOADING = False

def find_data_pairs():
    """Find pairs of raw images and their corresponding segmentation masks."""
    pairs = []
    
    # First try to use the sample datasets defined in configuration which include all specified pairs
    for dataset in SAMPLE_DATASETS:
        raw_file = os.path.join(DATA_DIR, dataset["raw_file"])
        
        # The mask file might contain a wildcard, so we need to find matching files
        mask_pattern = os.path.join(RESULTS_DIR, dataset["mask_file"])
        mask_files = glob.glob(mask_pattern)
        
        if os.path.exists(raw_file) and mask_files:
            # Use the most recent mask file if multiple exist
            mask_file = sorted(mask_files)[-1]
            pairs.append((raw_file, mask_file))
            if DEBUG_DATA_LOADING:
                print(f"Using dataset '{dataset['name']}':\n  - Raw: {raw_file}\n  - Mask: {mask_file}")
    
    # If we didn't find all expected pairs, also search through all subdirectories
    if len(pairs) < len(SAMPLE_DATASETS):
        if DEBUG_DATA_LOADING:
            print(f"Only found {len(pairs)} of {len(SAMPLE_DATASETS)} expected pairs. Searching directories...")
        
        # Walk through all subdirectories in the DATA_DIR
        for subdir in os.listdir(DATA_DIR):
            # Check if this is one of our target directories
            if subdir in [dataset["name"] for dataset in SAMPLE_DATASETS] and subdir not in [os.path.dirname(pair[0]).split('/')[-1] for pair in pairs]:
                subdir_path = os.path.join(DATA_DIR, subdir)
                if os.path.isdir(subdir_path):
                    # Find all .tif files in this subdirectory
                    tif_files = glob.glob(os.path.join(subdir_path, "*.tif"))
                    for tif_file in tif_files:
                        # Skip files with "mask" in the name as they're not raw data
                        if "mask" in os.path.basename(tif_file):
                            continue
                        
                        # Check if a matching segmentation exists in the results directory
                        base_name = os.path.basename(subdir)
                        mask_patterns = [
                            os.path.join(RESULTS_DIR, f"{base_name}.tif"),  # Basic pattern
                            os.path.join(RESULTS_DIR, f"vessel_segmentation_{base_name}*.tif")  # Pattern with prefix and date
                        ]
                        
                        # Try different mask patterns
                        for pattern in mask_patterns:
                            mask_files = glob.glob(pattern)
                            if mask_files:
                                # Use the most recent mask file if multiple exist
                                mask_file = sorted(mask_files)[-1]
                                pair = (tif_file, mask_file)
                                if pair not in pairs:  # Avoid duplicates
                                    pairs.append(pair)
                                    if DEBUG_DATA_LOADING:
                                        print(f"Found additional pair: \n  - Raw: {tif_file}\n  - Mask: {mask_file}")
                                break
    
    if DEBUG_DATA_LOADING:
        print(f"Total image-mask pairs found: {len(pairs)}")
    return pairs

def normalize_image(image):
    """Normalize image to [0, 1] range with NaN protection."""
    image = tf.cast(image, tf.float32)
    
    # Check for NaN and replace with zeros
    image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
    
    # Safe min-max calculation with epsilon to prevent division by zero
    min_val = tf.reduce_min(image)
    max_val = tf.reduce_max(image)
    epsilon = 1e-8
    
    normalized = (image - min_val) / tf.maximum(max_val - min_val, epsilon)
    
    # Ensure values are in valid range even if normalization failed
    return tf.clip_by_value(normalized, 0.0, 1.0)

def process_image_mask(image_path, mask_path, slice_idx=None):
    """Load and preprocess a single image and mask pair for binary classification."""
    # Load image and mask
    try:
        # Convert tensor to string if needed
        if isinstance(image_path, tf.Tensor):
            image_path = image_path.numpy().decode('utf-8')
        if isinstance(mask_path, tf.Tensor):
            mask_path = mask_path.numpy().decode('utf-8')
            
        image = tifffile.imread(image_path)
        mask = tifffile.imread(mask_path)
        
        if DEBUG_DATA_LOADING:
            print(f"Loaded image shape: {image.shape}, mask shape: {mask.shape}")
        
        if slice_idx is not None:
            # For 2D mode: extract specific slice
            slice_idx = int(slice_idx)
            if slice_idx < image.shape[0]:
                image = image[slice_idx]
            else:
                if DEBUG_DATA_LOADING:
                    print(f"Warning: slice_idx {slice_idx} out of bounds for image with {image.shape[0]} slices")
                image = np.zeros((1024, 1024), dtype=np.float32)
                
            if slice_idx < mask.shape[0]:
                mask = mask[slice_idx]
            else:
                if DEBUG_DATA_LOADING:
                    print(f"Warning: slice_idx {slice_idx} out of bounds for mask with {mask.shape[0]} slices")
                mask = np.zeros((1024, 1024), dtype=np.int32)
        
        # Normalize image to [0, 1]
        image = tf.cast(image, tf.float32)
        image = normalize_image(image)
        
        # Process mask into binary segmentation
        # 0: Background/other, 1: Blood vessels (segmented vessels are typically non-zero)
        segmentation = np.zeros_like(mask, dtype=np.int32)
        segmentation[mask > 0] = 1  # Any positive value in the mask indicates a vessel
        
        # Add channel dimension if 2D image
        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)
        
        # Resize if needed
        if IMAGE_SIZE != 1024:
            # Resize image using bilinear interpolation
            image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            
            # Resize mask using nearest neighbor to preserve segmentation labels
            segmentation = tf.expand_dims(segmentation, axis=-1)
            segmentation = tf.image.resize(segmentation, (IMAGE_SIZE, IMAGE_SIZE), 
                                          method='nearest')
            segmentation = tf.squeeze(segmentation, axis=-1)
        
        # Convert mask to int32
        segmentation = tf.cast(segmentation, tf.int32)
        
        # Ensure image has shape [height, width, channels]
        if len(image.shape) < 3:
            image = tf.expand_dims(image, axis=-1)
            
        return image, segmentation
        
    except Exception as e:
        if DEBUG_DATA_LOADING:
            print(f"Error loading image {image_path} or mask {mask_path}: {str(e)}")
        # Return empty placeholders in case of error
        return tf.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=tf.float32), tf.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)

def data_augmentation(image, mask):
    """Apply data augmentation to image and mask with enhanced background diversity."""
    # First, ensure that our inputs have the expected shape
    # The error indicates we're missing a dimension for flip_left_right,
    # which requires at least 3 dimensions
    
    # Check if there's a channel dimension already
    if len(tf.shape(image)) < 3:
        # Add channel dimension if missing
        image = tf.expand_dims(image, axis=-1)
    
    if len(tf.shape(mask)) < 3:
        # For mask, ensure it's also 3D for consistent augmentation
        mask = tf.expand_dims(mask, axis=-1)
    
    # Random flip left-right
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Random flip up-down
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    
    # Random brightness (image only)
    image = tf.image.random_brightness(image, 0.1)
    
    # Random contrast adjustment to help distinguish subtle background features
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random gamma correction to help with background variations
    if tf.random.uniform(()) > 0.5:
        gamma = tf.random.uniform([], 0.8, 1.2)
        image = tf.pow(image, gamma)
    
    # Random noise to simulate background texture
    if tf.random.uniform(()) > 0.7:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02)
        image = image + noise
        image = tf.clip_by_value(image, 0, 1)  # Keep values in valid range
    
    # Local intensity adjustment - randomly adjust regions to improve background variation
    if tf.random.uniform(()) > 0.8:
        # Create a random pattern for adjustment
        pattern = tf.random.normal([IMAGE_SIZE//8, IMAGE_SIZE//8, 1], mean=1.0, stddev=0.1)
        pattern = tf.image.resize(pattern, [IMAGE_SIZE, IMAGE_SIZE])
        # Apply the pattern mainly to background areas
        mask_expanded = tf.expand_dims(mask, axis=-1) if len(tf.shape(mask)) < 3 else mask
        mask_inv = 1.0 - tf.cast(mask_expanded, tf.float32)  # Inverse mask (1 for background)
        adjustment = 1.0 + (pattern - 1.0) * 0.3 * mask_inv  # Scale pattern effect
        image = image * adjustment
        image = tf.clip_by_value(image, 0, 1)
    
    # Ensure image values stay in [0, 1] range
    image = tf.clip_by_value(image, 0, 1)
    
    # Remove extra dimension from mask if we added it
    if len(tf.shape(mask)) == 3 and tf.shape(mask)[-1] == 1:
        mask = tf.squeeze(mask, axis=-1)
    
    return image, mask

def add_boundary_weights(mask, width=3):
    """
    Create boundary weight map to focus on vessel boundaries during training.
    
    Args:
        mask: Binary segmentation mask
        width: Width of boundary region in pixels
    
    Returns:
        Weight map with higher weights near boundaries
    """
    # Cast mask to float32 for dilation/erosion operations
    mask_float = tf.cast(mask, tf.float32)
    
    # Create kernels for dilation and erosion
    kernel_size = 2 * width + 1
    kernel = tf.ones((kernel_size, kernel_size), dtype=tf.float32)
    kernel = tf.reshape(kernel, [kernel_size, kernel_size, 1, 1])
    
    # Expand dimensions for conv operations
    mask_4d = tf.expand_dims(tf.expand_dims(mask_float, 0), -1)
    
    # Fix for dilation2d and erosion2d - properly specify all required parameters
    dilated = tf.nn.dilation2d(
        input=mask_4d,
        filters=kernel,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format="NHWC",
        dilations=[1, 1, 1, 1]
    )
    
    eroded = tf.nn.erosion2d(
        input=mask_4d,
        filters=kernel,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format="NHWC",
        dilations=[1, 1, 1, 1]
    )
    
    # Boundary is the difference between dilation and erosion
    boundary = tf.squeeze(dilated - eroded, [0, -1])
    
    # Create weight map (1 for non-boundary, higher for boundary)
    from configuration import BOUNDARY_AWARE_TRAINING
    weight_map = tf.ones_like(mask_float) + boundary * (BOUNDARY_AWARE_TRAINING['boundary_weight'] - 1.0)
    
    return weight_map

def resize_image_mask(image, mask, size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Resize image and mask to specified size."""
    # Resize image using bilinear interpolation
    image = tf.image.resize(image, size)
    
    # Resize mask using nearest neighbor to preserve segmentation labels
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.image.resize(mask, size, method='nearest')
    mask = tf.squeeze(mask, axis=-1)
    
    return image, mask

def extract_orthogonal_slices(image_volume, mask_volume, axis=0):
    """
    Extract slices along a specified axis from a 3D volume.
    
    Args:
        image_volume: 3D image volume
        mask_volume: 3D mask volume
        axis: The axis to slice along (0=Z, 1=Y, 2=X)
    
    Returns:
        List of (image_slice, mask_slice) tuples
    """
    slices = []
    
    # Get the size of the specified dimension
    dim_size = image_volume.shape[axis]
    
    # Create a function to extract a slice along the specified axis
    def get_slice(volume, idx, axis):
        if axis == 0:  # Z-axis (original slicing)
            return volume[idx, :, :]
        elif axis == 1:  # Y-axis
            return volume[:, idx, :]
        elif axis == 2:  # X-axis
            return volume[:, :, idx]
    
    # Extract slices
    for i in range(dim_size):
        img_slice = get_slice(image_volume, i, axis)
        mask_slice = get_slice(mask_volume, i, axis)
        
        # Ensure slices have consistent dimensions
        if axis == 1:  # Y-axis slices
            img_slice = np.transpose(img_slice)
            mask_slice = np.transpose(mask_slice)
        elif axis == 2:  # X-axis slices
            img_slice = np.transpose(img_slice)
            mask_slice = np.transpose(mask_slice)
            
        slices.append((img_slice, mask_slice))
    
    return slices

def load_multi_axis_slice(img_path, mask_path, slice_idx, axis):
    """Load a single slice from image and mask files along a specified axis."""
    try:
        # Convert tensor inputs to Python native types
        if isinstance(img_path, tf.Tensor):
            img_path = img_path.numpy().decode('utf-8')
        if isinstance(mask_path, tf.Tensor):
            mask_path = mask_path.numpy().decode('utf-8')
        if isinstance(slice_idx, tf.Tensor):
            slice_idx = int(slice_idx.numpy())
        if isinstance(axis, tf.Tensor):
            axis = int(axis.numpy())
        
        # Only use Z-axis (original direction) for reliable operation
        # This avoids the 'image data are not memory-mappable' error
        with tifffile.TiffFile(img_path) as tif:
            img_slice = tif.asarray(key=slice_idx)
        with tifffile.TiffFile(mask_path) as tif:
            mask_slice = tif.asarray(key=slice_idx)
        
        # Enhanced NaN handling - check and replace NaNs more thoroughly
        nan_count = np.sum(np.isnan(img_slice))
        inf_count = np.sum(np.isinf(img_slice))
        
        if nan_count > 0 or inf_count > 0:
            if DEBUG_DATA_LOADING:
                print(f"Warning: Found {nan_count} NaN values and {inf_count} infinite values in slice {slice_idx} from {img_path}")
            
            # Replace all NaN and infinite values
            img_slice = np.nan_to_num(img_slice, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Add robustness to normalization to prevent division by zero
        img_min = np.min(img_slice)
        img_max = np.max(img_slice)
        epsilon = 1e-8
        
        # Check if min and max are the same (constant image)
        if abs(img_max - img_min) < epsilon:
            # For constant images, just use the value divided by max possible value
            if img_min > 0:
                img_slice = np.ones_like(img_slice) * 0.5  # Use mid-gray for constant non-zero images
            else:
                img_slice = np.zeros_like(img_slice)  # Use black for constant zero images
        else:
            # Regular normalization with safeguards
            img_slice = (img_slice - img_min) / (img_max - img_min + epsilon)
        
        # Extra check for NaN after normalization
        if np.isnan(img_slice).any() or np.isinf(img_slice).any():
            if DEBUG_DATA_LOADING:
                print(f"Warning: NaNs or Infs appeared after normalization in slice {slice_idx}! Replacing with zeros.")
            img_slice = np.nan_to_num(img_slice, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure values are in valid range
        img_slice = np.clip(img_slice, 0.0, 1.0)
        
        # Process mask into binary segmentation
        mask_binary = np.zeros_like(mask_slice, dtype=np.int32)
        mask_binary[mask_slice > 0] = 1
        
        # Memory optimization: Resize large images immediately after loading
        if img_slice.shape[0] > IMAGE_SIZE or img_slice.shape[1] > IMAGE_SIZE:
            # Use smaller intermediate size for extremely large images
            max_dim = max(img_slice.shape[0], img_slice.shape[1])
            if max_dim > 2048:
                # Two-step downsampling for very large images
                interim_size = 1536
                img_slice = tf.image.resize(
                    tf.expand_dims(tf.convert_to_tensor(img_slice, dtype=tf.float32), axis=-1),
                    (interim_size, interim_size)
                ).numpy()
                img_slice = tf.image.resize(img_slice, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
                
                mask_binary = tf.image.resize(
                    tf.expand_dims(tf.convert_to_tensor(mask_binary, dtype=tf.int32), axis=-1),
                    (IMAGE_SIZE, IMAGE_SIZE),
                    method='nearest'
                ).numpy()
            else:
                # Standard downsampling for moderately large images
                img_slice = tf.image.resize(
                    tf.expand_dims(tf.convert_to_tensor(img_slice, dtype=tf.float32), axis=-1),
                    (IMAGE_SIZE, IMAGE_SIZE)
                ).numpy()
                
                mask_binary = tf.image.resize(
                    tf.expand_dims(tf.convert_to_tensor(mask_binary, dtype=tf.int32), axis=-1),
                    (IMAGE_SIZE, IMAGE_SIZE),
                    method='nearest'
                ).numpy()
            
            mask_binary = mask_binary.astype(np.int32).squeeze()
        else:
            # Add channel dimension for image
            img_slice = np.expand_dims(img_slice, axis=-1).astype(np.float32)
        
        # Final NaN check before returning
        img_slice = np.nan_to_num(img_slice, nan=0.0, posinf=1.0, neginf=0.0)
        
        return img_slice, mask_binary
        
    except Exception as e:
        if DEBUG_DATA_LOADING:
            print(f"Error loading slice {slice_idx} from axis {axis}: {str(e)}")
        # Return placeholder data in case of error
        return (np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32), 
               np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32))

def calculate_vessel_percentage(mask_path):
    """
    Calculate the percentage of vessel pixels in a mask.
    
    Args:
        mask_path: Path to the mask file
        
    Returns:
        Float: Percentage of vessel pixels in the mask (0-100)
    """
    try:
        with tifffile.TiffFile(mask_path) as tif:
            mask = tif.asarray()
        
        # Calculate percentage of non-zero pixels (vessels)
        total_pixels = mask.size
        vessel_pixels = np.sum(mask > 0)
        vessel_percentage = (vessel_pixels / total_pixels) * 100
        
        return vessel_percentage
    except Exception as e:
        if DEBUG_DATA_LOADING:
            print(f"Error calculating vessel percentage for {mask_path}: {str(e)}")
        return 0.0

def analyze_dataset_vessel_statistics(pairs):
    """
    Analyze vessel statistics across the dataset.
    
    Args:
        pairs: List of (image_path, mask_path) tuples
        
    Returns:
        Dictionary of vessel statistics
    """
    vessel_percentages = []
    
    for _, mask_path in pairs:
        percentage = calculate_vessel_percentage(mask_path)
        vessel_percentages.append(percentage)
    
    # Calculate statistics
    avg_percentage = np.mean(vessel_percentages) if vessel_percentages else 0
    median_percentage = np.median(vessel_percentages) if vessel_percentages else 0
    min_percentage = np.min(vessel_percentages) if vessel_percentages else 0
    max_percentage = np.max(vessel_percentages) if vessel_percentages else 0
    
    stats = {
        'average_vessel_percentage': avg_percentage,
        'median_vessel_percentage': median_percentage,
        'min_vessel_percentage': min_percentage,
        'max_vessel_percentage': max_percentage,
        'target_vessel_percentage': avg_percentage * 2.0,  # Target percentage is 2x the average
    }
    
    if DEBUG_DATA_LOADING:
        print(f"Dataset vessel statistics:")
        print(f"  Average vessel percentage: {avg_percentage:.4f}%")
        print(f"  Median vessel percentage: {median_percentage:.4f}%")
        print(f"  Min vessel percentage: {min_percentage:.4f}%")
        print(f"  Max vessel percentage: {max_percentage:.4f}%")
        print(f"  Target vessel percentage: {stats['target_vessel_percentage']:.4f}%")
    
    return stats

def filter_empty_slices(slice_info_list):
    """
    Filter out slices that have only background (no vessels) or contain NaN values.
    
    Args:
        slice_info_list: List of slice info dictionaries
        
    Returns:
        List of filtered slice info dictionaries
    """
    filtered_slices = []
    total_slices = len(slice_info_list)
    empty_slices = 0
    nan_slices = 0
    
    for info in slice_info_list:
        img_path = info['img_path']
        mask_path = info['mask_path']
        slice_idx = info['slice_idx']
        
        try:
            # Check for NaN values in the image slice - enhanced check
            with tifffile.TiffFile(img_path) as tif:
                if slice_idx < tif.series[0].shape[0]:
                    img_slice = tif.asarray(key=slice_idx)
                    
                    # Check if slice contains NaN or Inf values
                    has_nan = np.isnan(img_slice).any()
                    has_inf = np.isinf(img_slice).any()
                    
                    # Check if slice is constant (min == max) which can cause normalization issues
                    img_min = np.min(img_slice) if not has_nan and not has_inf else 0
                    img_max = np.max(img_slice) if not has_nan and not has_inf else 1
                    is_constant = abs(img_max - img_min) < 1e-8
                    
                    # Skip problematic slices
                    if has_nan or has_inf or is_constant:
                        if has_nan or has_inf:
                            nan_slices += 1
                            if DEBUG_DATA_LOADING:
                                print(f"Skipping slice {slice_idx} from {img_path} - contains NaN or Inf values")
                        if is_constant:
                            nan_slices += 1
                            if DEBUG_DATA_LOADING:
                                print(f"Skipping slice {slice_idx} from {img_path} - constant value slice")
                        continue  # Skip this slice, don't add to filtered_slices
            
            # Check if mask contains any vessels
            with tifffile.TiffFile(mask_path) as tif:
                if slice_idx < tif.series[0].shape[0]:
                    mask_slice = tif.asarray(key=slice_idx)
                    
                    # Check if slice has any vessels (non-zero pixels)
                    if np.any(mask_slice > 0):
                        filtered_slices.append(info)
                    else:
                        empty_slices += 1
                        if DEBUG_DATA_LOADING:
                            print(f"Skipping slice {slice_idx} from {mask_path} - contains only background")
        except Exception as e:
            # If there's an error, skip this slice rather than include it
            if DEBUG_DATA_LOADING:
                print(f"Error checking slice {slice_idx} from {mask_path}: {str(e)}")
            # Safer to skip problematic slices than include them
            nan_slices += 1
    
    if DEBUG_DATA_LOADING:
        print(f"Filtered {empty_slices} empty slices and {nan_slices} problematic slices out of {total_slices} total slices")
    else:
        print(f"Filtered {empty_slices} empty slices and {nan_slices} problematic slices")
        
    print(f"Remaining slices: {len(filtered_slices)}")
    
    return filtered_slices

def create_dataset_from_pairs(pairs, batch_size, slice_range=None, is_training=False):
    """Create a TensorFlow dataset with improved class balance for background."""
    if not pairs:
        raise ValueError("No image-mask pairs provided")
    
    # Create a list to store slice information (not the actual data)
    slice_info = []
    total_slices = 0
    
    # For each pair, gather metadata but don't load the data yet
    for img_path, mask_path in pairs:
        try:
            # Load image and mask volumes for shape information
            with tifffile.TiffFile(img_path) as tif:
                image_shape = tif.series[0].shape
            with tifffile.TiffFile(mask_path) as tif:
                mask_shape = tif.series[0].shape
            
            if DEBUG_DATA_LOADING:
                print(f"Found volume: {img_path}")
                print(f"  Image shape: {image_shape}, Mask shape: {mask_shape}")
            
            # Only use Z-axis slices regardless of multi-axis settings
            # Determine range of slices to use
            if slice_range is None:
                # Use all slices
                min_slices = min(image_shape[0], mask_shape[0])
                usable_indices = range(min_slices)
                if DEBUG_DATA_LOADING:
                    print(f"  Using all {min_slices} slices from this volume")
                total_slices += min_slices
            else:
                # Use specified slice range
                start, end = slice_range
                end = min(end, min(image_shape[0], mask_shape[0]))
                usable_indices = range(start, end)
                if DEBUG_DATA_LOADING:
                    print(f"  Using {len(usable_indices)} slices (range {start}-{end}) from this volume")
                total_slices += len(usable_indices)
            
            # Store info for each usable slice
            for slice_idx in usable_indices:
                slice_info.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'slice_idx': slice_idx,
                    'axis': 0  # Z-axis only
                })
                
        except Exception as e:
            if DEBUG_DATA_LOADING:
                print(f"Error getting info for pair {(img_path, mask_path)}: {str(e)}")
            continue
    
    if not slice_info:
        raise ValueError("No valid slices could be identified from the provided volumes")
    
    if DEBUG_DATA_LOADING:
        print(f"Initially found {len(slice_info)} slices from {len(pairs)} volumes")
    
    # Filter out slices with only background if we're training
    if is_training:
        print("Filtering out empty slices (no vessels)...")
        slice_info = filter_empty_slices(slice_info)
        print(f"After filtering: {len(slice_info)} slices remain")
    
    if DEBUG_DATA_LOADING:
        print(f"Using a total of {len(slice_info)} slices for {'training' if is_training else 'validation'}")

    # Extract separate lists for each component
    img_paths = [info['img_path'] for info in slice_info]
    mask_paths = [info['mask_path'] for info in slice_info]
    slice_indices = [info['slice_idx'] for info in slice_info]
    slice_axes = [info['axis'] for info in slice_info]
    
    # Print total slice count to verify
    if DEBUG_DATA_LOADING:
        print(f"Total slices to be used: {len(slice_indices)}")
        print(f"Expected steps per epoch with batch size {batch_size}: {len(slice_indices) // batch_size}")
    
    # Create the dataset from individual components
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths, slice_indices, slice_axes))
    
    # Shuffle if training (with larger buffer for better randomization)
    if is_training:
        # Increase buffer size to improve randomization
        buffer_size = min(10000, len(slice_info))
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    
    # Optimize parallel loading with more parallel calls but don't overdo
    dataset = dataset.map(
        lambda img_path, mask_path, slice_idx, axis: tf.py_function(
            load_multi_axis_slice,
            [img_path, mask_path, slice_idx, axis],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=NUM_WORKERS
    )
    
    # Set shapes explicitly - add extra NaN check after loading
    dataset = dataset.map(
        lambda image, mask: (
            tf.ensure_shape(
                tf.where(tf.math.is_nan(image), tf.zeros_like(image), image),  # Replace any NaN values
                [IMAGE_SIZE, IMAGE_SIZE, 1]
            ),
            tf.ensure_shape(mask, [IMAGE_SIZE, IMAGE_SIZE])
        )
    )
    
    # Apply data augmentation only if training - use smaller parallel calls for memory efficiency
    if is_training:
        dataset = dataset.map(
            data_augmentation,
            num_parallel_calls=2  # Reduced to avoid memory pressure
        )
    
    # Add boundary weights if enabled
    if is_training and BOUNDARY_AWARE_TRAINING['enable']:
        dataset = dataset.map(
            lambda image, mask: (image, mask, add_boundary_weights(mask)),
            num_parallel_calls=NUM_WORKERS
        )
        # Add a filter to keep slices with sufficient background
        dataset = dataset.filter(
            lambda image, mask, weights: tf.reduce_mean(1.0 - tf.cast(mask, tf.float32)) > 0.3
        )
    
    # Add a final filter before batching to exclude any remaining problematic data
    dataset = dataset.filter(
        lambda image, mask: tf.logical_not(tf.reduce_any(tf.math.is_nan(image)))
    )
    
    # Skip cache which can consume too much memory
    # dataset = dataset.cache()
    
    # Efficiently batch and prefetch with smaller prefetch buffer
    def ensure_no_nans(images, masks):
        safe_images = tf.where(tf.math.is_nan(images), tf.zeros_like(images), images)
        safe_images = tf.where(tf.math.is_inf(safe_images), tf.ones_like(safe_images), safe_images)
        return safe_images, masks
        
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(ensure_no_nans)  # Apply NaN safety after batching
    dataset = dataset.prefetch(2)
    
    return dataset

def prepare_data(batch_size=None):
    """Prepare data for training and validation."""
    if DEBUG_DATA_LOADING:
        print("\nPreparing data for vessel segmentation...")
    
    # Use the provided batch size or default from configuration
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    # Find pairs of raw images and their segmentation masks
    pairs = find_data_pairs()
    if DEBUG_DATA_LOADING:
        print(f"Found {len(pairs)} image-mask pairs")
    
    if not pairs:
        raise FileNotFoundError(f"No valid image-mask pairs found in:\n  - Raw dir: {DATA_DIR}\n  - Mask dir: {RESULTS_DIR}")
    
    # Analyze vessel statistics and store in configuration
    vessel_stats = analyze_dataset_vessel_statistics(pairs)
    # Store the target vessel percentage in a global variable
    global TARGET_VESSEL_PERCENTAGE
    TARGET_VESSEL_PERCENTAGE = vessel_stats['target_vessel_percentage']
    print(f"Target vessel percentage: {TARGET_VESSEL_PERCENTAGE:.4f}%")
    
    # Make sure we have at least 2 samples for train/val split
    if len(pairs) == 1:
        # Duplicate the pair to allow for train/val split
        pairs = pairs * 2
        if DEBUG_DATA_LOADING:
            print("Warning: Only one pair found, duplicating it to allow for train/val split")
    
    # Split into train and validation
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )
    
    # Log the correct split sizes
    if DEBUG_DATA_LOADING:
        print(f"Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")
    
    # Create datasets with the specified batch size
    train_dataset = create_dataset_from_pairs(train_pairs, batch_size, SLICE_RANGE, is_training=True)
    val_dataset = create_dataset_from_pairs(val_pairs, batch_size, SLICE_RANGE, is_training=False)
    
    return train_dataset, val_dataset, TARGET_VESSEL_PERCENTAGE

if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    train_dataset, val_dataset, target_vessel_percentage = prepare_data()
    if DEBUG_DATA_LOADING:
        print("Data loading successful!")