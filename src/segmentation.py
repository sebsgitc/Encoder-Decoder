import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import tifffile as tiff
import signal
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation, binary_closing
import gc
import heapq
import time

# Path definitions for lung CT scan processing
PREPROCESSED_DIR = "3d-stacks"  # Directory containing the CT scan data
OUTPUT_DIR = os.path.join("output", "segmentation_2d_stack")

# Create output directory with parents
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add signal handling for Ctrl+C
def signal_handler(sig, frame):
    print('\nCaught Ctrl+C! Cleaning up...')
    tf.keras.backend.clear_session()
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Configure TensorFlow to use GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Option 1: Use memory growth (more flexible)
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set environment variables for better memory management
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    except RuntimeError as e:
        print(f"GPU configuration failed: {e}")

def load_image(filename):
    """Loads a preprocessed 8-bit grayscale TIFF image."""
    print(f"Looking for file {filename} in subdirectories...")
    # Look in subfolders for the file
    for subfolder in ["r01_"]:  # We're focusing on r01_ subfolder as specified
        filepath = os.path.join(PREPROCESSED_DIR, subfolder, filename)
        if os.path.exists(filepath):
            print(f"Found file at: {filepath}")
            img = tiff.imread(filepath)
            print(f"Loaded image with shape: {img.shape}, dtype: {img.dtype}")
            print(f"Value range: [{np.min(img)}, {np.max(img)}]")
            return img
    raise FileNotFoundError(f"Could not find {filename} in any subfolder")

def invert_for_dark_vessels(image):
    """Inverts the image to optimize for dark vessel interiors.
    In CT scans, blood vessels often appear darker than surrounding tissue."""
    print("Inverting image for dark vessel segmentation...")
    max_val = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else np.max(image)
    return max_val - image

def fast_marching_segmentation_gpu(image, seed_points, threshold=150, stop_threshold=0.3, gradient_weight=2.0, intensity_weight=5.0):
    """
    Implements Fast Marching Method for blood vessel segmentation optimized for GPU.
    This is a completely redesigned version that prioritizes effective segmentation over speed.
    """
    print(f"Processing volume of shape: {image.shape}")
    print(f"Using {len(seed_points)} seed points for segmentation")
    print(f"Parameters: threshold={threshold}, stop_threshold={stop_threshold}, "
          f"gradient_weight={gradient_weight}, intensity_weight={intensity_weight}")
    
    # Convert to numpy and normalize
    image_np = np.array(image, dtype=np.float32)
    original_shape = image_np.shape
    
    # For dark vessels, invert the image
    image_np = 255 - image_np
    print("Image inverted for dark vessel segmentation")
    
    # Use SimpleITK for reliable segmentation
    print("Converting to SimpleITK for segmentation...")
    sitk_image = sitk.GetImageFromArray(image_np)
    
    # Create seed mask - using uint8 instead of bool for SimpleITK compatibility
    print("Creating seed mask...")
    seeds = np.zeros_like(image_np, dtype=np.uint8)
    for x, y, z in seed_points:
        if 0 <= z < seeds.shape[0] and 0 <= y < seeds.shape[1] and 0 <= x < seeds.shape[2]:
            seeds[z, y, x] = 1
    
    # Dilate seeds slightly for better initialization - ensure uint8 output
    print("Dilating seeds...")
    struct_elem = np.ones((3,3,3), dtype=np.uint8)
    seeds_dilated = binary_dilation(seeds, structure=struct_elem).astype(np.uint8)
    
    # Convert seeds to SimpleITK - ensure uint8 type
    seed_image = sitk.GetImageFromArray(seeds_dilated)
    
    # Apply Gaussian smoothing
    print("Applying gaussian smoothing...")
    smoothed = sitk.DiscreteGaussian(sitk_image, variance=1.0)
    
    # Compute gradient magnitude (edges)
    print("Computing gradient magnitude...")
    gradient = sitk.GradientMagnitude(smoothed)
    
    # Create edge-preserving speed function
    # Low values at boundaries (high gradient), high values inside vessels
    print("Creating speed function...")
    speed = 1.0 / (1.0 + gradient * gradient_weight)
    
    # Setup and execute fast marching
    print("Setting up Fast Marching...")
    try:
        fast_march = sitk.FastMarchingBaseImageFilter()
        
        # Set trial points from seed mask
        seeds_array = sitk.GetArrayFromImage(seed_image)
        trial_points = []
        trial_values = []
        
        # Find all seed points
        seed_indices = np.argwhere(seeds_array > 0)
        print(f"Setting up {len(seed_indices)} trial points...")
        
        # Convert to SimpleITK format (z,y,x order)
        for z, y, x in seed_indices:
            trial_points.append([int(z), int(y), int(x)])
            trial_values.append(0.0)  # Initial time is 0 for seeds
            
        # Set trial points
        if len(trial_points) > 0:
            fast_march.SetTrialPoints(trial_points, trial_values)
        else:
            print("Warning: No valid trial points found. Using original seed points directly.")
            # Fall back to direct seed points if no trial points found
            for x, y, z in seed_points:
                if 0 <= z < image_np.shape[0] and 0 <= y < image_np.shape[1] and 0 <= x < image_np.shape[2]:
                    trial_points.append([int(z), int(y), int(x)])
                    trial_values.append(0.0)
            fast_march.SetTrialPoints(trial_points, trial_values)
        
        # Set stopping parameters
        stopping_time = 1000.0  # Large value to avoid premature stopping
        fast_march.SetStoppingValue(stopping_time)
        
        # Run Fast Marching
        print("Running Fast Marching algorithm...")
        start_time = time.time()
        time_map = fast_march.Execute(speed)
        print(f"Fast Marching completed in {time.time() - start_time:.2f} seconds")
        
        # Create segmentation by thresholding time map
        print(f"Applying threshold {threshold} to time map...")
        segmentation = sitk.BinaryThreshold(
            time_map,
            lowerThreshold=0.0,  # Minimum time
            upperThreshold=threshold,  # Maximum time to include
            insideValue=1,
            outsideValue=0
        )
        
        # Convert back to numpy - ensure uint8 type
        result = sitk.GetArrayFromImage(segmentation).astype(np.uint8)
        
        # Post-processing: Clean up the segmentation
        print("Post-processing segmentation...")
        # Fill small holes - ensure uint8 output
        result_closed = binary_closing(result, structure=np.ones((3,3,3))).astype(np.uint8)
        
        # Keep only the largest connected components
        print("Finding connected components...")
        # Label connected components
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)
        components = cc_filter.Execute(sitk.GetImageFromArray(result_closed))
        
        # Relabel by size
        relabel = sitk.RelabelComponentImageFilter()
        relabel.SetSortByObjectSize(True)
        relabeled = relabel.Execute(components)
        
        # Keep only largest components
        num_components = min(10, relabel.GetNumberOfObjects())
        print(f"Keeping {num_components} largest components...")
        
        final_segmentation = sitk.BinaryThreshold(
            relabeled,
            lowerThreshold=1,
            upperThreshold=num_components,
            insideValue=255,
            outsideValue=0
        )
        
        # Convert back to numpy
        result = sitk.GetArrayFromImage(final_segmentation)
        
    except Exception as e:
        print(f"Error in SimpleITK Fast Marching: {str(e)}")
        print("Falling back to region growing...")
        
        # Fallback to simpler approach: Region growing
        result = np.zeros_like(image_np, dtype=np.uint8)
        
        # Convert to SimpleITK
        sitk_image = sitk.GetImageFromArray(image_np)
        
        # Create a segmentation for each seed point
        for i, (x, y, z) in enumerate(seed_points):
            print(f"Processing seed {i+1}/{len(seed_points)}: ({x}, {y}, {z})")
            
            # Ensure seed is within bounds
            if not (0 <= z < image_np.shape[0] and 0 <= y < image_np.shape[1] and 0 <= x < image_np.shape[2]):
                print(f"  Seed point out of bounds, skipping")
                continue
                
            try:
                # Get seed value
                seed_value = image_np[z, y, x]
                
                # Calculate local thresholds based on seed value
                lower = max(0, seed_value - 25)  # More permissive lower threshold
                upper = min(255, seed_value + 25)  # More permissive upper threshold
                
                # Create seed point array for SimpleITK (z,y,x order)
                seed = [int(z), int(y), int(x)]
                
                # Perform region growing
                print(f"  Region growing with thresholds: [{lower}, {upper}]")
                segmentation = sitk.ConnectedThreshold(
                    sitk_image,
                    seedList=[seed],
                    lower=lower,
                    upper=upper,
                    replaceValue=1
                )
                
                # Apply a small closing to fill gaps
                closed = sitk.BinaryMorphologicalClosing(segmentation, [2, 2, 2])
                
                # Convert back to numpy
                seed_result = sitk.GetArrayFromImage(closed)
                
                # Combine with overall result
                result = np.logical_or(result, seed_result).astype(np.uint8) * 255
                
                print(f"  Segmented {np.sum(seed_result > 0)} voxels from this seed")
                
            except Exception as seed_error:
                print(f"  Error processing seed: {str(seed_error)}")
                continue
    
    print(f"Total segmented voxels: {np.sum(result > 0)}")
    print(f"Segmentation percentage: {np.sum(result > 0) / np.prod(result.shape) * 100:.6f}%")
    
    return result

def calculate_gradient(tensor, axis=0):
    """Calculate gradient along specified axis using NumPy for reliability"""
    # Convert to numpy for calculating gradients
    tensor_np = tensor.numpy()
    gradient = np.zeros_like(tensor_np)
    
    # Calculate gradients using numpy slicing (more reliable)
    if axis == 0 and tensor_np.shape[0] > 2:
        # For first slice, use forward difference
        gradient[0] = tensor_np[1] - tensor_np[0]
        # For middle slices, use central difference
        gradient[1:-1] = (tensor_np[2:] - tensor_np[:-2]) / 2.0
        # For last slice, use backward difference
        gradient[-1] = tensor_np[-1] - tensor_np[-2]
    elif axis == 1 and tensor_np.shape[1] > 2:
        # For first slice, use forward difference
        gradient[:, 0] = tensor_np[:, 1] - tensor_np[:, 0]
        # For middle slices, use central difference
        gradient[:, 1:-1] = (tensor_np[:, 2:] - tensor_np[:, :-2]) / 2.0
        # For last slice, use backward difference
        gradient[:, -1] = tensor_np[:, -1] - tensor_np[:, -2]
    elif axis == 2 and tensor_np.shape[2] > 2:
        # For first slice, use forward difference
        gradient[:, :, 0] = tensor_np[:, :, 1] - tensor_np[:, :, 0]
        # For middle slices, use central difference
        gradient[:, :, 1:-1] = (tensor_np[:, :, 2:] - tensor_np[:, :, :-2]) / 2.0
        # For last slice, use backward difference
        gradient[:, :, -1] = tensor_np[:, :, -1] - tensor_np[:, :, -2]
    
    # Convert back to tensor
    return tf.convert_to_tensor(gradient, dtype=tf.float32)

def process_volume_in_chunks(image_tensor, chunk_size=128):
    """Process large volumes in chunks with Gaussian smoothing using SciPy"""
    print("Processing volume in chunks for Gaussian smoothing...")
    
    # Convert to numpy array for scipy processing
    image_np = image_tensor.numpy()
    processed = np.zeros_like(image_np)
    
    # Small sigma for minimal smoothing to preserve vessel details
    sigma = 0.5  # Reduced from 0.7 for speed
    
    print("Using SciPy gaussian_filter for smoothing...")
    try:
        # Process in z-slabs for better memory efficiency
        for start_z in range(0, image_np.shape[0], chunk_size):
            end_z = min(start_z + chunk_size, image_np.shape[0])
            print(f"Smoothing chunk {start_z//chunk_size + 1}/{(image_np.shape[0] + chunk_size - 1)//chunk_size}")
            
            # Extract chunk
            chunk = image_np[start_z:end_z]
            
            # Apply gaussian smoothing using scipy - optional skip for speed
            if sigma > 0:
                chunk_smoothed = gaussian_filter(chunk, sigma=sigma)
            else:
                chunk_smoothed = chunk  # Skip smoothing
            
            # Store smoothed chunk
            processed[start_z:end_z] = chunk_smoothed
            
        print("Gaussian smoothing completed")
        # Convert back to tensor
        return tf.convert_to_tensor(processed, dtype=tf.float32)
        
    except Exception as e:
        print(f"SciPy-based smoothing failed with error: {str(e)}")
        print("Returning original unsmoothed image")
        return image_tensor  # Return original as last resort

def save_segmentation(mask, filename):
    """Saves the segmentation mask as a TIFF file."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create full output path
    save_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        print(f"Saving segmentation mask to: {save_path}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Mask values: min={np.min(mask)}, max={np.max(mask)}, unique values={np.unique(mask)}")
        
        # Save as 8-bit TIFF
        tiff.imwrite(save_path, mask.astype(np.uint8))
        print(f"Successfully saved segmentation to: {save_path}")
    except Exception as e:
        print(f"Error saving file {save_path}: {str(e)}")
        raise

def segment_image(filename, seed_points, **kwargs):
    """Segment a single image using the Fast Marching algorithm."""
    print(f"\n{'='*80}")
    print(f"Segmenting image: {filename}")
    print(f"{'='*80}")
    
    image = load_image(filename)
    mask = fast_marching_segmentation_gpu(image, seed_points, **kwargs)
    save_segmentation(mask, filename)
    return mask

def segment_image_stack(pattern="*.tif", seed_points=None, **kwargs):
    """Segments 3D image stacks using provided (x,y,z) seed points."""
    default_params = {
        'threshold': 100,  # Threshold for distance in Fast Marching
        'stop_threshold': 0.5,  # When to stop propagation
        'gradient_weight': 1.0,  # Weight for edge detection (lower = more propagation)
        'intensity_weight': 2.0  # Weight for intensities (higher = favor brighter areas after inversion)
    }
    kwargs = {**default_params, **kwargs}
    
    print("\nBlood Vessel Segmentation for Lung CT Scans")
    print("=" * 60)
    print(f"Looking for files matching pattern: {pattern}")
    print(f"Segmentation parameters: {kwargs}")
    
    if seed_points is None:
        # Default 3D seed points in (x,y,z) format
        seed_points = [(478, 323, 32), (372, 648, 45), (920, 600, 72),
                    (420, 457, 24), (369, 326, 74), (753, 417, 124),
                    (755, 607, 174), (887, 507, 224), (305, 195, 274),
                    (574, 476, 324), (380, 625, 374), (313, 660, 424)]
    
    print(f"Using {len(seed_points)} seed points:")
    for i, (x, y, z) in enumerate(seed_points):
        print(f"  Seed point {i+1}: (x={x}, y={y}, z={z})")
    
    # Search in subfolders
    files = []
    subfolder = "r01_"  # We're focusing on r01_ subfolder as specified
    subfolder_path = os.path.join(PREPROCESSED_DIR, subfolder)
    if os.path.exists(subfolder_path):
        files.extend(glob(os.path.join(subfolder_path, pattern)))
    
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern} in {subfolder_path}")
    
    print(f"Found {len(files)} files to process")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    
    # Process each file
    for filepath in tqdm(files, desc="Processing stacks"):
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename}")
        try:
            image = tiff.imread(filepath)
            print(f"Loaded image shape: {image.shape}")
            
            mask = fast_marching_segmentation_gpu(image, seed_points, **kwargs)
            save_segmentation(mask, filename)
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            tf.keras.backend.clear_session()
            return
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            tf.keras.backend.clear_session()
            continue

# Add this new function to provide more options for speed
def segment_image_stack_downsampled(pattern="*.tif", seed_points=None, downsample_factor=2, **kwargs):
    """Segments 3D image stacks using downsampling for faster results"""
    print(f"\nUsing downsampled segmentation with factor {downsample_factor} for faster results")
    
    # ...same as segment_image_stack but with extra downsampling step...
    # Get code from segment_image_stack and add downsampling
    
    # Adjust default params for downsampled version
    default_params = {
        'threshold': 50,  # Lower threshold for downsampled data
        'stop_threshold': 0.5,
        'gradient_weight': 0.8,  # Lower for more propagation
        'intensity_weight': 3.0  # Higher to compensate for downsampling
    }
    kwargs = {**default_params, **kwargs}
    
    print("\nBlood Vessel Segmentation for Lung CT Scans")
    print("=" * 60)
    print(f"Looking for files matching pattern: {pattern}")
    print(f"Segmentation parameters: {kwargs}")
    
    if seed_points is None:
        # Default 3D seed points in (x,y,z) format
        seed_points = [(478, 323, 32), (372, 648, 45), (920, 600, 72),
                    (420, 457, 24), (369, 326, 74), (753, 417, 124),
                    (755, 607, 174), (887, 507, 224), (305, 195, 274),
                    (574, 476, 324), (380, 625, 374), (313, 660, 424)]
    
    print(f"Using {len(seed_points)} seed points:")
    for i, (x, y, z) in enumerate(seed_points):
        print(f"  Seed point {i+1}: (x={x}, y={y}, z={z})")
    
    # Search in subfolders
    files = []
    subfolder = "r01_"  # We're focusing on r01_ subfolder as specified
    subfolder_path = os.path.join(PREPROCESSED_DIR, subfolder)
    if os.path.exists(subfolder_path):
        files.extend(glob(os.path.join(subfolder_path, pattern)))
    
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern} in {subfolder_path}")
    
    print(f"Found {len(files)} files to process")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    
    # Process each file
    for filepath in tqdm(files, desc="Processing stacks"):
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename}")
        try:
            image = tiff.imread(filepath)
            print(f"Loaded image shape: {image.shape}")
            
            # Downsample image
            print(f"Downsampling by factor of {downsample_factor} for faster processing")
            image = image[::downsample_factor, ::downsample_factor, ::downsample_factor]
            # Adjust seed points
            seed_points = [(x//downsample_factor, y//downsample_factor, z//downsample_factor) 
                           for x, y, z in seed_points]
            print(f"New volume shape after downsampling: {image.shape}")
            print(f"New seed points: {seed_points}")
            
            mask = fast_marching_segmentation_gpu(image, seed_points, **kwargs)
            save_segmentation(mask, filename)
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            tf.keras.backend.clear_session()
            return
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            tf.keras.backend.clear_session()
            continue

# ENHANCEMENT: Add specialized version for blood vessel segmentation
def segment_blood_vessels(pattern="*.tif", seed_points=None, **kwargs):
    """
    Specialized function for blood vessel segmentation with optimized parameters
    """
    # Set optimal parameters for blood vessel segmentation
    vessel_params = {
        'threshold': 80,  # Lower threshold to capture more vessels
        'stop_threshold': 0.4,
        'gradient_weight': 1.5,  # Lower to allow more propagation across edges
        'intensity_weight': 6.0,  # Higher to emphasize vessel intensity
    }
    
    # Override with any user-provided parameters
    vessel_params.update(kwargs)
    
    # Call the general segmentation function with vessel-specific parameters
    return segment_image_stack(pattern, seed_points, **vessel_params)

# Main execution block with better GPU checks and error handling
if __name__ == "__main__":
    print("\nLung CT Scan Blood Vessel Segmentation")
    print("=" * 60)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  GPU {i}: {details.get('device_name', 'Unknown')}")
                
                # Configure memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  Memory growth enabled for GPU {i}")
            except Exception as e:
                print(f"  Error configuring GPU {i}: {e}")
    else:
        print("No GPUs found. Using CPU for computation (this will be slower).")
    
    # Define seed points for r01_ blood vessel segmentation
    # These points represent starting locations inside blood vessels
    seed_points_r01 = [
        (478, 323, 32),
        (372, 648, 45),
        (920, 600, 72),
        (420, 457, 24),
        (369, 326, 74),
        (753, 417, 124),
        (755, 607, 174),
        (887, 507, 224),
        (305, 195, 274),
        (574, 476, 324),
        (380, 625, 374),
        (313, 660, 424),
        (100, 512, 610),
        (512, 20, 730),
        (512, 200, 820),
        (512, 400, 940)
    ]
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Input directory: {os.path.abspath(PREPROCESSED_DIR)}")
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Number of seed points: {len(seed_points_r01)}")
    
    # Enable this line to use downsampled segmentation for faster results
    use_downsampling = False
    
    # Add options for different segmentation strategies
    segmentation_mode = "region_growing"  # Options: "fast", "accurate", "balanced"
    use_downsampling = False
    
    try:
        print("\nStarting segmentation...")
        
        if segmentation_mode == "fast":
            print("Using FAST mode - prioritizing speed over accuracy")
            use_downsampling = True
            # For faster results with lower resolution
            segment_image_stack_downsampled(
                pattern="r01_.8bit.tif",
                seed_points=seed_points_r01,
                downsample_factor=2,
                threshold=80,
                stop_threshold=0.2,  # Lower to stop earlier
                gradient_weight=1.0,  # Lower for more aggressive growth
                intensity_weight=4.0
            )
        elif segmentation_mode == "accurate":
            print("Using ACCURATE mode - prioritizing accuracy over speed")
            # Original full-resolution segmentation with vessel-specific parameters
            segment_blood_vessels(
                pattern="r01_.8bit.tif",
                seed_points=seed_points_r01,
                threshold=120,
                stop_threshold=0.5,  # Higher for more complete filling
                gradient_weight=2.0,
                intensity_weight=5.0
            )
        else:  # "balanced"
            print("Using BALANCED mode - compromise between speed and accuracy")
            segment_image_stack(
                pattern="r01_.8bit.tif",
                seed_points=seed_points_r01,
                threshold=100,
                stop_threshold=0.3,
                gradient_weight=1.5,
                intensity_weight=4.0
            )
            
        print("\nSegmentation complete!")
    except KeyboardInterrupt:
        print("\nSegmentation interrupted by user")
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"\nError during segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        tf.keras.backend.clear_session()