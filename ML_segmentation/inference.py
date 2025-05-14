"""Inference script for vessel segmentation on new 3D volumes."""

import os
import argparse
import numpy as np
import tensorflow as tf
import tifffile
from tqdm import tqdm
import time
import glob
import shutil
import psutil
import gc

from configuration import *
from model import get_model
from utils import dice_coefficient

def memory_stats():
    """Return current memory usage statistics as formatted string."""
    mem = psutil.virtual_memory()
    return f"Memory: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)"

def find_checkpoint_file(checkpoint_dir="checkpoints", priority_files=None):
    """
    Find the best available model checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        priority_files: List of filenames to prioritize, in order of preference
        
    Returns:
        Path to best checkpoint file or None if not found
    """
    if priority_files is None:
        # Default priority order for checkpoints
        priority_files = [
            "final_model.weights.h5",            # Final combined model (if available)
            "best_vessel_recall.weights.h5",      # Model with best vessel recall
            "model_best.weights.h5",             # Model with best validation loss
            "checkpoint.weights.h5"              # Latest checkpoint
        ]
    
    # First try exact matches in priority order
    for filename in priority_files:
        filepath = os.path.join(checkpoint_dir, filename)
        if os.path.exists(filepath):
            return filepath
            
    # If no exact match, look for any .h5 files
    h5_files = glob.glob(os.path.join(checkpoint_dir, "*.h5"))
    if h5_files:
        # Sort by modification time (newest first)
        return sorted(h5_files, key=os.path.getmtime, reverse=True)[0]
    
    return None

def create_test_dataset(input_path, batch_size=4, slice_range=None):
    """
    Create a TensorFlow dataset from a 3D TIFF file for inference.
    
    Args:
        input_path: Path to input 3D TIFF file
        batch_size: Number of slices to process at once
        slice_range: Optional tuple of (start, end) to process only a range of slices
    
    Returns:
        dataset: TensorFlow dataset containing normalized slice images
        volume_shape: Shape of the input volume
    """
    print(f"Creating test dataset from {input_path}")
    
    # Get the volume shape without loading the entire file
    with tifffile.TiffFile(input_path) as tif:
        volume_shape = tif.series[0].shape
    
    print(f"Volume has shape: {volume_shape}")
    
    # Determine which slices to use
    if slice_range is not None:
        start, end = slice_range
        end = min(end, volume_shape[0])
        slices_to_use = range(start, end)
    else:
        slices_to_use = range(volume_shape[0])
    
    print(f"Using {len(slices_to_use)} slices for prediction")
    
    # Function to load and normalize a specific slice
    def load_slice(slice_idx):
        try:
            # Load slice
            with tifffile.TiffFile(input_path) as tif:
                slice_data = tif.asarray(key=int(slice_idx))
            
            # Normalize to [0, 1]
            slice_data = slice_data.astype(np.float32)
            min_val = np.min(slice_data)
            max_val = np.max(slice_data)
            if max_val > min_val:
                normalized = (slice_data - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(slice_data, dtype=np.float32)
            
            # Add channel dimension
            normalized = np.expand_dims(normalized, axis=-1)
            
            # Return normalized slice
            return normalized
        except Exception as e:
            print(f"Error loading slice {slice_idx}: {e}")
            # Return an empty slice of the expected shape
            return np.zeros((volume_shape[1], volume_shape[2], 1), dtype=np.float32)
    
    # Create a dataset of slice indices
    slice_indices = tf.data.Dataset.from_tensor_slices(list(slices_to_use))
    
    # Map each index to its corresponding image
    dataset = slice_indices.map(
        lambda idx: tf.py_function(
            load_slice,
            [idx],
            tf.float32
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Set the shapes explicitly
    dataset = dataset.map(lambda x: tf.ensure_shape(x, [volume_shape[1], volume_shape[2], 1]))
    
    # Batch the slices
    dataset = dataset.batch(batch_size)
    
    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, volume_shape

def segment_volume_tiled(model, input_path, output_path=None, threshold=0.5, 
                         batch_size=4, tile_size=256, overlap=64,
                         export_formats=None):
    """
    Segment a large 3D volume using tiling approach to manage memory.
    
    Args:
        model: Trained TensorFlow model
        input_path: Path to input 3D TIFF file
        output_path: Path to save the segmentation result (if None, a default path is used)
        threshold: Threshold for binary segmentation (default: 0.5)
        batch_size: Number of slices to process at once
        tile_size: Size of tiles to process at once (for large volumes)
        overlap: Overlap between tiles to avoid edge artifacts
        export_formats: List of additional export formats ('npy', 'png_slices', etc.)
        
    Returns:
        Path to the saved segmentation
    """
    start_time = time.time()
    print(f"Segmenting volume: {input_path}")
    print(f"Initial {memory_stats()}")
    
    # Create output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(RESULTS_DIR, "predicted")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_segmented.tif")
    
    # Create directory for the base name (for additional exports)
    export_dir = os.path.join(os.path.dirname(output_path), 
                             os.path.splitext(os.path.basename(output_path))[0])
    if export_formats:
        os.makedirs(export_dir, exist_ok=True)
    
    # Load the volume information without loading all data
    with tifffile.TiffFile(input_path) as tif:
        volume_shape = tif.series[0].shape
        dtype = tif.series[0].dtype
        
    print(f"Volume shape: {volume_shape}, dtype: {dtype}")
    
    # Initialize segmentation volume as memory-mapped file to avoid OOM
    mmap_file = os.path.join(os.path.dirname(output_path), f"temp_segmentation_{int(time.time())}.dat")
    segmentation = np.memmap(mmap_file, dtype=np.uint8, mode='w+', 
                            shape=volume_shape)
    
    # Set model to inference mode
    model.trainable = False
    
    # Determine if we need tiling in XY dimensions
    use_tiling = (volume_shape[1] > IMAGE_SIZE or volume_shape[2] > IMAGE_SIZE)
    
    # Process slices with or without tiling
    if not use_tiling:
        # Process whole slices in batches
        print(f"Processing volume as whole slices in batches of {batch_size}")
        
        # Create test dataset without loading the entire volume at once
        test_dataset, _ = create_test_dataset(input_path, batch_size)
        
        # Process batches
        slice_idx = 0
        for batch in tqdm(test_dataset, desc="Processing slices"):
            # Predict on batch
            predictions = model(batch, training=False).numpy()
            
            # Apply threshold
            binary_preds = (predictions > threshold).astype(np.uint8)
            
            # Store in segmentation volume
            batch_size_actual = binary_preds.shape[0]
            end_idx = min(slice_idx + batch_size_actual, volume_shape[0])
            segmentation[slice_idx:end_idx] = binary_preds.squeeze()[:end_idx-slice_idx]
            
            slice_idx += batch_size_actual
            
            # Force synchronization of memory map
            if slice_idx % 10 == 0:
                segmentation.flush()
        
    else:
        # Process with tiling - handle one Z slice at a time but tile in XY
        print(f"Processing volume with tiling (tile size: {tile_size}, overlap: {overlap})")
        
        # Calculate step size
        step_size = tile_size - overlap
        
        # Calculate number of tiles in each dimension
        n_tiles_y = int(np.ceil((volume_shape[1] - tile_size) / step_size)) + 1
        n_tiles_x = int(np.ceil((volume_shape[2] - tile_size) / step_size)) + 1
        
        print(f"Will use {n_tiles_y}×{n_tiles_x} tiles per slice")
        
        # Prepare normalization function
        def normalize_tile(tile):
            tile = tile.astype(np.float32)
            min_val = np.min(tile)
            max_val = np.max(tile)
            if max_val > min_val:
                tile = (tile - min_val) / (max_val - min_val)
            else:
                tile = np.zeros_like(tile)
            return tile
        
        # Process each slice
        with tifffile.TiffFile(input_path) as tif:
            for z in tqdm(range(volume_shape[0]), desc="Processing slices"):
                # Load one slice
                slice_data = tif.asarray(key=z)
                
                # Initialize slice prediction and weight maps
                slice_pred = np.zeros((volume_shape[1], volume_shape[2]), dtype=np.float32)
                weight_map = np.zeros((volume_shape[1], volume_shape[2]), dtype=np.float32)
                
                # Process each tile
                tile_batch = []
                tile_positions = []
                
                # Build batches of tiles
                for y in range(0, volume_shape[1], step_size):
                    if y + tile_size > volume_shape[1]:
                        y = max(0, volume_shape[1] - tile_size)
                        
                    for x in range(0, volume_shape[2], step_size):
                        if x + tile_size > volume_shape[2]:
                            x = max(0, volume_shape[2] - tile_size)
                        
                        # Extract and normalize the tile
                        tile = slice_data[y:y+tile_size, x:x+tile_size]
                        tile = normalize_tile(tile)
                        
                        # Add channel dimension
                        tile = np.expand_dims(tile, axis=-1)
                        
                        # Add to batch
                        tile_batch.append(tile)
                        tile_positions.append((y, x))
                        
                        # Process batch if it reaches batch_size
                        if len(tile_batch) == batch_size:
                            # Convert batch to tensor
                            batch_tensor = tf.convert_to_tensor(np.array(tile_batch), dtype=tf.float32)
                            
                            # Predict
                            predictions = model(batch_tensor, training=False).numpy()
                            
                            # Add predictions to slice prediction
                            for i, (y_pos, x_pos) in enumerate(tile_positions):
                                # Create weight mask (higher in center, lower at edges)
                                y_grid, x_grid = np.mgrid[0:tile_size, 0:tile_size]
                                y_center = tile_size // 2
                                x_center = tile_size // 2
                                dist_from_center = np.sqrt((y_grid - y_center)**2 + (x_grid - x_center)**2)
                                weight = np.clip(1.0 - dist_from_center / (tile_size // 2), 0.2, 1.0)
                                
                                # Add prediction with weight
                                pred = predictions[i].squeeze()
                                slice_pred[y_pos:y_pos+tile_size, x_pos:x_pos+tile_size] += pred * weight
                                weight_map[y_pos:y_pos+tile_size, x_pos:x_pos+tile_size] += weight
                            
                            # Clear batch
                            tile_batch = []
                            tile_positions = []
                
                # Process remaining tiles
                if tile_batch:
                    batch_tensor = tf.convert_to_tensor(np.array(tile_batch), dtype=tf.float32)
                    predictions = model(batch_tensor, training=False).numpy()
                    
                    for i, (y_pos, x_pos) in enumerate(tile_positions):
                        # Create weight mask
                        y_grid, x_grid = np.mgrid[0:tile_size, 0:tile_size]
                        y_center = tile_size // 2
                        x_center = tile_size // 2
                        dist_from_center = np.sqrt((y_grid - y_center)**2 + (x_grid - x_center)**2)
                        weight = np.clip(1.0 - dist_from_center / (tile_size // 2), 0.2, 1.0)
                        
                        # Add prediction with weight
                        pred = predictions[i].squeeze()
                        slice_pred[y_pos:y_pos+tile_size, x_pos:x_pos+tile_size] += pred * weight
                        weight_map[y_pos:y_pos+tile_size, x_pos:x_pos+tile_size] += weight
                
                # Average by weights
                weight_map[weight_map == 0] = 1.0  # Avoid division by zero
                slice_pred = slice_pred / weight_map
                
                # Apply threshold and store in segmentation volume
                binary_slice = (slice_pred > threshold).astype(np.uint8)
                segmentation[z] = binary_slice
                
                # Force synchronization of memory map
                if z % 10 == 0:
                    segmentation.flush()
                
                # Clear memory
                del slice_data, slice_pred, weight_map
                gc.collect()
    
    # Save segmentation result
    print(f"Saving segmentation to: {output_path}")
    segmentation.flush()  # Ensure all data is written to disk
    
    # Convert memory map to array for saving
    # Use memory-efficient approach by processing in chunks
    print(f"Converting result to final output format...")
    
    # If the volume is very large, write directly from memmap to tiff
    if volume_shape[0] * volume_shape[1] * volume_shape[2] > 800 * 1024 * 1024:  # >800M voxels
        print("Volume is very large, writing directly from memmap...")
        
        # For TIFF output, write slices in chunks
        with tifffile.TiffWriter(output_path) as tif:
            chunk_size = 32  # Number of slices to process at once
            for start_z in range(0, volume_shape[0], chunk_size):
                end_z = min(start_z + chunk_size, volume_shape[0])
                chunk = np.array(segmentation[start_z:end_z])  # Copy from memmap to array
                tif.write(chunk, contiguous=True)
                print(f"  Wrote slices {start_z} to {end_z-1}")
                del chunk
                gc.collect()
    else:
        # For smaller volumes, convert whole memmap to array first
        print("Converting entire memmap to array...")
        segmentation_array = np.array(segmentation)  # Copy from memmap to array
        tifffile.imwrite(output_path, segmentation_array)
        
        # Clean up the memmap
        del segmentation_array
        gc.collect()
    
    # Process additional export formats if requested
    if export_formats:
        print(f"Creating additional export formats: {export_formats}")
        
        # Export to NPY format
        if 'npy' in export_formats:
            npy_path = os.path.join(export_dir, "segmentation.npy")
            print(f"  Saving NPY file to: {npy_path}")
            
            # For large volumes, we need to do this in chunks
            if volume_shape[0] * volume_shape[1] * volume_shape[2] > 800 * 1024 * 1024:
                # Save directly from memmap
                np.save(npy_path, segmentation)
            else:
                # Load from TIFF and save as NPY
                seg_array = tifffile.imread(output_path)
                np.save(npy_path, seg_array)
                del seg_array
                gc.collect()
        
        # Export slices as PNG images
        if 'png_slices' in export_formats:
            png_dir = os.path.join(export_dir, "slices")
            os.makedirs(png_dir, exist_ok=True)
            print(f"  Saving PNG slices to: {png_dir}")
            
            # For memory efficiency, load and save slices one by one
            import matplotlib.pyplot as plt
            
            # Determine step size based on volume depth
            step = max(1, volume_shape[0] // 200)  # Save at most ~200 slices
            
            for z in tqdm(range(0, volume_shape[0], step), desc="Saving PNG slices"):
                # Read the slice
                with tifffile.TiffFile(output_path) as tif:
                    slice_data = tif.asarray(key=z)
                
                # Save as PNG
                plt.figure(figsize=(10, 10))
                plt.imshow(slice_data, cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(png_dir, f"slice_{z:04d}.png"), dpi=100, bbox_inches='tight')
                plt.close()
        
        # Export a ZIP archive
        if 'zip' in export_formats:
            import zipfile
            
            # Create a ZIP file of the segmentation
            zip_path = os.path.join(os.path.dirname(output_path), 
                                  f"{os.path.splitext(os.path.basename(output_path))[0]}.zip")
            
            print(f"  Creating ZIP archive: {zip_path}")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Always include the TIFF
                zipf.write(output_path, os.path.basename(output_path))
                
                # Include any additional formats
                if 'npy' in export_formats:
                    npy_path = os.path.join(export_dir, "segmentation.npy")
                    if os.path.exists(npy_path):
                        zipf.write(npy_path, "segmentation.npy")
                
                # For PNG slices, only include a subset if there are many
                if 'png_slices' in export_formats:
                    png_dir = os.path.join(export_dir, "slices")
                    png_files = sorted(glob.glob(os.path.join(png_dir, "*.png")))
                    
                    if len(png_files) > 50:
                        # If more than 50 PNGs, select every Nth one
                        step = len(png_files) // 50 + 1
                        selected_pngs = png_files[::step]
                    else:
                        selected_pngs = png_files
                    
                    for png in selected_pngs:
                        zipf.write(png, os.path.join("slices", os.path.basename(png)))
            
            print(f"  ZIP archive created: {zip_path}")
    
    # Clean up temporary files
    try:
        if os.path.exists(mmap_file):
            del segmentation
            gc.collect()
            os.unlink(mmap_file)
            print(f"Removed temporary memmap file.")
    except Exception as e:
        print(f"Warning: Could not remove temporary file {mmap_file}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Segmentation completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Final {memory_stats()}")
    
    return output_path

def evaluate_segmentation(prediction_path, ground_truth_path):
    """
    Evaluate a segmentation against ground truth.
    
    Args:
        prediction_path: Path to predicted segmentation
        ground_truth_path: Path to ground truth segmentation
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating segmentation:")
    print(f"  Prediction: {prediction_path}")
    print(f"  Ground truth: {ground_truth_path}")
    
    # For large volumes, evaluate in chunks
    metrics = {}
    
    # Get the shapes first without loading data
    with tifffile.TiffFile(prediction_path) as tif:
        pred_shape = tif.series[0].shape
    
    with tifffile.TiffFile(ground_truth_path) as tif:
        gt_shape = tif.series[0].shape
    
    # Check if shapes match
    if pred_shape != gt_shape:
        print(f"Error: Prediction shape {pred_shape} doesn't match ground truth shape {gt_shape}")
        return metrics
    
    # Define chunk size based on volume size
    chunk_size = max(1, min(100, pred_shape[0] // 10))  # Process at most 1/10 of slices at a time
    
    # Initialize metrics
    total_dice = 0
    total_intersection = 0
    total_union = 0
    total_accuracy = 0
    total_gt_volume = 0
    total_pred_volume = 0
    
    # Process chunks
    for start_z in range(0, pred_shape[0], chunk_size):
        end_z = min(start_z + chunk_size, pred_shape[0])
        print(f"  Evaluating slices {start_z} to {end_z-1}...")
        
        # Load chunks
        with tifffile.TiffFile(prediction_path) as tif:
            pred_chunk = tif.asarray(key=range(start_z, end_z))
        
        with tifffile.TiffFile(ground_truth_path) as tif:
            gt_chunk = tif.asarray(key=range(start_z, end_z))
        
        # Make ground truth binary
        gt_chunk = (gt_chunk > 0).astype(np.uint8)
        
        # Calculate metrics for this chunk
        dice = dice_coefficient(gt_chunk, pred_chunk)
        intersection = np.sum(gt_chunk & pred_chunk)
        union = np.sum(gt_chunk | pred_chunk)
        accuracy = np.mean((gt_chunk == pred_chunk).astype(float))
        gt_volume = np.sum(gt_chunk)
        pred_volume = np.sum(pred_chunk)
        
        # Accumulate metrics
        total_dice += dice * (end_z - start_z)  # Weight by number of slices
        total_intersection += intersection
        total_union += union
        total_accuracy += accuracy * (end_z - start_z)  # Weight by number of slices
        total_gt_volume += gt_volume
        total_pred_volume += pred_volume
        
        # Clean up
        del pred_chunk, gt_chunk
        gc.collect()
    
    # Calculate final metrics
    metrics['dice'] = total_dice / pred_shape[0]  # Average Dice over all slices
    metrics['iou'] = total_intersection / (total_union + 1e-8)
    metrics['accuracy'] = total_accuracy / pred_shape[0]  # Average accuracy over all slices
    metrics['volume_ratio'] = total_pred_volume / (total_gt_volume + 1e-8)
    
    # Print metrics
    print(f"Evaluation results:")
    print(f"  Dice coefficient: {metrics['dice']:.4f}")
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Volume ratio (pred/gt): {metrics['volume_ratio']:.4f}")
    print(f"  Ground truth volume: {total_gt_volume} voxels")
    print(f"  Prediction volume: {total_pred_volume} voxels")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Segment blood vessels in 3D volume")
    parser.add_argument("--input", default=None, help="Path to input 3D TIFF file")
    parser.add_argument("--output", default=None, help="Path to save segmentation output")
    parser.add_argument("--model", default=None, help="Path to trained model weights file")
    parser.add_argument("--model_type", choices=["best_loss", "best_vessel", "final"], 
                       default="best_vessel", help="Type of model to use if explicit path not provided")
    parser.add_argument("--threshold", type=float, default=0.4, help="Segmentation threshold (default: 0.4)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--tile_size", type=int, default=256, 
                       help="Size of tiles for processing large images (default: 256)")
    parser.add_argument("--overlap", type=int, default=64, 
                       help="Overlap between tiles to prevent edge artifacts (default: 64)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate against ground truth")
    parser.add_argument("--ground-truth", default=None, help="Path to ground truth segmentation")
    parser.add_argument("--export", nargs='+', choices=['npy', 'png_slices', 'zip'], default=None,
                       help="Additional export formats (npy, png_slices, zip)")
    
    args = parser.parse_args()
    
    # Determine model path based on model_type if not explicitly provided
    if args.model is None:
        checkpoint_map = {
            "best_loss": "model_best.weights.h5",
            "best_vessel": "best_vessel_recall.weights.h5",
            "final": "final_model.weights.h5"
        }
        model_filename = checkpoint_map.get(args.model_type, "model_best.weights.h5")
        args.model = find_checkpoint_file(priority_files=[model_filename])
        
        if args.model is None:
            print(f"No model file found for type '{args.model_type}'. Searching for any available checkpoint...")
            args.model = find_checkpoint_file()
    
    if args.model is None:
        print("No model file found. Please specify a model file with --model.")
        return
    
    # If no input is provided, use the most recently modified raw file from the sample datasets
    if args.input is None:
        sample_files = []
        for dataset in SAMPLE_DATASETS:
            raw_file = os.path.join(DATA_DIR, dataset["raw_file"])
            if os.path.exists(raw_file):
                sample_files.append(raw_file)
        
        if sample_files:
            # Use most recently modified file
            args.input = max(sample_files, key=os.path.getmtime)
            print(f"No input file specified, using most recent sample: {args.input}")
        else:
            print("No input file specified and no sample files found.")
            return
    
    # Initialize model
    print(f"Initializing model...")
    print(f"Initial {memory_stats()}")
    model = get_model(in_channels=1, num_classes=NUM_CLASSES)
    
    # Load weights
    print(f"Loading model weights from: {args.model}")
    if os.path.exists(args.model):
        try:
            model.load_weights(args.model)
            print(f"Successfully loaded weights.")
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            print("Continuing with random initialization")
    else:
        print(f"Model weights file not found: {args.model}")
        print("Continuing with random initialization")
    
    # For large volumes, use tiled segmentation
    print(f"Starting segmentation with threshold: {args.threshold}")
    print(f"Using tile size: {args.tile_size}, overlap: {args.overlap}, batch size: {args.batch_size}")
    output_path = segment_volume_tiled(
        model, 
        args.input, 
        args.output,
        threshold=args.threshold,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        export_formats=args.export
    )
    
    # Evaluate if requested
    if args.evaluate:
        ground_truth_path = args.ground_truth
        if ground_truth_path is None:
            # Try to find ground truth automatically
            base_name = os.path.basename(os.path.dirname(args.input))
            ground_truth_path = os.path.join(RESULTS_DIR, f"{base_name}.tif")
            if not os.path.exists(ground_truth_path):
                print(f"Ground truth not found at {ground_truth_path}")
                print("Skipping evaluation")
                return
        
        evaluate_segmentation(output_path, ground_truth_path)
    
    # Print final path for easy reference
    print(f"\nSegmentation completed")
    print(f"Output saved to: {output_path}")
    
    # If we exported additional formats, print those paths too
    if args.export:
        export_dir = os.path.join(os.path.dirname(output_path), 
                                 os.path.splitext(os.path.basename(output_path))[0])
        print("\nAdditional exports:")
        
        if 'npy' in args.export:
            npy_path = os.path.join(export_dir, "segmentation.npy")
            if os.path.exists(npy_path):
                print(f"  NPY: {npy_path}")
        
        if 'png_slices' in args.export:
            png_dir = os.path.join(export_dir, "slices")
            if os.path.exists(png_dir):
                print(f"  PNG slices: {png_dir}")
                
        if 'zip' in args.export:
            zip_path = os.path.join(os.path.dirname(output_path), 
                                  f"{os.path.splitext(os.path.basename(output_path))[0]}.zip")
            if os.path.exists(zip_path):
                print(f"  ZIP archive: {zip_path}")
                print(f"  ZIP file size: {os.path.getsize(zip_path)/1024/1024:.2f} MB")
                print(f"  (Use this ZIP file to transfer results to another computer)")

if __name__ == "__main__":
    main()
