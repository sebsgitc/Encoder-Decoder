"""
Script to compare segmentation masks from FMM and ML model, calculating evaluation metrics
and visualizing the confusion matrix.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
import nibabel as nib
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from datetime import datetime
import argparse
import logging
import tensorflow as tf
from tensorflow.keras import backend as K

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('segmentation_comparison')

# Define the specific datasets to evaluate
SPECIFIC_DATASETS = ['r01_', 'r33_', 'r34_', 'rL1_', 'rL4_']

# Check for GPU availability at the start
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Try to allocate memory on GPUs as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Found {len(gpus)} GPU(s): {', '.join([gpu.name for gpu in gpus])}")
        # Log GPU memory info
        for i, gpu in enumerate(gpus):
            gpu_memory = tf.config.experimental.get_memory_info(f'GPU:{i}')
            if gpu_memory:
                logger.info(f"GPU:{i} Memory - Current: {gpu_memory['current']/1e9:.2f} GB")
    except RuntimeError as e:
        logger.warning(f"Error setting GPU memory growth: {e}")
else:
    logger.warning("No GPUs found. Running on CPU.")

def setup_directories():
    """Setup directories for input and output"""
    base_dir = '/home2/ad4631sv-s/TestSebbe/Encoder-Decoder'
    
    # FMM segmentation directory (may need to be adjusted based on actual structure)
    fmm_dir = os.path.join(base_dir, "results")
    
    # ML segmentation directory
    ml_dir = os.path.join(base_dir, "evaluation_results/orthogonal_20250521_151219/orthogonal_binary_20250521_151219")
    
    # Create output directory for metrics if it doesn't exist
    output_dir = os.path.join(base_dir, "evaluation_metrics/comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    
    return fmm_dir, ml_dir, output_dir

def find_matching_files(fmm_dir, ml_dir):
    """Find matching files between FMM and ML segmentations for specific datasets"""
    matching_pairs = []
    
    # Process only the specific datasets defined in SPECIFIC_DATASETS
    for dataset_name in SPECIFIC_DATASETS:
        logger.info(f"Looking for files for dataset: {dataset_name}")
        
        # Find ML file
        ml_pattern = os.path.join(ml_dir, f"{dataset_name}binary.nii.gz")
        ml_files = glob.glob(ml_pattern)
        if not ml_files:
            ml_pattern = os.path.join(ml_dir, f"{dataset_name}_binary.tif")
            ml_files = glob.glob(ml_pattern)
        
        # Find FMM file
        fmm_pattern = os.path.join(fmm_dir, f"vessel_segmentation_{dataset_name}*.tif")
        fmm_files = glob.glob(fmm_pattern)
        
        if not fmm_files:
            # Try more flexible pattern
            fmm_pattern = os.path.join(fmm_dir, f"*{dataset_name}*.tif")
            fmm_files = glob.glob(fmm_pattern)
            
            # Try recursive search if still not found
            if not fmm_files:
                fmm_pattern = os.path.join(fmm_dir, "**", f"*{dataset_name}*.tif")
                fmm_files = glob.glob(fmm_pattern, recursive=True)
        
        # Check if we found both files
        if ml_files and fmm_files:
            # Get the latest FMM file
            latest_fmm = sorted(fmm_files, key=os.path.getmtime, reverse=True)[0]
            
            # Add to matching pairs
            matching_pairs.append({
                'dataset_name': dataset_name,
                'ml_file': ml_files[0],
                'fmm_file': latest_fmm
            })
            logger.info(f"Found matching files for {dataset_name}:")
            logger.info(f"  ML: {ml_files[0]}")
            logger.info(f"  FMM: {latest_fmm}")
        else:
            if not ml_files:
                logger.warning(f"No ML file found for {dataset_name}")
            if not fmm_files:
                logger.warning(f"No FMM file found for {dataset_name}")
    
    return matching_pairs

def load_segmentation(file_path):
    """Load segmentation file (supports NIfTI and TIFF formats), optimized for GPU processing"""
    try:
        if file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
            # Load NIfTI file
            nifti_img = nib.load(file_path)
            data = nifti_img.get_fdata()
            
            # NIfTI typically uses RAS+ orientation while TIFF uses a different convention
            # In NIfTI: data[x,y,z] but TIFF is often data[z,y,x]
            logger.info(f"Original NIfTI shape: {data.shape}")
            
            # For NIfTI with true XYZ ordering, reshape to ZYX for consistency with our TIFF files
            data = np.transpose(data, (2, 1, 0))
            
            # Convert to binary (after reshaping to ensure we're working with the correct data)
            if np.max(data) > 1:
                data = (data > 0).astype(np.float32)
                
            logger.info(f"Transformed NIfTI shape to ZYX ordering: {data.shape}")
            return data
        else:
            # Load TIFF file - which we'll assume is already in ZYX ordering
            data = io.imread(file_path)
            
            # Convert to binary if needed
            if np.max(data) > 1:
                data = (data > 0).astype(np.float32)
                
            logger.info(f"Loaded TIFF file with shape: {data.shape}")
            return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

# Add a new function to check and fix orientation issues
def ensure_orientation_match(fmm_data, ml_data, dataset_name):
    """Ensure that FMM and ML data have matching orientations"""
    logger.info(f"Checking orientation for {dataset_name}: FMM shape {fmm_data.shape}, ML shape {ml_data.shape}")
    
    # If shapes don't match at all, try different transpositions
    if fmm_data.shape != ml_data.shape:
        logger.warning(f"Shape mismatch for {dataset_name}. Attempting to fix orientation.")
        
        # Try swapping axes to match dimensions
        # If it's a simple axis order issue, one of these permutations should work
        permutations = [
            (0, 1, 2),  # original
            (0, 2, 1),  # swap y and z
            (1, 0, 2),  # swap x and y
            (1, 2, 0),  # swap x with z, y with x
            (2, 0, 1),  # swap x with z, y with x
            (2, 1, 0),  # swap x with z
        ]
        
        matched = False
        for perm in permutations:
            # Try transposing
            transposed = np.transpose(ml_data, perm)
            if transposed.shape == fmm_data.shape:
                ml_data = transposed
                logger.info(f"Fixed shape mismatch with transpose {perm}")
                matched = True
                break
        
        if not matched:
            # If still no match, try with flips too
            for perm in permutations:
                transposed = np.transpose(ml_data, perm)
                for axes in [(0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]:
                    flipped = np.flip(transposed, axis=axes)
                    if flipped.shape == fmm_data.shape:
                        ml_data = flipped
                        logger.info(f"Fixed shape mismatch with transpose {perm} and flip axes {axes}")
                        matched = True
                        break
                if matched:
                    break
    
    # Check foreground/background balance to detect inverted masks
    if fmm_data.shape == ml_data.shape:
        # Count pixels
        fmm_ones = np.sum(fmm_data > 0.5)
        ml_ones = np.sum(ml_data > 0.5)
        fmm_ratio = fmm_ones / fmm_data.size
        ml_ratio = ml_ones / ml_data.size
        
        logger.info(f"{dataset_name} foreground ratios - FMM: {fmm_ratio:.6f}, ML: {ml_ratio:.6f}")
        
        # Check for extreme imbalance or potential inverted masks
        if ml_ratio > 0.9 and fmm_ratio < 0.3:
            logger.warning(f"ML mask appears inverted (ML: {ml_ratio:.2%}, FMM: {fmm_ratio:.2%}) - inverting.")
            ml_data = 1 - ml_data
        
        # If shapes match, check and optimize alignment
        # Calculate Dice to assess similarity, looking at center slice for speed
        if len(fmm_data.shape) == 3:
            z_mid = fmm_data.shape[0] // 2
            if ml_ratio > 0 and fmm_ratio > 0:  # Only if both have foreground pixels
                # Calculate initial dice
                fmm_slice = fmm_data[z_mid] > 0.5
                ml_slice = ml_data[z_mid] > 0.5
                intersection = np.logical_and(fmm_slice, ml_slice).sum()
                dice = 2 * intersection / (np.sum(fmm_slice) + np.sum(ml_slice) + 1e-6)
                logger.info(f"Initial Dice on center slice: {dice:.4f}")
                
                # If Dice is very low, try additional alignments to maximize it
                if dice < 0.2:
                    best_dice = dice
                    best_data = ml_data.copy()
                    
                    # Try 90-degree rotations on xy plane
                    for k in range(1, 4):  # 90, 180, 270 degrees
                        rotated = np.zeros_like(ml_data)
                        for i in range(ml_data.shape[0]):
                            rotated[i] = np.rot90(ml_data[i], k=k)
                        
                        # Check Dice on center slice
                        ml_rot_slice = rotated[z_mid] > 0.5
                        intersection = np.logical_and(fmm_slice, ml_rot_slice).sum()
                        new_dice = 2 * intersection / (np.sum(fmm_slice) + np.sum(ml_rot_slice) + 1e-6)
                        
                        if new_dice > best_dice:
                            best_dice = new_dice
                            best_data = rotated.copy()
                            logger.info(f"Improved Dice to {new_dice:.4f} with {k*90}° rotation")
                    
                    if best_dice > dice:
                        ml_data = best_data
                        logger.info(f"Applied best transformation with Dice {best_dice:.4f}")
    
    return fmm_data, ml_data

def calculate_metrics(gt_mask, pred_mask):
    """Calculate evaluation metrics between ground truth and prediction masks using TensorFlow/GPU with chunking"""
    
    # Ensure binary masks (in case of floating point values)
    gt_mask_binary = gt_mask > 0.5
    pred_mask_binary = pred_mask > 0.5
    
    # Quick sanity check for inverted labels by comparing foreground percentages
    gt_foreground_percent = np.mean(gt_mask_binary) * 100
    pred_foreground_percent = np.mean(pred_mask_binary) * 100
    
    logger.info(f"GT foreground percentage: {gt_foreground_percent:.2f}%")
    logger.info(f"Pred foreground percentage: {pred_foreground_percent:.2f}%")
    
    # If ML mask has much more foreground than GT, might be inverted
    if pred_foreground_percent > 80 and gt_foreground_percent < 20:
        logger.warning("Possible label inversion detected in prediction mask")
    
    # Get the total size to determine if we need chunking
    total_size = np.prod(gt_mask.shape)
    
    # If the data is too large (over 500 million elements), process in chunks
    if total_size > 500000000:  # Adjust this threshold based on GPU memory
        logger.info(f"Large volume detected ({total_size} elements). Processing in chunks...")
        
        # Initialize counters
        tp_total = 0
        fp_total = 0
        tn_total = 0
        fn_total = 0
        
        # Calculate the chunk size (adjust based on GPU memory)
        chunk_size = 268435456  # 4 chunks per volume (256MB)
        
        # Flatten arrays for chunking
        gt_flat_np = gt_mask_binary.flatten()
        pred_flat_np = pred_mask_binary.flatten()
        
        # Process data in chunks
        for i in range(0, total_size, chunk_size):
            end_idx = min(i + chunk_size, total_size)
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(total_size + chunk_size - 1)//chunk_size} ({i}-{end_idx})")
            
            # Get chunk data
            gt_chunk = gt_flat_np[i:end_idx]
            pred_chunk = pred_flat_np[i:end_idx]
            
            # Process on CPU to avoid potential GPU issues
            tp = np.sum(np.logical_and(gt_chunk, pred_chunk))
            fp = np.sum(np.logical_and(np.logical_not(gt_chunk), pred_chunk))
            tn = np.sum(np.logical_and(np.logical_not(gt_chunk), np.logical_not(pred_chunk)))
            fn = np.sum(np.logical_and(gt_chunk, np.logical_not(pred_chunk)))
            
            # Add to totals
            tp_total += int(tp)
            fp_total += int(fp)
            tn_total += int(tn)
            fn_total += int(fn)
        
        # Use the totals for metric calculation
        tp = tp_total
        fp = fp_total
        tn = tn_total
        fn = fn_total
        
    else:
        # For smaller volumes, process the entire volume at once
        # Process on CPU for consistency
        tp = np.sum(np.logical_and(gt_mask_binary, pred_mask_binary))
        fp = np.sum(np.logical_and(np.logical_not(gt_mask_binary), pred_mask_binary))
        tn = np.sum(np.logical_and(np.logical_not(gt_mask_binary), np.logical_not(pred_mask_binary)))
        fn = np.sum(np.logical_and(gt_mask_binary, np.logical_not(pred_mask_binary)))
    
    # Log raw counts for debugging
    logger.info(f"Raw counts: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    # Calculate metrics
    metrics = {}
    metrics['true_positive'] = int(tp)
    metrics['false_positive'] = int(fp)
    metrics['true_negative'] = int(tn)
    metrics['false_negative'] = int(fn)
    
    # Calculate derived metrics
    total = tp + fp + tn + fn
    metrics['accuracy'] = (tp + tn) / (total + 1e-10)
    metrics['precision'] = tp / (tp + fp + 1e-10)
    metrics['recall'] = tp / (tp + fn + 1e-10)
    metrics['f1_score'] = 2 * tp / (2 * tp + fp + fn + 1e-10)
    metrics['iou'] = tp / (tp + fp + fn + 1e-10)
    metrics['specificity'] = tn / (tn + fp + 1e-10)
    metrics['npv'] = tn / (tn + fn + 1e-10)
    
    return metrics

def plot_confusion_matrix(metrics, dataset_name, output_path):
    """Plot confusion matrix from metrics with normalization to handle class imbalance"""
    # Create the raw confusion matrix
    cm_raw = np.array([
        [metrics['true_negative'], metrics['false_positive']],
        [metrics['false_negative'], metrics['true_positive']]
    ])
    
    # Create normalized confusion matrix (by row)
    cm_normalized = np.zeros_like(cm_raw, dtype=float)
    for i in range(2):  # Normalize each row
        row_sum = cm_raw[i, 0] + cm_raw[i, 1]
        if row_sum > 0:
            cm_normalized[i, 0] = cm_raw[i, 0] / row_sum
            cm_normalized[i, 1] = cm_raw[i, 1] / row_sum
    
    # Create figure with TWO subplots (fixing the layout issue)
    plt.figure(figsize=(15, 6))
    
    # Fix: Use a 1x2 grid instead of having inconsistent number of subplots
    
    # Plot raw confusion matrix with formatted numbers (commas for thousands)
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    # Define a custom format function to add commas to large numbers
    fmt_d = lambda x: format(int(x), ',d')
    # Create formatted annotations
    formatted_annotations = [[fmt_d(val) for val in row] for row in cm_raw]
    
    sns.heatmap(cm_raw, annot=formatted_annotations, fmt="", cmap="Blues", 
                xticklabels=["Background", "Vessel"], 
                yticklabels=["Background", "Vessel"],
                annot_kws={"size": 10})
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.ylabel('FMM label')
    plt.xlabel('ML label')
    
    # Plot normalized confusion matrix
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    sns.heatmap(cm_normalized, annot=True, fmt=".3f", cmap="Blues", 
                xticklabels=["Background", "Vessel"], 
                yticklabels=["Background", "Vessel"])
    plt.title("Normalized Confusion Matrix")
    plt.ylabel('FMM label')
    plt.xlabel('ML label')
    
    # Add spacing between subplots to prevent crowding
    plt.subplots_adjust(wspace=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_metrics_comparison(all_metrics, output_path):
    """Plot comparison of metrics across all datasets"""
    # Create a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame(all_metrics).T
    
    # Drop the 'COMBINED' row if present as it's not a separate dataset
    if 'COMBINED' in metrics_df.index:
        metrics_df = metrics_df.drop('COMBINED')
    
    # Reset index to have a column for dataset names
    metrics_df = metrics_df.reset_index()
    metrics_df = metrics_df.rename(columns={"index": "dataset"})
    
    # Select only the metrics we want to plot
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "iou", "specificity", "npv"]
    
    # Melt the DataFrame for seaborn plotting (convert from wide to long format)
    metrics_melted = pd.melt(
        metrics_df, 
        id_vars=["dataset"],
        value_vars=metrics_to_plot,
        var_name="metric", 
        value_name="value"
    )
    
    # Create a figure with a suitable size
    plt.figure(figsize=(14, 8))
    
    # Create a custom boxplot with more control
    ax = plt.subplot(111)
    
    # Use standard boxplot with proper whiskers instead of seaborn boxplot
    boxplot = ax.boxplot(
        [metrics_melted[metrics_melted['metric'] == metric]['value'] for metric in metrics_to_plot],
        positions=range(len(metrics_to_plot)),
        patch_artist=True,
        widths=0.5,
        showfliers=False  # Don't show fliers (outliers) as they'll be shown by stripplot
    )
    
    # Color the boxes with the same palette as before
    colors = sns.color_palette("Set3", len(metrics_to_plot))
    for i, box in enumerate(boxplot['boxes']):
        box.set(facecolor=colors[i], alpha=0.8)
    
    # Add individual data points with jitter to avoid overplotting
    for i, metric in enumerate(metrics_to_plot):
        # Filter data for this metric
        metric_data = metrics_melted[metrics_melted['metric'] == metric]
        
        # Add jitter to x positions
        jitter = np.random.normal(0, 0.1, size=len(metric_data))
        x_jittered = np.full(len(metric_data), i) + jitter
        
        # Plot points
        ax.scatter(
            x_jittered, 
            metric_data['value'],
            s=60,               # Size of points
            color='black',      # Color of points
            alpha=0.7,          # Transparency
            zorder=3            # Make sure points are above boxplot
        )
        
        # Only add labels for specified metrics
        if metric in ["precision", "recall", "f1_score", "iou"]:
            # Sort values to avoid overlapping labels
            sorted_indices = np.argsort(metric_data['value'].values)
            label_y_offset = 0
            
            for idx in sorted_indices:
                dataset = metric_data.iloc[idx]['dataset']
                value = metric_data.iloc[idx]['value']
                
                # Alternate label placement based on position to avoid overlap
                if len(sorted_indices) > 1:
                    # Calculate vertical spacing between labels
                    spacing = 0.03
                    # Position for this label (alternate sides for dense regions)
                    x_offset = 0.12 if idx % 2 == 0 else -0.12
                    y_offset = label_y_offset
                    # Update vertical position for next label
                    label_y_offset += spacing
                    
                    # Text alignment based on which side the label is on
                    ha = 'left' if x_offset > 0 else 'right'
                    
                    # Place label with offset
                    ax.annotate(
                        dataset,
                        xy=(i + jitter[idx], value),
                        xytext=(x_offset * 100, y_offset * 100),  # Convert to points
                        textcoords='offset points',
                        ha=ha,
                        va='center',
                        fontsize=8,
                        alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
                    )
                else:
                    # If only one point, place label to the right
                    ax.annotate(
                        dataset,
                        xy=(i + jitter[idx], value),
                        xytext=(7, 0),
                        textcoords='offset points',
                        ha='left',
                        va='center',
                        fontsize=8,
                        alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
                    )
    
    # Set the tick labels for x-axis
    ax.set_xticks(range(len(metrics_to_plot)))
    ax.set_xticklabels(metrics_to_plot)
    
    # Customize the plot
    plt.title("Comparison of Segmentation Metrics Across Datasets", fontsize=14)
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.ylim(0, 1.05)  # Set y-axis limits from 0 to 1.05 to leave room for labels
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Make the axis labels and ticks larger
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Ensure plot is not cut off
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Also save a combined bar chart showing each dataset's metrics side by side
    plt.figure(figsize=(16, 10))
    
    # Pivot the data to get datasets as rows and metrics as columns
    pivot_df = metrics_df.copy()
    
    # Create a bar chart
    bar_width = 0.1
    x = np.arange(len(metrics_to_plot))
    
    # Plot each dataset as a group of bars
    for i, dataset in enumerate(pivot_df['dataset']):
        values = [pivot_df.loc[i, metric] for metric in metrics_to_plot]
        plt.bar(x + i*bar_width, values, width=bar_width, label=dataset)
    
    # Add labels and legend
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Comparison of Metrics by Dataset', fontsize=14)
    plt.xticks(x + bar_width * (len(pivot_df) / 2 - 0.5), metrics_to_plot, rotation=45)
    plt.legend(title='Dataset', loc='upper right')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    bar_path = output_path.replace('.png', '_bar.png')
    plt.savefig(bar_path, dpi=300)
    plt.close()

def visualize_comparison(fmm_data, ml_data, dataset_name, output_path):
    """Create visualization of segmentation differences with enhanced slice selection"""
    # Choose a slice where both segmentations have vessels for better visualization
    if len(fmm_data.shape) == 3:
        # Find slices with most foreground pixels
        fmm_sums = [np.sum(fmm_data[i]) for i in range(fmm_data.shape[0])]
        ml_sums = [np.sum(ml_data[i]) for i in range(ml_data.shape[0])]
        
        # Find slice indices with maximum overlap
        combined_sums = [(fmm_sums[i] + ml_sums[i])/2 for i in range(len(fmm_sums))]
        slice_idx = np.argmax(combined_sums)
        
        # If the best slice has very few pixels, use middle slice instead
        if combined_sums[slice_idx] < 100:
            slice_idx = fmm_data.shape[0] // 2
            
        fmm_slice = fmm_data[slice_idx]
        ml_slice = ml_data[slice_idx]
    else:
        # For 2D data
        fmm_slice = fmm_data
        ml_slice = ml_data
        slice_idx = 0
    
    # Create a difference map with three categories:
    # 1. Both agree (white)
    # 2. Only FMM (red) 
    # 3. Only ML (blue)
    agreement = np.logical_and(fmm_slice > 0.5, ml_slice > 0.5)
    fmm_only = np.logical_and(fmm_slice > 0.5, ml_slice <= 0.5)
    ml_only = np.logical_and(fmm_slice <= 0.5, ml_slice > 0.5)
    
    # Log metrics for this slice to help with debugging
    slice_tp = np.sum(agreement)
    slice_fp = np.sum(ml_only)
    slice_fn = np.sum(fmm_only)
    dice = 2 * slice_tp / (2 * slice_tp + slice_fp + slice_fn + 1e-6)
    logger.info(f"Slice {slice_idx} visualization - TP: {slice_tp}, FP: {slice_fp}, FN: {slice_fn}, Dice: {dice:.4f}")
    
    # Create RGB image for visualization
    comparison = np.zeros((*fmm_slice.shape, 3), dtype=np.uint8)
    comparison[agreement] = [255, 255, 255]     # Agreement in white
    comparison[fmm_only] = [255, 0, 0]          # FMM only in red
    comparison[ml_only] = [0, 0, 255]           # ML only in blue
    
    # Create figure with subplots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(fmm_slice, cmap='gray')
    plt.title(f'FMM Segmentation\n{dataset_name} - Slice {slice_idx if len(fmm_data.shape) == 3 else "2D"}')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(ml_slice, cmap='gray')
    plt.title(f'ML Segmentation\n{dataset_name} - Slice {slice_idx if len(ml_data.shape) == 3 else "2D"}')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(comparison)
    plt.title('Segmentation Comparison\nWhite: Agreement, Red: FMM only, Blue: ML only')
    plt.axis('off')
    
    # Create a custom legend for the comparison plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='gray', label='Both agree'),
        Patch(facecolor='red', edgecolor='gray', label='FMM only'),
        Patch(facecolor='blue', edgecolor='gray', label='ML only')
    ]
    plt.legend(handles=legend_elements, loc='lower right', framealpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def calculate_combined_metrics(all_data):
    """Calculate metrics across all volumes combined using chunked processing"""
    # Initialize counters
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    
    # Process each dataset separately to avoid memory issues
    for data in all_data:
        logger.info(f"Processing combined metrics for {data['name']}...")
        
        # Get ground truth and prediction data
        gt_mask = data['gt']
        pred_mask = data['pred']
        
        # Get total size of this dataset
        total_size = np.prod(gt_mask.shape)
        logger.info(f"Dataset size: {total_size} elements")
        
        # Use chunking for large datasets
        if total_size > 100000000:  # 100M elements threshold
            # Calculate chunk size (adjust based on GPU memory)
            chunk_size = 50000000  # 50M elements per chunk
            
            # Flatten arrays for chunking
            gt_flat = gt_mask.flatten()
            pred_flat = pred_mask.flatten()
            
            # Process data in chunks
            for i in range(0, total_size, chunk_size):
                end_idx = min(i + chunk_size, total_size)
                logger.info(f"  Processing chunk {i//chunk_size + 1}/{(total_size + chunk_size - 1)//chunk_size}")
                
                # Get chunk data
                gt_chunk = gt_flat[i:end_idx]
                pred_chunk = pred_flat[i:end_idx]
                
                # Process on CPU to avoid GPU memory issues
                tp = np.sum((gt_chunk == 1) & (pred_chunk == 1))
                fp = np.sum((gt_chunk == 0) & (pred_chunk == 1))
                tn = np.sum((gt_chunk == 0) & (pred_chunk == 0))
                fn = np.sum((gt_chunk == 1) & (pred_chunk == 0))
                
                # Add to totals
                total_tp += int(tp)
                total_fp += int(fp)
                total_tn += int(tn)
                total_fn += int(fn)
        else:
            # For smaller volumes, process on CPU directly
            gt_flat = gt_mask.flatten()
            pred_flat = pred_mask.flatten()
            
            tp = np.sum((gt_flat == 1) & (pred_flat == 1))
            fp = np.sum((gt_flat == 0) & (pred_flat == 1))
            tn = np.sum((gt_flat == 0) & (pred_flat == 0))
            fn = np.sum((gt_flat == 1) & (pred_flat == 0))
            
            # Add to totals
            total_tp += int(tp)
            total_fp += int(fp)
            total_tn += int(tn)
            total_fn += int(fn)
    
    # Calculate metrics from the totals
    logger.info("Computing final metrics from combined counts")
    metrics = {}
    metrics['true_positive'] = total_tp
    metrics['false_positive'] = total_fp
    metrics['true_negative'] = total_tn
    metrics['false_negative'] = total_fn
    
    # Calculate derived metrics
    total = total_tp + total_fp + total_fn + total_tn
    metrics['accuracy'] = (total_tp + total_tn) / (total + 1e-10)
    metrics['precision'] = total_tp / (total_tp + total_fp + 1e-10)
    metrics['recall'] = total_tp / (total_tp + total_fn + 1e-10)
    metrics['f1_score'] = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-10)
    metrics['iou'] = total_tp / (total_tp + total_fp + total_fn + 1e-10)
    metrics['specificity'] = total_tn / (total_tn + total_fp + 1e-10)
    metrics['npv'] = total_tn / (total_tn + total_fn + 1e-10)
    
    return metrics

def plot_combined_confusion_matrix(metrics, output_path):
    """Plot confusion matrix from combined metrics with normalization"""
    # Create the raw confusion matrix
    cm_raw = np.array([
        [metrics['true_negative'], metrics['false_positive']],
        [metrics['false_negative'], metrics['true_positive']]
    ])
    
    # Create normalized confusion matrix (by row)
    cm_normalized = np.zeros_like(cm_raw, dtype=float)
    for i in range(2):  # Normalize each row
        row_sum = cm_raw[i, 0] + cm_raw[i, 1]
        if row_sum > 0:
            cm_normalized[i, 0] = cm_raw[i, 0] / row_sum
            cm_normalized[i, 1] = cm_raw[i, 1] / row_sum
    
    # Also create column-normalized matrix to show precision
    cm_col_normalized = np.zeros_like(cm_raw, dtype=float)
    for j in range(2):  # Normalize each column
        col_sum = cm_raw[0, j] + cm_raw[1, j]
        if col_sum > 0:
            cm_col_normalized[0, j] = cm_raw[0, j] / col_sum
            cm_col_normalized[1, j] = cm_raw[1, j] / col_sum
    
    # Create a figure with three subplots
    plt.figure(figsize=(18, 6))
    
    # Define formatter for thousands
    fmt_d = lambda x: format(int(x), ',d')
    # Create formatted annotations
    formatted_annotations = [[fmt_d(val) for val in row] for row in cm_raw]
    
    # Plot raw counts with formatted numbers
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
    sns.heatmap(cm_raw, annot=formatted_annotations, fmt="", cmap="Blues", 
                xticklabels=["Background", "Vessel"], 
                yticklabels=["Background", "Vessel"],
                annot_kws={"size": 10})
    plt.title("Raw Counts - All Datasets")
    plt.ylabel('FMM label')
    plt.xlabel('ML label')
    
    # Plot row-normalized matrix (sensitivity/recall)
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
    sns.heatmap(cm_normalized, annot=True, fmt=".3f", cmap="Blues", 
                xticklabels=["Background", "Vessel"], 
                yticklabels=["Background", "Vessel"])
    plt.title("Row Normalized - Recall/Sensitivity")
    plt.ylabel('FMM label')
    plt.xlabel('ML label')
    
    # Plot column-normalized matrix (precision)
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
    sns.heatmap(cm_col_normalized, annot=True, fmt=".3f", cmap="Blues", 
                xticklabels=["Background", "Vessel"], 
                yticklabels=["Background", "Vessel"])
    plt.title("Column Normalized - Precision")
    plt.ylabel('FMM label')
    plt.xlabel('ML label')
    
    # Add spacing between subplots to prevent crowding
    plt.subplots_adjust(wspace=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Also save a separate file with just the normalized matrix for clearer viewing
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".3f", cmap="Blues", 
                xticklabels=["Background", "Vessel"], 
                yticklabels=["Background", "Vessel"])
    plt.title("Normalized Confusion Matrix - All Datasets")
    plt.ylabel('FMM label')
    plt.xlabel('ML label')
    plt.tight_layout()
    norm_path = output_path.replace(".png", "_normalized.png")
    plt.savefig(norm_path)
    plt.close()

def main():
    """Main function to compare segmentations and calculate metrics with GPU acceleration"""
    # Set up directories
    fmm_dir, ml_dir, output_dir = setup_directories()
    logger.info(f"FMM segmentations directory: {fmm_dir}")
    logger.info(f"ML segmentations directory: {ml_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find matching files for specific datasets
    matching_pairs = find_matching_files(fmm_dir, ml_dir)
    logger.info(f"Found {len(matching_pairs)} matching pairs out of {len(SPECIFIC_DATASETS)} requested datasets")
    
    if len(matching_pairs) == 0:
        logger.error("No matching pairs found. Exiting.")
        return
    
    # Process each pair
    all_metrics = {}
    all_data = []  # Store data for combined metrics calculation
    
    # Configure TensorFlow to use available GPUs
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"TensorFlow using {strategy.num_replicas_in_sync} device(s)")
    
    for pair in tqdm(matching_pairs, desc="Processing segmentation pairs"):
        dataset_name = pair['dataset_name']
        logger.info(f"Processing {dataset_name}")
        
        # Load segmentations
        fmm_data = load_segmentation(pair['fmm_file'])
        ml_data = load_segmentation(pair['ml_file'])
        
        if fmm_data is None or ml_data is None:
            logger.error(f"Failed to load segmentations for {dataset_name}")
            continue
        
        # Fix orientation issues
        fmm_data, ml_data = ensure_orientation_match(fmm_data, ml_data, dataset_name)
        
        # Check if shapes match after orientation fixing
        if fmm_data.shape != ml_data.shape:
            logger.warning(f"Shape mismatch still exists for {dataset_name}: FMM {fmm_data.shape}, ML {ml_data.shape}")
            continue
        
        # Calculate metrics - this is now using TensorFlow/GPU
        logger.info(f"Calculating metrics for {dataset_name} on GPU...")
        metrics = calculate_metrics(fmm_data, ml_data)
        all_metrics[dataset_name] = metrics
        
        # Store data for combined metrics
        all_data.append({
            'gt': fmm_data,
            'pred': ml_data,
            'name': dataset_name
        })
        
        # Plot confusion matrix
        cm_path = os.path.join(output_dir, f"{dataset_name}confusion_matrix.png")
        plot_confusion_matrix(metrics, dataset_name, cm_path)
        
        # Create visualization
        vis_path = os.path.join(output_dir, f"{dataset_name}comparison.png")
        visualize_comparison(fmm_data, ml_data, dataset_name, vis_path)
        
        # Log metrics
        logger.info(f"Metrics for {dataset_name}:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Clear GPU memory after each dataset
        tf.keras.backend.clear_session()
    
    # Calculate and save combined metrics
    if all_data:
        logger.info("Calculating combined metrics across all datasets...")
        combined_metrics = calculate_combined_metrics(all_data)
        all_metrics['COMBINED'] = combined_metrics
        
        # Log combined metrics
        logger.info("Combined metrics across all datasets:")
        for key, value in combined_metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Plot combined confusion matrix
        cm_path = os.path.join(output_dir, "combined_confusion_matrix.png")
        plot_combined_confusion_matrix(combined_metrics, cm_path)
    logger.info(f"Metrics for {dataset_name}:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Save metrics to CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics).T
        csv_path = os.path.join(output_dir, "segmentation_metrics.csv")
        metrics_df.to_csv(csv_path)
        logger.info(f"Metrics saved to {csv_path}")
        
        # Plot metrics comparison
        comparison_path = os.path.join(output_dir, "metrics_comparison.png")
        plot_metrics_comparison(all_metrics, comparison_path)
        
        # Generate summary HTML report
        html_report_path = os.path.join(output_dir, "evaluation_report.html")
        generate_html_report(all_metrics, matching_pairs, html_report_path)
        logger.info(f"HTML report saved to {html_report_path}")

def generate_html_report(all_metrics, matching_pairs, output_path):
    """Generate HTML report with all metrics and visualizations"""
    # Convert metrics to DataFrame for easier handling
    metrics_df = pd.DataFrame(all_metrics).T
    
    # Calculate average metrics (excluding combined)
    avg_metrics = metrics_df.drop('COMBINED', errors='ignore').mean()
    
    # Get combined metrics if available
    combined_metrics = all_metrics.get('COMBINED', {})
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Segmentation Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .header {{ background-color: #4CAF50; color: white; padding: 10px; margin-bottom: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .metric-highlight {{ font-weight: bold; color: #2196F3; }}
            img {{ max-width: 100%; margin: 10px 0; }}
            .comparison-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .metric-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
            .metric-card h4 {{ margin-top: 0; color: #333; }}
            .dataset-nav {{ position: sticky; top: 0; background-color: white; padding: 10px; border-bottom: 1px solid #ddd; z-index: 100; }}
            .dataset-nav a {{ margin-right: 15px; text-decoration: none; color: #2196F3; }}
            .dataset-nav a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Segmentation Evaluation Report</h1>
            <p>Comparison between FMM and ML segmentation methods</p>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="dataset-nav">
            <a href="#summary">Summary</a>
            {' '.join([f'<a href="#{name}">{name}</a>' for name in metrics_df.index if name != 'COMBINED'])}
        </div>
        
        <div id="summary" class="section">
            <h2>Summary</h2>
            <p>Total datasets evaluated: {len(all_metrics) - (1 if 'COMBINED' in all_metrics else 0)}</p>
            
            <h3>Combined Metrics (All Volumes)</h3>
    """
    
    # Add combined metrics if available
    if combined_metrics:
        html_content += f"""
            <table>
                <tr>
                    <th>Metric</th><th>Value</th>
                </tr>
                <tr>
                    <td>Accuracy</td><td class="metric-highlight">{combined_metrics['accuracy']:.4f}</td>
                </tr>
                <tr>
                    <td>Precision</td><td class="metric-highlight">{combined_metrics['precision']:.4f}</td>
                </tr>
                <tr>
                    <td>Recall (Sensitivity)</td><td class="metric-highlight">{combined_metrics['recall']:.4f}</td>
                </tr>
                <tr>
                    <td>F1 Score</td><td class="metric-highlight">{combined_metrics['f1_score']:.4f}</td>
                </tr>
                <tr>
                    <td>IoU (Jaccard Index)</td><td class="metric-highlight">{combined_metrics['iou']:.4f}</td>
                </tr>
                <tr>
                    <td>Specificity</td><td class="metric-highlight">{combined_metrics['specificity']:.4f}</td>
                </tr>
                <tr>
                    <td>NPV</td><td class="metric-highlight">{combined_metrics['npv']:.4f}</td>
                </tr>
            </table>
            
            <div class="metric-card">
                <h4>Confusion Matrix</h4>
                <img src="combined_confusion_matrix.png" alt="Combined Confusion Matrix">
                <img src="combined_confusion_matrix_normalized.png" alt="Normalized Combined Confusion Matrix">
            </div>
        """
    
    html_content += f"""
            <h3>Average Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th><th>Value</th>
                </tr>
                <tr>
                    <td>Accuracy</td><td class="metric-highlight">{avg_metrics['accuracy']:.4f}</td>
                </tr>
                <tr>
                    <td>Precision</td><td class="metric-highlight">{avg_metrics['precision']:.4f}</td>
                </tr>
                <tr>
                    <td>Recall (Sensitivity)</td><td class="metric-highlight">{avg_metrics['recall']:.4f}</td>
                </tr>
                <tr>
                    <td>F1 Score</td><td class="metric-highlight">{avg_metrics['f1_score']:.4f}</td>
                </tr>
                <tr>
                    <td>IoU (Jaccard Index)</td><td class="metric-highlight">{avg_metrics['iou']:.4f}</td>
                </tr>
                <tr>
                    <td>Specificity</td><td class="metric-highlight">{avg_metrics['specificity']:.4f}</td>
                </tr>
                <tr>
                    <td>NPV</td><td class="metric-highlight">{avg_metrics['npv']:.4f}</td>
                </tr>
            </table>
            
            <div class="metric-card">
                <h4>Metrics Comparison Across Datasets</h4>
                <img src="metrics_comparison.png" alt="Metrics Comparison">
            </div>
        </div>
        
        <div class="section">
            <h2>Dataset Details</h2>
    """
    
    # Add details for each dataset
    for dataset_name in metrics_df.index:
        if dataset_name == 'COMBINED':  # Skip combined metrics in the details section
            continue
        
        metrics = all_metrics[dataset_name]
        html_content += f"""
            <div id="{dataset_name}" class="section">
                <h3>Dataset: {dataset_name}</h3>
                <table>
                    <tr>
                        <th>Metric</th><th>Value</th>
                    </tr>
                    <tr>
                        <td>Accuracy</td><td>{metrics['accuracy']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Precision</td><td>{metrics['precision']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Recall (Sensitivity)</td><td>{metrics['recall']:.4f}</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td><td>{metrics['f1_score']:.4f}</td>
                    </tr>
                    <tr>
                        <td>IoU (Jaccard Index)</td><td>{metrics['iou']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Specificity</td><td>{metrics['specificity']:.4f}</td>
                    </tr>
                    <tr>
                        <td>NPV</td><td>{metrics['npv']:.4f}</td>
                    </tr>
                </table>
                
                <div class="comparison-grid">
                    <div class="metric-card">
                        <h4>Pixel Counts</h4>
                        <table>
                            <tr>
                                <th>Metric</th><th>Count</th>
                            </tr>
                            <tr>
                                <td>True Positives</td><td>{metrics['true_positive']:,}</td>
                            </tr>
                            <tr>
                                <td>False Positives</td><td>{metrics['false_positive']:,}</td>
                            </tr>
                            <tr>
                                <td>True Negatives</td><td>{metrics['true_negative']:,}</td>
                            </tr>
                            <tr>
                                <td>False Negatives</td><td>{metrics['false_negative']:,}</td>
                            </tr>
                        </table>
                    </div>
                
                    <div class="metric-card">
                        <h4>Confusion Matrix</h4>
                        <img src="{dataset_name}confusion_matrix.png" alt="Confusion Matrix">
                    </div>
                </div>
                
                <div class="metric-card">
                    <h4>Visual Comparison</h4>
                    <img src="{dataset_name}comparison.png" alt="Visual Comparison">
                    <p><strong>Color Legend:</strong> White = Both agree | Red = FMM only | Blue = ML only</p>
                </div>
            </div>
        """
    
    # Close HTML
    html_content += """
        </div>
        
        <script>
            // Add smooth scrolling for navigation links
            document.querySelectorAll('.dataset-nav a').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    
                    document.querySelector(this.getAttribute('href')).scrollIntoView({
                        behavior: 'smooth'
                    });
                });
            });
        </script>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated and saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compare FMM and ML segmentations')
    parser.add_argument('--fmm-dir', type=str, help='Directory containing FMM segmentations')
    parser.add_argument('--ml-dir', type=str, help='Directory containing ML segmentations')
    parser.add_argument('--output', type=str, help='Output directory for results')
    args = parser.parse_args()
    
    # Override default directories if provided
    if args.fmm_dir or args.ml_dir or args.output:
        base_dir = '/home2/ad4631sv-s/TestSebbe/Encoder-Decoder'
        fmm_dir = args.fmm_dir if args.fmm_dir else os.path.join(base_dir, "results")
        ml_dir = args.ml_dir if args.ml_dir else os.path.join(base_dir, "evaluation_results/orthogonal_20250521_151219/orthogonal_binary_20250521_151219")
        output_dir = args.output if args.output else os.path.join(base_dir, "evaluation_metrics/comparison_results")
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using custom directories: FMM={fmm_dir}, ML={ml_dir}, Output={output_dir}")
        
        # Create a matching_pairs list with the provided directories
        matching_pairs = find_matching_files(fmm_dir, ml_dir)
        logger.info(f"Found {len(matching_pairs)} matching pairs")
        
        # Process each pair
        all_metrics = {}
        all_data = []
        
        # Continue with processing (similar to main function)
        # ...
        
        # Alternatively, we can call a function to process the data
        # process_data(matching_pairs, fmm_dir, ml_dir, output_dir)
    else:
        # Run the main function with default directories
        main()
