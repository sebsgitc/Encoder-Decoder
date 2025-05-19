"""
NaN detection utility for TensorFlow datasets.
This module checks datasets for NaN or Inf values to prevent training issues.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

def check_dataset_for_nans(dataset, num_batches=None):
    """
    Check a TensorFlow dataset for NaN or Inf values.
    
    Args:
        dataset: TensorFlow dataset to check
        num_batches: Maximum number of batches to check (None for all)
    
    Returns:
        Dictionary with statistics about NaN/Inf values
    """
    nan_count = 0
    inf_count = 0
    batch_count = 0
    nan_batches = 0
    inf_batches = 0
    total_elements = 0
    
    print("Checking dataset for NaN/Inf values...")
    
    # Use tqdm for progress tracking
    dataset_iter = dataset
    if num_batches:
        dataset_iter = iter(dataset)
    
    for i, batch_data in enumerate(tqdm(dataset_iter, desc="Checking batches")):
        if num_batches and i >= num_batches:
            break
            
        batch_count += 1
        
        try:
            # Handle different batch formats - could be tuple, list, or tensor
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) >= 1:
                    images = batch_data[0]
                    
                    # Check dimensions before using is_nan/is_inf
                    # Count NaNs in images
                    batch_nan_count = tf.reduce_sum(
                        tf.cast(tf.math.is_nan(images), tf.int32)
                    ).numpy()
                    batch_inf_count = tf.reduce_sum(
                        tf.cast(tf.math.is_inf(images), tf.int32)
                    ).numpy()
                    
                    total_elements += tf.size(images).numpy()
                    
                    if batch_nan_count > 0:
                        nan_count += batch_nan_count
                        nan_batches += 1
                        
                    if batch_inf_count > 0:
                        inf_count += batch_inf_count
                        inf_batches += 1
                
                # Also check masks/labels if available
                if len(batch_data) >= 2:
                    masks = batch_data[1]
                    
                    # Check masks for NaNs/Infs
                    mask_nan_count = tf.reduce_sum(
                        tf.cast(tf.math.is_nan(masks), tf.int32)
                    ).numpy()
                    mask_inf_count = tf.reduce_sum(
                        tf.cast(tf.math.is_inf(masks), tf.int32)
                    ).numpy()
                    
                    total_elements += tf.size(masks).numpy()
                    nan_count += mask_nan_count
                    inf_count += mask_inf_count
                    
                    if mask_nan_count > 0 and nan_batches != batch_count:
                        nan_batches += 1
                    if mask_inf_count > 0 and inf_batches != batch_count:
                        inf_batches += 1
            else:
                # If batch_data is a single tensor (rare case)
                batch_nan_count = tf.reduce_sum(
                    tf.cast(tf.math.is_nan(batch_data), tf.int32)
                ).numpy()
                batch_inf_count = tf.reduce_sum(
                    tf.cast(tf.math.is_inf(batch_data), tf.int32)
                ).numpy()
                
                total_elements += tf.size(batch_data).numpy()
                
                if batch_nan_count > 0:
                    nan_count += batch_nan_count
                    nan_batches += 1
                    
                if batch_inf_count > 0:
                    inf_count += batch_inf_count
                    inf_batches += 1
        except Exception as e:
            print(f"Error checking batch {i}: {str(e)}")
            continue
    
    # Calculate statistics
    nan_percentage = (nan_count / max(1, total_elements)) * 100
    inf_percentage = (inf_count / max(1, total_elements)) * 100
    
    results = {
        'nan_count': nan_count,
        'inf_count': inf_count,
        'nan_batches': nan_batches,
        'inf_batches': inf_batches,
        'total_batches': batch_count,
        'total_elements': total_elements,
        'nan_percentage': nan_percentage,
        'inf_percentage': inf_percentage
    }
    
    # Print summary
    print(f"\nNaN detection summary:")
    print(f"- Found {nan_count} NaN values ({nan_percentage:.6f}% of elements)")
    print(f"- Found {inf_count} Inf values ({inf_percentage:.6f}% of elements)")
    print(f"- {nan_batches}/{batch_count} batches contain NaN values")
    print(f"- {inf_batches}/{batch_count} batches contain Inf values")
    
    return results

def fix_nans_in_dataset(dataset, output_path=None):
    """
    Create a copy of a dataset with NaN/Inf values replaced.
    NaN values are replaced with 0, Inf values with 1.
    
    Args:
        dataset: TensorFlow dataset to fix
        output_path: Path to save fixed dataset (optional)
    
    Returns:
        Fixed dataset with no NaN/Inf values
    """
    def fix_batch(batch_data):
        """Fix NaN/Inf values in a batch."""
        try:
            # Handle different batch formats
            if isinstance(batch_data, tuple) or isinstance(batch_data, list):
                fixed_batch = []
                for element in batch_data:
                    if element is not None:
                        # Replace NaNs with 0 and Infs with 1
                        fixed_element = tf.where(tf.math.is_nan(element), tf.zeros_like(element), element)
                        fixed_element = tf.where(tf.math.is_inf(fixed_element), tf.ones_like(fixed_element), fixed_element)
                        fixed_batch.append(fixed_element)
                    else:
                        fixed_batch.append(None)
                return tuple(fixed_batch)
            else:
                # Single tensor batch
                fixed_batch = tf.where(tf.math.is_nan(batch_data), tf.zeros_like(batch_data), batch_data)
                fixed_batch = tf.where(tf.math.is_inf(fixed_batch), tf.ones_like(fixed_batch), fixed_batch)
                return fixed_batch
        except Exception as e:
            print(f"Error fixing batch: {str(e)}")
            return batch_data  # Return original batch if fixing fails
    
    # Apply the fix_batch function to each batch
    fixed_dataset = dataset.map(fix_batch)
    
    return fixed_dataset