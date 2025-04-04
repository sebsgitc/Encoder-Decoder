"""Script to identify and fix NaN values in the dataset."""

import os
import numpy as np
import tensorflow as tf
import tifffile
from tqdm import tqdm
import glob
from configuration import DATA_DIR, RESULTS_DIR
from nan_detector import check_dataset_for_nans, run_dataset_nan_diagnostics
from data_loader import prepare_data, find_data_pairs

def run_dataset_check():
    """Run a check on the dataset for NaN values."""
    print("Preparing dataset for NaN check...")
    train_dataset, val_dataset, _ = prepare_data(batch_size=4)
    
    print("\nChecking training dataset for NaNs...")
    train_stats = check_dataset_for_nans(train_dataset, num_batches=20)
    
    print("\nChecking validation dataset for NaNs...")
    val_stats = check_dataset_for_nans(val_dataset, num_batches=10)
    
    return train_stats, val_stats

def fix_nan_files():
    """Identify and fix files with NaN values."""
    print("Finding all image files with NaN problems...")
    
    # Get all data pairs
    pairs = find_data_pairs()
    
    # Create backup directory
    backup_dir = os.path.join(DATA_DIR, "backups_nan_fix")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create fixed files directory
    fixed_dir = os.path.join(DATA_DIR, "fixed_files")
    os.makedirs(fixed_dir, exist_ok=True)
    
    # Process each file
    for i, (img_path, mask_path) in enumerate(pairs):
        print(f"\nProcessing file {i+1}/{len(pairs)}: {os.path.basename(img_path)}")
        
        try:
            # Load the file
            with tifffile.TiffFile(img_path) as tif:
                img_data = tif.asarray()
            
            # Check for NaN and Inf values
            nan_count = np.sum(np.isnan(img_data))
            inf_count = np.sum(np.isinf(img_data))
            
            if nan_count > 0 or inf_count > 0:
                print(f"  Found {nan_count} NaN and {inf_count} Inf values. Fixing...")
                
                # Create backup of original file
                backup_path = os.path.join(backup_dir, os.path.basename(img_path))
                if not os.path.exists(backup_path):
                    print(f"  Creating backup at {backup_path}")
                    # Create backup with binary copy
                    import shutil
                    shutil.copy2(img_path, backup_path)
                
                # Fix NaN and Inf values
                fixed_data = np.nan_to_num(img_data, nan=0.0, posinf=np.max(img_data[~np.isnan(img_data) & ~np.isinf(img_data)]), neginf=0.0)
                
                # Create a completely fixed version
                fixed_path = os.path.join(fixed_dir, os.path.basename(img_path))
                print(f"  Saving fixed version to {fixed_path}")
                tifffile.imwrite(fixed_path, fixed_data)
                
                print(f"  File fixed. Original: {img_path}, Fixed: {fixed_path}")
            else:
                print(f"  No NaN or Inf values found in this file.")
        
        except Exception as e:
            print(f"  Error processing file {img_path}: {e}")
    
    print("\nFinished processing files.")
    print(f"Backups saved to: {backup_dir}")
    print(f"Fixed files saved to: {fixed_dir}")

if __name__ == "__main__":
    print("Starting NaN detection and fixing process...")
    
    # Run dataset check
    train_stats, val_stats = run_dataset_check()
    
    # If NaNs were found, offer to fix them
    if train_stats['nan_count'] > 0 or val_stats['nan_count'] > 0:
        print("\n\nNaN values were detected in the dataset.")
        print("Would you like to run the file fixing process? (y/n)")
        response = input("> ")
        
        if response.lower() == 'y':
            print("\nRunning file fixing process...")
            fix_nan_files()
        else:
            print("\nSkipping file fixing process.")
            
            # Run diagnostics instead
            pairs = find_data_pairs()
            if pairs:
                img_paths = [p[0] for p in pairs]
                mask_paths = [p[1] for p in pairs]
                slice_indices = list(range(0, 300, 5))  # Check every 5th slice up to 300
                
                print("\nRunning detailed diagnostics instead...")
                run_dataset_nan_diagnostics(img_paths, mask_paths, slice_indices)
    else:
        print("\nNo NaN values were detected in the sample batches. Your dataset appears to be clean!")
