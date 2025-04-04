"""Debug utilities for troubleshooting the vessel segmentation pipeline."""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from configuration import *
from data_loader import find_data_pairs, process_image_mask, data_augmentation, load_multi_axis_slice

def visualize_dataset_samples(dataset, num_samples=3, title="Dataset Samples"):
    """Visualize samples from a dataset to verify data loading and augmentation."""
    print(f"\nVisualizing {num_samples} samples from dataset...")
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples*4))
    
    if num_samples == 1:
        axes = [axes]
        
    sample_iter = iter(dataset)
    
    for i in range(num_samples):
        try:
            # Get next batch
            images, masks = next(sample_iter)
            
            # Get first sample from batch
            image = images[0].numpy()
            mask = masks[0].numpy()
            
            # Remove channel dimension for display if present
            if len(image.shape) > 2 and image.shape[-1] == 1:
                image = image[:, :, 0]
                
            # Display image
            axes[i][0].imshow(image, cmap='gray')
            axes[i][0].set_title(f"Image {i+1}")
            axes[i][0].axis('off')
            
            # Display mask
            axes[i][1].imshow(mask, cmap='binary')
            axes[i][1].set_title(f"Mask {i+1}")
            axes[i][1].axis('off')
            
            # Print shapes for debugging
            print(f"Sample {i+1} - Image shape: {image.shape}, Mask shape: {mask.shape}")
            print(f"Image min/max: {image.min():.4f}/{image.max():.4f}")
            print(f"Mask unique values: {np.unique(mask)}")
            
        except StopIteration:
            print(f"Only {i} samples available in dataset")
            break
        except Exception as e:
            print(f"Error visualizing sample {i+1}: {str(e)}")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs("debug", exist_ok=True)
    plt.savefig(f"debug/{title.lower().replace(' ', '_')}.png")
    plt.close()
    
    print(f"Visualization saved to debug/{title.lower().replace(' ', '_')}.png")

def test_data_pipeline():
    """Test the data loading and augmentation pipeline with a single pair."""
    print("\nTesting data pipeline...")
    
    # Get data pairs
    pairs = find_data_pairs()
    if not pairs:
        print("No data pairs found!")
        return
    
    # Use first pair for testing
    img_path, mask_path = pairs[0]
    print(f"Testing with pair:\n  Image: {img_path}\n  Mask: {mask_path}")
    
    # Test slice extraction
    slice_idx = 150  # Middle slice from SLICE_RANGE
    print(f"\nTesting slice extraction for index {slice_idx}...")
    
    try:
        image, mask = process_image_mask(img_path, mask_path, slice_idx)
        
        # Print shapes
        print(f"After process_image_mask:")
        print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"  Image min/max: {tf.reduce_min(image).numpy():.4f}/{tf.reduce_max(image).numpy():.4f}")
        print(f"  Mask unique values: {np.unique(mask.numpy())}")
        
        # Test data augmentation
        print("\nTesting data augmentation...")
        aug_image, aug_mask = data_augmentation(image, mask)
        
        print(f"After data_augmentation:")
        print(f"  Image shape: {aug_image.shape}, dtype: {aug_image.dtype}")
        print(f"  Mask shape: {aug_mask.shape}, dtype: {aug_mask.dtype}")
        print(f"  Image min/max: {tf.reduce_min(aug_image).numpy():.4f}/{tf.reduce_max(aug_image).numpy():.4f}")
        print(f"  Mask unique values: {np.unique(aug_mask.numpy())}")
        
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Original image and mask
        axes[0, 0].imshow(image.numpy()[:,:,0] if len(image.shape) > 2 else image.numpy(), cmap='gray')
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask.numpy(), cmap='binary')
        axes[0, 1].set_title("Original Mask")
        axes[0, 1].axis('off')
        
        # Augmented image and mask
        axes[1, 0].imshow(aug_image.numpy()[:,:,0] if len(aug_image.shape) > 2 else aug_image.numpy(), cmap='gray')
        axes[1, 0].set_title("Augmented Image")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(aug_mask.numpy(), cmap='binary')
        axes[1, 1].set_title("Augmented Mask")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        os.makedirs("debug", exist_ok=True)
        plt.savefig("debug/data_pipeline_test.png")
        plt.close()
        
        print("Visualization saved to debug/data_pipeline_test.png")
        
    except Exception as e:
        print(f"Error testing data pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

def verify_all_pairs():
    """Verify that all expected image-mask pairs are loaded correctly."""
    print("\nVerifying all image-mask pairs are loaded correctly...")
    
    # Get data pairs
    pairs = find_data_pairs()
    
    # Check for expected pairs
    expected_prefixes = [dataset["name"] for dataset in SAMPLE_DATASETS]
    found_prefixes = []
    
    for img_path, mask_path in pairs:
        # Extract directory name (which should contain the prefix)
        dir_name = os.path.basename(os.path.dirname(img_path))
        found_prefixes.append(dir_name)
        
        # Verify mask file exists and has content
        mask_exists = os.path.exists(mask_path)
        mask_size = os.path.getsize(mask_path) if mask_exists else 0
        
        print(f"Pair: {dir_name}")
        print(f"  - Image: {img_path} (exists: {os.path.exists(img_path)})")
        print(f"  - Mask: {mask_path} (exists: {mask_exists}, size: {mask_size} bytes)")
        
        # Test loading a slice to ensure data format is correct
        try:
            # Try loading the middle slice
            with tifffile.TiffFile(img_path) as tif:
                num_slices = tif.series[0].shape[0]
                middle_slice_idx = num_slices // 2
                
            image, mask = process_image_mask(img_path, mask_path, middle_slice_idx)
            print(f"  - Successfully loaded slice {middle_slice_idx}/{num_slices}")
            print(f"  - Image shape: {image.shape}, Mask shape: {mask.shape}")
            print(f"  - Mask contains {np.sum(mask > 0)} positive pixels")
        except Exception as e:
            print(f"  - Error loading slice: {str(e)}")
    
    # Check for missing pairs
    missing = [prefix for prefix in expected_prefixes if prefix not in found_prefixes]
    if missing:
        print(f"\nWARNING: Could not find pairs for: {', '.join(missing)}")
    else:
        print(f"\nAll {len(expected_prefixes)} expected pairs were found successfully!")
    
    return pairs

def verify_multi_dimensional_slicing():
    """Verify that multi-dimensional slicing is working correctly."""
    print("\nVerifying multi-dimensional slicing...")
    
    # Get data pairs
    pairs = find_data_pairs()
    if not pairs:
        print("No data pairs found!")
        return
    
    # Use first pair for testing
    img_path, mask_path = pairs[0]
    print(f"Testing with pair:\n  Image: {img_path}\n  Mask: {mask_path}")
    
    # Load volumes
    try:
        image = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)
        
        print(f"Volume dimensions: {image.shape}")
        z_slices, y_slices, x_slices = image.shape
        
        # Test slices from each axis
        axes = ['Z', 'Y', 'X']
        middle_indices = [z_slices//2, y_slices//2, x_slices//2]
        
        fig, axes_plot = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (axis_idx, middle_idx) in enumerate(zip(range(3), middle_indices)):
            # Extract slice using load_multi_axis_slice
            img_slice, mask_slice = load_multi_axis_slice(
                img_path, mask_path, middle_idx, axis_idx
            )
            
            # Display the slice
            axes_plot[i].imshow(img_slice[:, :, 0], cmap='gray')
            axes_plot[i].set_title(f"{axes[i]}-axis slice (idx={middle_idx})")
            axes_plot[i].axis('off')
            
            # Print stats
            print(f"{axes[i]}-axis slice (idx={middle_idx}):")
            print(f"  Image shape: {img_slice.shape}")
            print(f"  Mask shape: {mask_slice.shape}")
            print(f"  Image min/max: {np.min(img_slice):.4f}/{np.max(img_slice):.4f}")
            print(f"  Positive mask pixels: {np.sum(mask_slice > 0)}")
        
        plt.tight_layout()
        os.makedirs("debug", exist_ok=True)
        plt.savefig("debug/multi_dimensional_slicing.png")
        plt.close()
        
        print("Multi-dimensional slicing visualization saved to debug/multi_dimensional_slicing.png")
        
    except Exception as e:
        print(f"Error testing multi-dimensional slicing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configure TensorFlow logging
    tf.get_logger().setLevel('INFO')
    
    # Test data pipeline
    test_data_pipeline()
    
    # Verify all pairs
    verify_all_pairs()
    
    # Verify multi-dimensional slicing
    verify_multi_dimensional_slicing()
