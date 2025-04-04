"""Utilities specifically designed to improve background segmentation accuracy."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from configuration import *

def analyze_background_errors(model, dataset, output_dir="debug/background_analysis"):
    """
    Analyze and visualize where background errors are occurring to help diagnose issues.
    
    Args:
        model: Trained model
        dataset: Validation dataset
        output_dir: Directory to save analysis visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Store error maps for analysis
    background_error_maps = []
    intensity_values = []
    
    # Process batches
    for batch_idx, (images, masks) in enumerate(tqdm(dataset, desc="Analyzing background errors")):
        # Get predictions
        outputs = model(images, training=False)
        predictions = tf.cast(outputs > 0.5, tf.int32).numpy()
        
        # For each sample in the batch
        for i in range(images.shape[0]):
            image = images[i].numpy()
            mask = masks[i].numpy()
            pred = predictions[i].squeeze()
            
            # Create error map (1 where prediction is wrong, 0 where correct)
            error_map = (mask != pred).astype(np.int32)
            
            # Separate background errors (false positives)
            bg_mask = (mask == 0).astype(np.int32)
            bg_error_map = error_map * bg_mask  # Background pixels that were misclassified
            
            if np.sum(bg_error_map) > 0:
                background_error_maps.append(bg_error_map)
                
                # Record intensity values at error locations
                flat_image = image.squeeze().flatten()
                flat_bg_error = bg_error_map.flatten()
                error_intensities = flat_image[flat_bg_error == 1]
                intensity_values.extend(error_intensities)
            
            # Save some visualizations
            if batch_idx < 5 and i == 0:  # Limit to a few examples
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(141)
                plt.imshow(image.squeeze(), cmap='gray')
                plt.title('Original Image')
                plt.axis('off')
                
                # Ground truth
                plt.subplot(142)
                plt.imshow(mask, cmap='binary')
                plt.title('Ground Truth')
                plt.axis('off')
                
                # Prediction
                plt.subplot(143)
                plt.imshow(pred, cmap='binary')
                plt.title('Prediction')
                plt.axis('off')
                
                # Background errors
                plt.subplot(144)
                plt.imshow(bg_error_map, cmap='hot')
                plt.title('Background Errors')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"bg_errors_batch{batch_idx}_sample{i}.png"))
                plt.close()
    
    # Analyze error intensity distribution
    if intensity_values:
        plt.figure(figsize=(10, 6))
        plt.hist(intensity_values, bins=50, alpha=0.7)
        plt.title('Intensity Distribution at Background Error Locations')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "bg_error_intensity_distribution.png"))
        plt.close()
    
    # Aggregate error maps for heatmap
    if background_error_maps:
        # Resize all maps to the same size
        target_shape = background_error_maps[0].shape
        resized_maps = []
        
        for error_map in background_error_maps:
            if error_map.shape != target_shape:
                # Resize to target shape
                error_map_tensor = tf.convert_to_tensor(np.expand_dims(error_map, -1), dtype=tf.float32)
                resized = tf.image.resize(error_map_tensor, target_shape, method='nearest')
                resized_maps.append(resized.numpy().squeeze())
            else:
                resized_maps.append(error_map)
        
        # Create aggregate heatmap
        aggregate_heatmap = np.mean(np.stack(resized_maps, axis=0), axis=0)
        
        # Visualize aggregate heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(aggregate_heatmap, cmap='hot')
        plt.title('Aggregate Background Error Heatmap')
        plt.colorbar(label='Error Frequency')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "aggregate_bg_error_heatmap.png"))
        plt.close()
    
    return {
        'num_samples_with_errors': len(background_error_maps),
        'avg_error_intensity': np.mean(intensity_values) if intensity_values else 0,
        'output_dir': output_dir
    }

def get_background_specific_augmentation_pipeline():
    """
    Create a data augmentation pipeline specifically designed to improve background segmentation.
    This can be used as an additional preprocessing step.
    """
    def background_augment(image, mask):
        # Convert to float32 if needed
        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.int32)
        
        # 1. Add subtle random noise to background regions
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.03)
        bg_mask = 1.0 - tf.cast(tf.expand_dims(mask, -1), tf.float32)  # Background mask
        noisy_image = image + noise * bg_mask
        image = tf.clip_by_value(noisy_image, 0.0, 1.0)
        
        # 2. Random subtle intensity shifts in background
        if tf.random.uniform(()) > 0.5:
            intensity_shift = tf.random.uniform((), -0.1, 0.1)
            shifted_bg = image + intensity_shift * bg_mask
            image = tf.clip_by_value(shifted_bg, 0.0, 1.0)
        
        # 3. Random local contrast in background
        if tf.random.uniform(()) > 0.7:
            # Create small random pattern and upsample
            pattern = tf.random.uniform([IMAGE_SIZE//16, IMAGE_SIZE//16, 1], 0.8, 1.2)
            pattern = tf.image.resize(pattern, [IMAGE_SIZE, IMAGE_SIZE])
            # Apply pattern to background
            image = image * (1.0 + (pattern - 1.0) * bg_mask * 0.5)
            image = tf.clip_by_value(image, 0.0, 1.0)
            
        return image, mask
    
    return background_augment

def apply_background_refinement(predictions, images, threshold=0.5, refinement_intensity=True):
    """
    Post-processing function to refine predictions based on background intensity.
    
    Args:
        predictions: Model predictions [batch, height, width]
        images: Input images [batch, height, width, channels]
        threshold: Classification threshold
        refinement_intensity: Whether to use intensity-based refinement
    
    Returns:
        Refined binary predictions
    """
    binary_preds = tf.cast(predictions > threshold, tf.int32)
    
    if refinement_intensity:
        # Convert images to right format
        images_processed = tf.squeeze(images, axis=-1) if images.shape[-1] == 1 else images
        
        # Define intensity thresholds for background (adjust as needed based on analysis)
        bg_intensity_high = 0.7  # Very bright pixels are likely background
        bg_intensity_low = 0.1   # Very dark pixels could be background or vessel
        
        # Create intensity-based refinement masks
        high_intensity_mask = tf.cast(images_processed > bg_intensity_high, tf.int32)
        
        # Refine predictions: high intensity areas are more likely to be background
        refined_preds = tf.where(
            high_intensity_mask > 0,
            tf.zeros_like(binary_preds),  # Set to background
            binary_preds                 # Keep original prediction
        )
        
        return refined_preds
    else:
        return binary_preds

if __name__ == "__main__":
    print("Background improvement utilities loaded.")
    print("Run analyze_background_errors() on your validation dataset to diagnose issues.")
    print("Apply background_augment() during training to improve background diversity.")
    print("Use apply_background_refinement() for post-processing predictions.")
