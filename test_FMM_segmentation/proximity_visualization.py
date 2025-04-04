"""Utility for visualizing proximity-based loss weighting."""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

def visualize_distance_weights(mask, threshold_encourage=5, threshold_discourage=20, 
                              encourage_factor=0.8, discourage_factor=1.5, 
                              save_path="proximity_weights_visualization.png"):
    """
    Visualize the distance-based weight map for a given mask.
    
    Args:
        mask: Binary mask where 1 indicates annotated vessel pixels
        threshold_encourage: Distance threshold for encouraging pixels
        threshold_discourage: Distance threshold for discouraging pixels
        encourage_factor: Factor for pixels close to annotations (< 1)
        discourage_factor: Factor for pixels far from annotations (> 1)
        save_path: Path to save the visualization
        
    Returns:
        Path to saved visualization
    """
    # Calculate distance transform
    distances = distance_transform_edt(1 - mask)
    
    # Create weight map based on distances
    close_mask = distances < threshold_encourage
    far_mask = distances > threshold_discourage
    middle_mask = ~(close_mask | far_mask)  # Neither close nor far
    
    weight_map = np.ones_like(distances, dtype=np.float32)
    weight_map[close_mask] = encourage_factor
    weight_map[middle_mask] = 1.0
    weight_map[far_mask] = discourage_factor
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original mask
    axes[0, 0].imshow(mask, cmap='binary')
    axes[0, 0].set_title('Original Mask')
    axes[0, 0].axis('off')
    
    # Distance transform
    im = axes[0, 1].imshow(distances, cmap='plasma')
    axes[0, 1].set_title('Distance from Annotations')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Weight zones visualization
    zone_map = np.zeros_like(mask, dtype=np.float32)
    zone_map[close_mask] = 1  # Encourage zone
    zone_map[middle_mask] = 2  # Neutral zone
    zone_map[far_mask] = 3     # Discourage zone
    
    cmap = plt.cm.get_cmap('viridis', 3)
    im = axes[1, 0].imshow(zone_map, cmap=cmap)
    axes[1, 0].set_title('Weighting Zones')
    axes[1, 0].axis('off')
    cbar = plt.colorbar(im, ax=axes[1, 0], ticks=[1, 2, 3], fraction=0.046, pad=0.04)
    cbar.set_ticklabels(['Encourage', 'Neutral', 'Discourage'])
    
    # Final weight map
    im = axes[1, 1].imshow(weight_map, cmap='RdBu_r', vmin=encourage_factor, vmax=discourage_factor)
    axes[1, 1].set_title('Loss Weight Map')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return save_path

def analyze_dataset_proximity_weights(dataset, output_dir="proximity_analysis", samples=5):
    """
    Analyze and visualize proximity weighting for a dataset.
    
    Args:
        dataset: TensorFlow dataset
        output_dir: Directory to save visualizations
        samples: Number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing proximity weighting on {samples} dataset samples...")
    
    # Parameters
    threshold_encourage = 5
    threshold_discourage = 20
    encourage_factor = 0.8
    discourage_factor = 1.5
    
    sample_count = 0
    
    # Process batches
    for batch_idx, batch_data in enumerate(dataset):
        if len(batch_data) == 3:  # With boundary weights
            images, masks, weights = batch_data
        else:  # Regular data
            images, masks = batch_data
        
        # Process each sample in the batch
        for i in range(len(masks)):
            if sample_count >= samples:
                break
                
            image = images[i].numpy()
            mask = masks[i].numpy()
            
            # Visualize proximity weighting
            save_path = os.path.join(output_dir, f"proximity_weights_sample{sample_count}.png")
            visualize_distance_weights(
                mask, 
                threshold_encourage, 
                threshold_discourage, 
                encourage_factor, 
                discourage_factor,
                save_path
            )
            
            # Create model input/output visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(np.squeeze(image), cmap='gray')
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # Mask
            axes[1].imshow(mask, cmap='binary')
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            
            # Proximity weight map
            distances = distance_transform_edt(1 - mask)
            close_mask = distances < threshold_encourage
            far_mask = distances > threshold_discourage
            middle_mask = ~(close_mask | far_mask)
            
            weight_map = np.ones_like(distances, dtype=np.float32)
            weight_map[close_mask] = encourage_factor
            weight_map[middle_mask] = 1.0
            weight_map[far_mask] = discourage_factor
            
            im = axes[2].imshow(weight_map, cmap='RdBu_r', vmin=encourage_factor, vmax=discourage_factor)
            axes[2].set_title('Proximity Weight Map')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"input_output_sample{sample_count}.png"), dpi=200, bbox_inches='tight')
            plt.close()
            
            sample_count += 1
        
        if sample_count >= samples:
            break
    
    print(f"Saved {sample_count} proximity weight visualizations to {output_dir}")

def test_proximity_visualization():
    """Test the proximity visualization with a synthetic example."""
    # Create a sample mask
    mask = np.zeros((256, 256), dtype=np.int32)
    
    # Add some vessel-like structures
    mask[100:120, 80:200] = 1  # Horizontal vessel
    mask[50:180, 120:140] = 1  # Vertical vessel
    mask[30:50, 30:50] = 1     # Small vessel segment
    mask[200:250, 200:210] = 1 # Another small vessel
    
    # Visualize with different parameters
    os.makedirs("proximity_tests", exist_ok=True)
    
    # Default parameters
    visualize_distance_weights(
        mask, 
        threshold_encourage=5, 
        threshold_discourage=20, 
        encourage_factor=0.8, 
        discourage_factor=1.5, 
        save_path="proximity_tests/default_params.png"
    )
    
    # More aggressive parameters
    visualize_distance_weights(
        mask, 
        threshold_encourage=3, 
        threshold_discourage=15, 
        encourage_factor=0.5, 
        discourage_factor=2.0, 
        save_path="proximity_tests/aggressive_params.png"
    )
    
    # More conservative parameters
    visualize_distance_weights(
        mask, 
        threshold_encourage=10, 
        threshold_discourage=30, 
        encourage_factor=0.9, 
        discourage_factor=1.2, 
        save_path="proximity_tests/conservative_params.png"
    )
    
    print("Test visualizations created in 'proximity_tests' directory")

if __name__ == "__main__":
    test_proximity_visualization()
    print("Run analyze_dataset_proximity_weights() on your dataset to see the effect on real data")
