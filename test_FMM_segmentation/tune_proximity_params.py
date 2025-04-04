"""Script to tune proximity weighting parameters for segmentation loss."""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import prepare_data
from proximity_visualization import visualize_distance_weights, analyze_dataset_proximity_weights
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import argparse

def analyze_vessel_distributions(dataset):
    """
    Analyze vessel size and spacing distributions in the dataset.
    
    Args:
        dataset: TensorFlow dataset
        
    Returns:
        Dictionary with statistics
    """
    # Store data for analysis
    vessel_sizes = []
    vessel_spacings = []
    
    print("Analyzing vessel distributions in dataset...")
    
    # Process a limited number of batches
    for batch_idx, batch_data in enumerate(tqdm(dataset.take(20))):
        if len(batch_data) == 3:  # With boundary weights
            images, masks, weights = batch_data
        else:  # Regular data
            images, masks = batch_data
        
        # Process each sample in the batch
        for i in range(len(masks)):
            mask = masks[i].numpy()
            
            # Skip empty masks
            if np.sum(mask) == 0:
                continue
            
            # Calculate distance transform for non-vessel regions
            bg_distances = distance_transform_edt(1 - mask)
            
            # Calculate distance transform for vessel regions (to get vessel sizes)
            vessel_distances = distance_transform_edt(mask)
            
            # Get vessel size approximations (max distance within vessel is ~radius)
            vessel_mask = mask > 0
            if np.any(vessel_mask):
                sizes = vessel_distances[vessel_mask]
                vessel_sizes.extend(sizes.flatten())
            
            # Get vessel spacing approximations
            # (focus on areas not too far from vessels)
            spacing_mask = (bg_distances > 0) & (bg_distances < 50)
            if np.any(spacing_mask):
                spacings = bg_distances[spacing_mask]
                vessel_spacings.extend(spacings.flatten())
    
    # Calculate statistics
    stats = {}
    if vessel_sizes:
        vessel_sizes = np.array(vessel_sizes)
        stats['vessel_size_mean'] = np.mean(vessel_sizes)
        stats['vessel_size_median'] = np.median(vessel_sizes)
        stats['vessel_size_p90'] = np.percentile(vessel_sizes, 90)
        stats['vessel_size_max'] = np.max(vessel_sizes)
    
    if vessel_spacings:
        vessel_spacings = np.array(vessel_spacings)
        stats['vessel_spacing_mean'] = np.mean(vessel_spacings)
        stats['vessel_spacing_median'] = np.median(vessel_spacings)
        stats['vessel_spacing_p90'] = np.percentile(vessel_spacings, 90)
    
    # Print statistics
    print("\nVessel Distribution Statistics:")
    print(f"  Vessel size (approximate radius):")
    print(f"    Mean: {stats.get('vessel_size_mean', 0):.2f} pixels")
    print(f"    Median: {stats.get('vessel_size_median', 0):.2f} pixels")
    print(f"    90th percentile: {stats.get('vessel_size_p90', 0):.2f} pixels")
    print(f"    Maximum: {stats.get('vessel_size_max', 0):.2f} pixels")
    print(f"  Vessel spacing:")
    print(f"    Mean: {stats.get('vessel_spacing_mean', 0):.2f} pixels")
    print(f"    Median: {stats.get('vessel_spacing_median', 0):.2f} pixels")
    print(f"    90th percentile: {stats.get('vessel_spacing_p90', 0):.2f} pixels")
    
    # Recommend parameters based on the statistics
    if vessel_spacings and vessel_sizes:
        # Good threshold_encourage is roughly the median vessel size 
        # (to catch branching points and nearby vessels)
        recommended_encourage = max(3, int(stats['vessel_size_median'] * 1.5))
        
        # Good threshold_discourage is roughly median vessel spacing
        # (to avoid predictions far from annotations)
        recommended_discourage = max(10, int(stats['vessel_spacing_median']))
        
        print("\nRecommended Parameters:")
        print(f"  threshold_encourage: {recommended_encourage}")
        print(f"  threshold_discourage: {recommended_discourage}")
        
        stats['recommended_encourage'] = recommended_encourage
        stats['recommended_discourage'] = recommended_discourage
    
    # Create visualization for the distributions
    os.makedirs("vessel_stats", exist_ok=True)
    
    if vessel_sizes:
        plt.figure(figsize=(10, 6))
        plt.hist(vessel_sizes, bins=50, alpha=0.7)
        plt.title('Vessel Size Distribution')
        plt.xlabel('Approximate Vessel Radius (pixels)')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.savefig("vessel_stats/vessel_size_distribution.png")
        plt.close()
    
    if vessel_spacings:
        plt.figure(figsize=(10, 6))
        plt.hist(vessel_spacings, bins=50, alpha=0.7)
        plt.axvline(x=stats.get('recommended_encourage', 5), color='r', linestyle='--', 
                   label=f'Recommended threshold_encourage: {stats.get("recommended_encourage", 5)}')
        plt.axvline(x=stats.get('recommended_discourage', 20), color='g', linestyle='--',
                   label=f'Recommended threshold_discourage: {stats.get("recommended_discourage", 20)}')
        plt.title('Vessel Spacing Distribution')
        plt.xlabel('Distance from Vessels (pixels)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("vessel_stats/vessel_spacing_distribution.png")
        plt.close()
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Tune proximity weighting parameters for vessel segmentation')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze dataset without visualizing')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for dataset loading')
    parser.add_argument('--threshold_encourage', type=int, default=5, help='Distance threshold for encouraging predictions')
    parser.add_argument('--threshold_discourage', type=int, default=20, help='Distance threshold for discouraging predictions')
    parser.add_argument('--encourage_factor', type=float, default=0.8, help='Weight factor for pixels close to annotations')
    parser.add_argument('--discourage_factor', type=float, default=1.5, help='Weight factor for pixels far from annotations')
    args = parser.parse_args()
    
    print("Loading dataset...")
    train_dataset, val_dataset, _ = prepare_data(batch_size=args.batch_size)
    
    # Analyze vessel distributions if requested
    stats = analyze_vessel_distributions(train_dataset)
    
    if args.analyze_only:
        print("Analysis complete. Recommended parameters:")
        print(f"  --threshold_encourage {stats.get('recommended_encourage', 5)}")
        print(f"  --threshold_discourage {stats.get('recommended_discourage', 20)}")
        print(f"  --encourage_factor 0.8")
        print(f"  --discourage_factor 1.5")
        return
    
    # Visualize proximity weighting with provided parameters
    print(f"Visualizing proximity weighting with parameters:")
    print(f"  threshold_encourage: {args.threshold_encourage}")
    print(f"  threshold_discourage: {args.threshold_discourage}")
    print(f"  encourage_factor: {args.encourage_factor}")
    print(f"  discourage_factor: {args.discourage_factor}")
    
    analyze_dataset_proximity_weights(
        val_dataset, 
        output_dir="proximity_analysis_custom", 
        samples=10
    )
    
    print("Visualization complete. To use these parameters in the model, update the CombinedLoss initialization.")
    print("Here's an example of the parameters to use in your training script:")
    print(f"""
    loss_fn = CombinedLoss(
        class_weights=CLASS_WEIGHTS,
        use_focal=True,
        focal_gamma=FOCAL_LOSS_GAMMA,
        background_boost=BACKGROUND_FOCUS['background_boost_factor'],
        border_weight=BACKGROUND_FOCUS['border_weight_factor'],
        target_vessel_percentage=target_vessel_percentage,
        use_proximity_weighting=True,
        threshold_encourage={args.threshold_encourage},
        threshold_discourage={args.threshold_discourage},
        encourage_factor={args.encourage_factor},
        discourage_factor={args.discourage_factor}
    )
    """)

if __name__ == "__main__":
    main()
