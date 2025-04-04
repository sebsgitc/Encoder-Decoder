"""Script for batch processing multiple 3D volumes with blood vessel segmentation."""

import os
import glob
import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime

from configuration import *
from model import get_model
from inference import segment_volume, evaluate_segmentation, process_multiple_volumes

def find_all_raw_volumes():
    """Find all raw .tif volumes in the data directory."""
    volumes = []
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        # Skip the root directory itself
        if dirpath == DATA_DIR:
            continue
            
        # Look for .tif files in this directory
        tif_files = [f for f in filenames if f.endswith('.tif') and 'mask' not in f]
        
        for tif_file in tif_files:
            full_path = os.path.join(dirpath, tif_file)
            volumes.append(full_path)
    
    return volumes

def main():
    parser = argparse.ArgumentParser(description="Batch process multiple 3D volumes for vessel segmentation")
    parser.add_argument("--model", default="checkpoints/model_best.h5", help="Path to trained model weights")
    parser.add_argument("--all", action="store_true", help="Process all volumes in data directory")
    parser.add_argument("--output-dir", default=None, help="Directory to save segmentation results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--slice-range", nargs=2, type=int, default=None, 
                        help="Optional slice range to process (start end)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate against ground truth if available")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(RESULTS_DIR, f"batch_segmentation_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")
    
    # Initialize model
    print("Initializing model...")
    model = get_model(in_channels=1, num_classes=NUM_CLASSES)
    
    # Load weights
    if os.path.exists(args.model):
        print(f"Loading model weights from: {args.model}")
        model.load_weights(args.model)
    else:
        print(f"Model weights not found at: {args.model}")
        print("Exiting.")
        return
    
    # Get input volumes
    if args.all:
        # Find all volumes in data directory
        input_volumes = find_all_raw_volumes()
        print(f"Found {len(input_volumes)} volumes to process")
    else:
        # Use the sample datasets
        input_volumes = [os.path.join(DATA_DIR, dataset["raw_file"]) for dataset in SAMPLE_DATASETS]
        input_volumes = [v for v in input_volumes if os.path.exists(v)]
        print(f"Using {len(input_volumes)} sample volumes")
    
    if not input_volumes:
        print("No volumes found to process. Exiting.")
        return
    
    # Print the volumes to be processed
    print("Volumes to process:")
    for i, vol in enumerate(input_volumes):
        print(f"  {i+1}. {vol}")
    
    # Process all volumes
    results = []
    for i, input_path in enumerate(input_volumes):
        print(f"\nProcessing volume {i+1}/{len(input_volumes)}: {input_path}")
        
        # Create output path
        dirname = os.path.basename(os.path.dirname(input_path))
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, f"{dirname}_{filename}_segmented.tif")
        
        try:
            # Segment volume
            output_path = segment_volume(
                model,
                input_path,
                output_path,
                threshold=args.threshold,
                batch_size=args.batch_size,
                slice_range=args.slice_range
            )
            
            results.append((input_path, output_path))
            
            # Evaluate if requested
            if args.evaluate:
                # Try to find ground truth
                base_name = os.path.basename(os.path.dirname(input_path))
                mask_patterns = [
                    os.path.join(RESULTS_DIR, f"{base_name}.tif"),
                    os.path.join(RESULTS_DIR, f"vessel_segmentation_{base_name}*.tif")
                ]
                
                ground_truth_path = None
                for pattern in mask_patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        ground_truth_path = sorted(matches)[-1]
                        break
                
                if ground_truth_path:
                    print(f"\nEvaluating against ground truth: {ground_truth_path}")
                    evaluate_segmentation(output_path, ground_truth_path)
                else:
                    print(f"\nNo ground truth found for {base_name}")
            
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
    
    print(f"\nProcessing completed. {len(results)} volumes were successfully segmented.")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
