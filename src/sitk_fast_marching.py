#!/usr/bin/env python3
"""
Optimized blood vessel segmentation using SimpleITK's Fast Marching implementation
"""
import os
import numpy as np
import SimpleITK as sitk
import tifffile as tiff
from glob import glob
import time
import argparse

def load_image(filepath):
    """Load image from file and convert to SimpleITK format"""
    print(f"Loading image from: {filepath}")
    
    # Load with tifffile
    image_np = tiff.imread(filepath)
    print(f"Loaded numpy array with shape: {image_np.shape}, dtype: {image_np.dtype}")
    
    # Convert to SimpleITK image
    image_sitk = sitk.GetImageFromArray(image_np)
    print(f"Converted to SimpleITK image with size: {image_sitk.GetSize()}")
    
    return image_np, image_sitk

def invert_for_dark_vessels(image_sitk):
    """Invert image for dark vessel segmentation"""
    print("Inverting image for dark vessel segmentation...")
    
    # Get image statistics
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image_sitk)
    max_val = stats.GetMaximum()
    
    # Invert image
    return sitk.InvertIntensity(image_sitk, maximum=max_val)

def enhance_contrast(image_sitk):
    """Enhance image contrast to improve vessel detection"""
    print("Enhancing image contrast...")
    
    # Normalize to [0,1]
    image_sitk = sitk.RescaleIntensity(image_sitk)
    
    # Apply sigmoid contrast enhancement
    return sitk.SigmoidImageFilter(image_sitk, alpha=3.0, beta=0.7, 
                                 outputMinimum=0.0, outputMaximum=1.0)

def segment_vessels_fast_marching(image_sitk, seed_points, output_path, threshold=0.5):
    """Segment blood vessels using Fast Marching"""
    print(f"Segmenting blood vessels using {len(seed_points)} seed points...")
    start_time = time.time()
    
    # Invert image for dark vessel segmentation
    inverted = invert_for_dark_vessels(image_sitk)
    
    # Enhance contrast
    enhanced = enhance_contrast(inverted)
    
    # Smooth image to reduce noise
    print("Applying anisotropic diffusion smoothing...")
    smoothed = sitk.CurvatureAnisotropicDiffusion(enhanced, 
                                                timeStep=0.0625,
                                                conductanceParameter=3.0,
                                                numberOfIterations=5)
    
    # Create speed image (high values inside vessels, low values at boundaries)
    print("Creating speed image...")
    speed_image = sitk.BoundedReciprocal(sitk.GradientMagnitudeRecursiveGaussian(smoothed, sigma=0.5))
    
    # Check SimpleITK FastMarching availability (2.0+ vs 1.x has different API)
    is_new_sitk = hasattr(sitk, 'FastMarchingBaseImageFilter')
    
    # Setup Fast Marching with compatibility handling
    print("Setting up Fast Marching algorithm...")
    if is_new_sitk:
        # SimpleITK 2.0+ API
        fast_marching = sitk.FastMarchingBaseImageFilter()
        
        # Convert seed points to SimpleITK format
        print("Converting seed points (z,y,x SimpleITK format)...")
        trial_points = []
        trial_values = []
        
        for i, (x, y, z) in enumerate(seed_points):
            # Convert to ITK index format (z,y,x) for SimpleITK
            point = [int(z), int(y), int(x)]
            trial_points.append(point)
            trial_values.append(0.0)  # Initial value of 0 for all seeds
            
        # Set trial points
        if trial_points:
            fast_marching.SetTrialPoints(trial_points, trial_values)
        
        # Set stopping criteria
        fast_marching.SetStoppingValue(1000.0)
        
        print(f"Running Fast Marching algorithm...")
        try:
            time_map = fast_marching.Execute(speed_image)
            
            # Threshold to get binary segmentation
            print(f"Applying threshold ({threshold}) to get final segmentation...")
            segmentation = sitk.BinaryThreshold(
                time_map,
                lowerThreshold=0.0,
                upperThreshold=threshold,
                insideValue=1,
                outsideValue=0
            )
        except Exception as e:
            print(f"Error running Fast Marching: {e}")
            raise  # Re-raise to go to fallback
        
    else:
        # SimpleITK 1.x API fallback
        print("Using older SimpleITK API for Fast Marching")
        fast_marching = sitk.FastMarchingImageFilter()
        
        # For older API, set seeds differently
        for i, (x, y, z) in enumerate(seed_points[:5]):  # Limit to first 5 seeds for older API
            if 0 <= z < image_sitk.GetSize()[2] and 0 <= y < image_sitk.GetSize()[1] and 0 <= x < image_sitk.GetSize()[0]:
                # Add seed points directly
                fast_marching.AddTrialPoint([int(z), int(y), int(x)])
        
        # Set stopping criteria
        fast_marching.SetStoppingValue(1000.0)
        
        # Execute Fast Marching
        try:
            time_map = fast_marching.Execute(speed_image)
            
            # Threshold to get binary segmentation
            print(f"Applying threshold ({threshold}) to get final segmentation...")
            segmentation = sitk.BinaryThreshold(
                time_map,
                lowerThreshold=0.0,
                upperThreshold=threshold,
                insideValue=1,
                outsideValue=0
            )
        except Exception as e:
            print(f"Error running Fast Marching: {e}")
            raise  # Re-raise to go to fallback
    
    try:
        # Post-processing to clean up segmentation
        print("Performing post-processing on segmentation...")
        
        # Remove small isolated components
        component_image = sitk.ConnectedComponent(segmentation)
        relabeled = sitk.RelabelComponent(component_image)
        
        # Keep only the largest N components
        largest_components = 5
        cleaned = sitk.BinaryThreshold(relabeled, lowerThreshold=1, 
                                      upperThreshold=largest_components)
        
        # Fill holes
        print("Filling holes in segmentation...")
        filled = sitk.BinaryMorphologicalClosing(cleaned, [1, 1, 1])
        
        # Convert to numpy array
        segmentation_np = sitk.GetArrayFromImage(filled)
        
    except Exception as e:
        print(f"Error during post-processing: {e}")
        print("Using unprocessed segmentation...")
        segmentation_np = sitk.GetArrayFromImage(segmentation).astype(np.uint8)
    
    # Save segmentation
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving segmentation to: {output_path}")
    tiff.imwrite(output_path, segmentation_np.astype(np.uint8) * 255)
    
    elapsed_time = time.time() - start_time
    print(f"Segmentation completed in {elapsed_time:.2f} seconds")
    print(f"Segmented {np.sum(segmentation_np)} voxels "
          f"({np.sum(segmentation_np)/np.size(segmentation_np)*100:.4f}% of volume)")
    
    return segmentation_np

def segment_with_region_growing(image_sitk, seed_points, output_path, lower_threshold=75, upper_threshold=255):
    """Alternative segmentation using Connected Threshold"""
    print(f"Segmenting using Connected Threshold with {len(seed_points)} seed points...")
    start_time = time.time()
    
    # Invert for dark vessels
    inverted = invert_for_dark_vessels(image_sitk)
    
    # Apply smoothing
    smoothed = sitk.CurvatureAnisotropicDiffusion(inverted, 
                                                timeStep=0.0625,
                                                conductanceParameter=3.0,
                                                numberOfIterations=5)
    
    # Connected Threshold segmentation
    seg = sitk.Image(smoothed.GetSize(), sitk.sitkUInt8)
    seg.CopyInformation(smoothed)
    
    # Convert seed points to ITK format
    itk_seeds = []
    for x, y, z in seed_points:
        itk_seeds.append((int(z), int(y), int(x)))  # z,y,x order for ITK
    
    # Apply Connected Threshold from each seed
    for i, seed in enumerate(itk_seeds):
        if i < 5:
            print(f"Processing seed {i+1}: {seed}")
        
        # Create a separate segmentation for this seed
        seed_seg = sitk.ConnectedThreshold(
            smoothed, 
            seedList=[seed],
            lower=lower_threshold,
            upper=upper_threshold,
            replaceValue=1
        )
        
        # Combine with main segmentation
        seg = sitk.Or(seg, seed_seg)
    
    # Post-processing
    print("Applying post-processing...")
    cleaned = sitk.BinaryMorphologicalOpening(seg, [1,1,1])
    
    # Convert to numpy
    segmentation_np = sitk.GetArrayFromImage(cleaned)
    
    # Save segmentation
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving segmentation to: {output_path}")
    tiff.imwrite(output_path, segmentation_np.astype(np.uint8) * 255)
    
    elapsed_time = time.time() - start_time
    print(f"Segmentation completed in {elapsed_time:.2f} seconds")
    print(f"Segmented {np.sum(segmentation_np)} voxels "
          f"({np.sum(segmentation_np)/np.size(segmentation_np)*100:.4f}% of volume)")
    
    return segmentation_np

def main():
    parser = argparse.ArgumentParser(description="Blood vessel segmentation using SimpleITK")
    parser.add_argument("--input", type=str, default="3d-stacks/r01_/r01_.8bit.tif", 
                        help="Path to input image")
    parser.add_argument("--output", type=str, default="output/sitk_segmentation/r01_.8bit.tif", 
                        help="Output path for segmentation")
    parser.add_argument("--threshold", type=float, default=200.0,
                        help="Threshold for Fast Marching")
    parser.add_argument("--method", type=str, choices=["fast_marching", "region_growing"],
                        default="region_growing", help="Segmentation method to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Define seed points
    seed_points_r01 = [
        (478, 323, 32), (372, 648, 45), (920, 600, 72),
        (420, 457, 24), (369, 326, 74), (753, 417, 124),
        (755, 607, 174), (887, 507, 224), (305, 195, 274),
        (574, 476, 324), (380, 625, 374), (313, 660, 424),
        (100, 512, 610), (512, 20, 730), (512, 200, 820), (512, 400, 940)
    ]
    
    # Load image
    try:
        image_np, image_sitk = load_image(args.input)
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Make sure the input file exists and is a valid TIFF image")
        return 1
    
    # Print SimpleITK version
    print(f"SimpleITK version: {sitk.Version().VersionString()}")
    
    # Choose segmentation method based on availability and user preference
    try:
        if args.method == "region_growing" or not hasattr(sitk, 'FastMarchingBaseImageFilter'):
            print("Using Region Growing segmentation")
            segmentation = segment_with_region_growing(
                image_sitk, seed_points_r01, args.output, 
                lower_threshold=50, upper_threshold=200
            )
        else:
            print("Using Fast Marching segmentation")
            segmentation = segment_vessels_fast_marching(
                image_sitk, seed_points_r01, args.output, args.threshold
            )
    except Exception as e:
        print(f"Segmentation failed: {e}")
        print("Falling back to region growing as last resort")
        try:
            segmentation = segment_with_region_growing(
                image_sitk, seed_points_r01, args.output, 
                lower_threshold=50, upper_threshold=200
            )
        except Exception as e2:
            print(f"Final fallback also failed: {e2}")
            print("Cannot complete segmentation")
            return 1
    
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
