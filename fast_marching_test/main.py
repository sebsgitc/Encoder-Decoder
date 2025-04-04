import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from fastmarching import constrained_fast_marching
from utils import load_image, save_result
from visualization import visualize_result
import datetime  # Add this import to get today's date

def load_seed_points(csv_file):
    """
    Load seed points from a CSV file.
    Expected format: CSV with columns where:
    - Column 1: Index (ignored)
    - Column 2: x coordinate
    - Column 3: y coordinate
    - Column 4: z coordinate
    
    Parameters:
        csv_file: Path to the CSV file
        
    Returns:
        List of tuples (z,y,x) with seed point coordinates
    """
    try:
        # Try to load with pandas
        df = pd.read_csv(csv_file)
        
        # Check number of columns
        if len(df.columns) < 4:
            print(f"Warning: CSV file has less than 4 columns. Expected format: [index, x, y, z]")
            print("Attempting to use available columns...")
        
        # Check column names to determine format
        col_names = df.columns.tolist()
        print(f"Found columns: {col_names}")
        
        # If the file has explicit 'x','y','z' headers, use those
        if 'x' in col_names and 'y' in col_names and 'z' in col_names:
            x_col = 'x'
            y_col = 'y'
            z_col = 'z'
            print("Using named columns: x, y, z")
        else:
            # Otherwise, use columns by position (index=0, x=1, y=2, z=3)
            if len(df.columns) >= 4:
                # Handle zero or one indexing
                if col_names[0].lower() in ['index', 'id', 'idx', '#']:
                    # First column is index, use columns 1,2,3 (0-indexed)
                    x_col = col_names[1]
                    y_col = col_names[2]
                    z_col = col_names[3]
                    print(f"Using positional columns: {x_col}, {y_col}, {z_col}")
                else:
                    # First column might be data, check
                    
                    # First column not numeric, use 1,2,3
                    x_col = col_names[1]
                    y_col = col_names[2]
                    z_col = col_names[3]
                    print(f"Using columns 2-4 as coordinates: {x_col}, {y_col}, {z_col}")
            else:
                # Not enough columns, use what we have
                available_cols = col_names
                print(f"Warning: Not enough columns. Using available columns: {available_cols}")
                x_col = available_cols[0]
                y_col = available_cols[min(1, len(available_cols)-1)]
                z_col = available_cols[min(2, len(available_cols)-1)]
        
        # Extract coordinates - convert to the expected (z,y,x) order
        try:
            # Convert to numeric values in case they're stored as strings
            x_values = pd.to_numeric(df[x_col], errors='coerce')
            y_values = pd.to_numeric(df[y_col], errors='coerce')
            z_values = pd.to_numeric(df[z_col], errors='coerce')
            
            # Drop any rows with NaN values
            valid_indices = (~x_values.isna()) & (~y_values.isna()) & (~z_values.isna())
            x_values = x_values[valid_indices]
            y_values = y_values[valid_indices]
            z_values = z_values[valid_indices]
            
            # Convert to list of tuples in (z,y,x) order for the segmentation algorithm
            seed_points = list(zip(z_values, y_values, x_values))
            
            if len(seed_points) == 0:
                raise ValueError("No valid seed points found after processing")
                
            return seed_points
            
        except Exception as coord_err:
            print(f"Error extracting coordinates: {coord_err}")
            raise
        
    except Exception as e:
        print(f"Error loading seed points from CSV: {e}")
        print("Using default seed points instead.")
        return [(100, 150, 150), (105, 155, 155)]  # Default fallback

def main():
    # Parameters
    file = "r04_"
    # Get today's date in YYYYMMDD format automatically instead of hardcoding it
    date = datetime.datetime.now().strftime("%Y%m%d")
    input_file = f"3d-stacks/{file}/{file}.8bit.tif"
    seed_file = f"seed_points/{file}seed_points.csv"
    output_dir = "results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print("Loading image...")
    image, spacing = load_image(input_file)
    
    # Load seed points from CSV
    print(f"Loading seed points from {seed_file}...")
    seed_points = load_seed_points(seed_file)
    print(f"Loaded {len(seed_points)} seed points")
    
    # Show a few seed points for verification
    if len(seed_points) > 0:
        print("Sample seed points:")
        for i, point in enumerate(seed_points[:5]):  # Show first 5 points
            print(f"  Point {i+1}: z={point[0]}, y={point[1]}, x={point[2]}")
    
    # Update the stopping_criteria in main.py
    stopping_criteria = {
        'threshold': 100,                  # Intensity threshold                             # Was set to 100
        'max_distance': 2500,               # Maximum geodesic distance                     # Was set to 500
        'use_vesselness': True,            # Use vessel enhancement
        'vesselness_weight': 0.9,          # Weight for vesselness in speed function        # Used to be 0.7
        'sigma': 1.0,                      # Scale for vesselness calculation
        'check_growth_rate': True,         # Enable growth rate checking
        'growth_rate_threshold': 2.0,      # Threshold for detecting leakage                # Used to be 2.0
        'z_dependent_constraints': False,   # Enable z-level adaptive constraints
        'xy_scale_base': 0.7,              # Base growth speed in xy plane                  # Used to be 0.7
        'xy_scale_z_factor': 0.1,         # Factor for z-dependent xy scale adjustment
        'max_xy_scale': 0.8,               # Maximum allowed xy scale
        'min_xy_scale': 0.2,               # Minimum allowed xy scale                       # Used to be 0.8
        'edge_weight': 0.6,                # Weight for luminosity edge constraints
        'time_threshold': 50,              # Time threshold for stopping criteria          # Used to be 50

        'min_vessel_radius_low_z': 10,      # Minimum vessel radius at low z-levels
        'max_vessel_radius_low_z': 100,      # Maximum vessel radius at low z-levels
        'min_vessel_radius_high_z': 16,     # Minimum vessel radius at high z-levels
        'max_vessel_radius_high_z': 1000,    # Maximum vessel radius at high z-levels
        'vessel_size_weight': 0.3          # Weight for vessel size vs distance constraints # Used to be 0.5
    }
    
    # Run segmentation
    print("Running constrained fast marching...")
    segmentation, distance_map = constrained_fast_marching(image, seed_points, stopping_criteria)
    
    # Scale segmentation to [0-255] range before saving if it's in [0-1]
    if np.max(segmentation) <= 1:
        print("Scaling segmentation from [0,1] to [0,255] range")
        segmentation = (segmentation * 255).astype(np.uint8)
    
    # Save results with automatically generated date
    print("Saving results...")
    save_result(segmentation, os.path.join(output_dir, f"vessel_segmentation_{file}{date}.tif"), spacing)
    
    # Visualize results
    print("Generating visualization...")
    visualize_result(image, segmentation, distance_map, os.path.join(output_dir, "visualization"))
    
    print("Segmentation complete!")

if __name__ == "__main__":
    main()