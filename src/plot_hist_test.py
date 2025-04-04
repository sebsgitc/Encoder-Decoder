def plot_histogram(input_path="3d-stacks/r01_/r01_.8bit.tif", output_path="output/hist_test.png"):
    """
    Plots and saves histogram of pixel values in a 3D TIFF stack.
    
    Args:
        input_path: Path to input TIFF file
        output_path: Path to save histogram plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile as tiff
    import os

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the image
    print(f"Loading image from {input_path}")
    image = tiff.imread(input_path)
    print(f"Image shape: {image.shape}")
    print(f"Value range: [{image.min()}, {image.max()}]")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(image.ravel(), bins=256, range=(0, 255), density=True, alpha=0.7, color='blue')
    
    # Add labels and title
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Image Values')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = f'Mean: {image.mean():.2f}\n'
    stats_text += f'Std: {image.std():.2f}\n'
    stats_text += f'Min: {image.min()}\n'
    stats_text += f'Max: {image.max()}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved histogram to {output_path}")
    
    # Close plot to free memory
    plt.close()

def plot_histograms(input_dir="3d-stacks", output_dir="output/histograms"):
    """
    Plots and saves histograms of pixel values for all TIFF files in input directory.
    
    Args:
        input_dir: Path to directory containing TIFF files
        output_dir: Path to save histogram plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile as tiff
    import os
    from glob import glob
    from tqdm import tqdm

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all TIFF files in input directory and its subdirectories
    tiff_files = []
    for root, dirs, files in os.walk(input_dir):
        tiff_files.extend(glob(os.path.join(root, "*.tif")))
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Process each file
    for file_path in tqdm(tiff_files, desc="Processing files"):
        try:
            # Load the image
            print(f"\nProcessing: {file_path}")
            image = tiff.imread(file_path)
            print(f"Image shape: {image.shape}")
            print(f"Value range: [{image.min()}, {image.max()}]")
            
            # Create histogram
            plt.figure(figsize=(12, 8))
            plt.hist(image.ravel(), bins=256, density=True, alpha=0.7, color='blue')
            
            # Add labels and title
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Image Values\n{os.path.basename(file_path)}')
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Add statistics as text
            stats_text = f'Shape: {image.shape}\n'
            stats_text += f'Mean: {image.mean():.2f}\n'
            stats_text += f'Std: {image.std():.2f}\n'
            stats_text += f'Min: {image.min()}\n'
            stats_text += f'Max: {image.max()}'
            plt.text(0.95, 0.95, stats_text,
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save plot
            output_path = os.path.join(output_dir, 
                                     f"histogram_{os.path.basename(file_path).replace('.tif', '.png')}")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved histogram to {output_path}")
            
            # Close plot to free memory
            plt.close()
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    # Create a summary plot of all histograms overlaid
    plt.figure(figsize=(15, 10))
    for file_path in tqdm(tiff_files, desc="Creating summary plot"):
        try:
            image = tiff.imread(file_path)
            plt.hist(image.ravel(), bins=256, density=True, alpha=0.3, 
                    label=os.path.basename(file_path))
        except Exception as e:
            print(f"Error including {file_path} in summary: {str(e)}")
            continue
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Combined Histogram of All Images')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save summary plot
    summary_path = os.path.join(output_dir, "histogram_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary histogram to {summary_path}")
    plt.close()

if __name__ == "__main__":
    plot_histograms()