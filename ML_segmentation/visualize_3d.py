"""3D visualization utilities for vessel segmentation results."""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile
import argparse
from skimage import measure
import plotly.graph_objects as go
from configuration import *

def visualize_3d_volume(segmentation_path, output_dir=None, threshold=0.5, 
                       step_size=1, use_plotly=True, sample_points=50000):
    """
    Create 3D visualization of a segmented vessel network.
    
    Args:
        segmentation_path: Path to the segmentation TIFF file
        output_dir: Directory to save visualization output (if None, use segmentation directory)
        threshold: Threshold value for binary segmentation (default: 0.5)
        step_size: Step size for volume rendering (higher values = faster but coarser)
        use_plotly: Whether to use Plotly for interactive visualization (True) or Matplotlib (False)
        sample_points: Number of points to sample for plotly visualization (lower = faster)
    
    Returns:
        Path to the saved visualization
    """
    print(f"Loading segmentation from: {segmentation_path}")
    
    # Load the segmentation volume
    segmentation = tifffile.imread(segmentation_path)
    
    # Create binary segmentation
    binary_seg = segmentation > threshold
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.dirname(segmentation_path)
        output_dir = os.path.join(output_dir, "3d_visualization")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base filename for outputs
    base_name = os.path.splitext(os.path.basename(segmentation_path))[0]
    
    # Extract the volume dimensions
    z_dim, y_dim, x_dim = binary_seg.shape
    print(f"Segmentation shape: {binary_seg.shape}")
    print(f"Total segmented voxels: {np.sum(binary_seg)}")
    
    if use_plotly:
        # Use Plotly for interactive 3D visualization
        return visualize_with_plotly(binary_seg, output_dir, base_name, sample_points)
    else:
        # Use Matplotlib for static 3D visualization
        return visualize_with_matplotlib(binary_seg, output_dir, base_name, step_size)

def visualize_with_matplotlib(binary_seg, output_dir, base_name, step_size=1):
    """Create 3D visualization using Matplotlib (slower but more compatible)."""
    print("Creating 3D visualization with Matplotlib...")
    
    # Get the coordinates of the segmented voxels
    z, y, x = np.where(binary_seg)
    
    # Downsample for faster rendering if needed
    if step_size > 1:
        z = z[::step_size]
        y = y[::step_size]
        x = x[::step_size]
    
    # Create a new figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(x, y, z, c='r', marker='.', alpha=0.05)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Visualization of {base_name}')
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{base_name}_3d_matplotlib.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matplotlib visualization saved to: {output_path}")
    
    # Create orthogonal views
    create_orthogonal_views(binary_seg, output_dir, base_name)
    
    return output_path

def visualize_with_plotly(binary_seg, output_dir, base_name, sample_points=50000):
    """Create interactive 3D visualization using Plotly (better for interactive exploration)."""
    print("Creating 3D visualization with Plotly...")
    
    # Use marching cubes to get a surface mesh
    try:
        verts, faces, normals, values = measure.marching_cubes(binary_seg, level=0.5)
        
        # Create a plotly figure
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.7,
                colorscale='Reds',
                intensity=values,
                name='Vessel Network'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f"3D Visualization of {base_name}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=900,
            height=800,
            margin=dict(t=100)
        )
        
        # Save as HTML for interactive viewing
        html_output = os.path.join(output_dir, f"{base_name}_3d_interactive.html")
        fig.write_html(html_output)
        
        # Also save as an image
        img_output = os.path.join(output_dir, f"{base_name}_3d_plotly.png")
        fig.write_image(img_output)
        
        print(f"Interactive 3D visualization saved to: {html_output}")
        print(f"Static image saved to: {img_output}")
        
        return html_output
    
    except Exception as e:
        print(f"Error creating Plotly visualization: {str(e)}")
        print("Falling back to point-based visualization...")
        
        # Get the coordinates of the segmented voxels
        z, y, x = np.where(binary_seg)
        
        # If there are too many points, sample randomly
        if len(x) > sample_points:
            indices = np.random.choice(len(x), sample_points, replace=False)
            x = x[indices]
            y = y[indices]
            z = z[indices]
        
        # Create a plotly figure
        fig = go.Figure(data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=2,
                    color='red',
                    opacity=0.5
                ),
                name='Vessel Network'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f"3D Visualization of {base_name}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=900,
            height=800,
            margin=dict(t=100)
        )
        
        # Save as HTML for interactive viewing
        html_output = os.path.join(output_dir, f"{base_name}_3d_interactive.html")
        fig.write_html(html_output)
        
        print(f"Interactive 3D visualization saved to: {html_output}")
        
        return html_output

def create_orthogonal_views(binary_seg, output_dir, base_name):
    """Create maximum intensity projections along the three principal axes."""
    print("Creating maximum intensity projections...")
    
    # Create maximum intensity projections
    z_proj = np.max(binary_seg, axis=0)
    y_proj = np.max(binary_seg, axis=1)
    x_proj = np.max(binary_seg, axis=2)
    
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the projections
    axes[0].imshow(z_proj, cmap='gray')
    axes[0].set_title('Z Projection (Top View)')
    axes[0].axis('off')
    
    axes[1].imshow(y_proj, cmap='gray')
    axes[1].set_title('Y Projection (Front View)')
    axes[1].axis('off')
    
    axes[2].imshow(x_proj, cmap='gray')
    axes[2].set_title('X Projection (Side View)')
    axes[2].axis('off')
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{base_name}_projections.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Orthogonal projections saved to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Create 3D visualization of vessel segmentation")
    parser.add_argument("--input", required=True, help="Path to segmentation TIFF file")
    parser.add_argument("--output-dir", default=None, help="Directory to save visualization output")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation")
    parser.add_argument("--step-size", type=int, default=1, help="Step size for downsampling (higher = faster)")
    parser.add_argument("--use-matplotlib", action="store_true", help="Use Matplotlib instead of Plotly")
    parser.add_argument("--sample-points", type=int, default=50000, help="Number of points to sample in Plotly visualization")
    
    args = parser.parse_args()
    
    # Call the visualization function
    visualize_3d_volume(
        args.input,
        args.output_dir,
        args.threshold,
        args.step_size,
        not args.use_matplotlib,
        args.sample_points
    )

if __name__ == "__main__":
    main()
