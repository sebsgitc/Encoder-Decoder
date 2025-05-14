import os
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

def visualize_result(image, segmentation, distance_map, output_dir):
    """Visualize segmentation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot middle slices in each direction
    z_mid = image.shape[0] // 2
    y_mid = image.shape[1] // 2
    x_mid = image.shape[2] // 2
    
    # Axial view
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image[z_mid], cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(132)
    plt.imshow(image[z_mid], cmap='gray')
    plt.imshow(segmentation[z_mid], alpha=0.3, cmap='hot')
    plt.title('Segmentation Overlay')
    
    plt.subplot(133)
    plt.imshow(distance_map[z_mid], cmap='jet')
    plt.colorbar()
    plt.title('Distance Map')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'axial_view.png'))
    
    # Sagittal view
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image[:, y_mid, :].T, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(132)
    plt.imshow(image[:, y_mid, :].T, cmap='gray')
    plt.imshow(segmentation[:, y_mid, :].T, alpha=0.3, cmap='hot')
    plt.title('Segmentation Overlay')
    
    plt.subplot(133)
    plt.imshow(distance_map[:, y_mid, :].T, cmap='jet')
    plt.colorbar()
    plt.title('Distance Map')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sagittal_view.png'))
    
    # Create max intensity projection
    plt.figure(figsize=(10, 10))
    plt.imshow(np.max(image * segmentation, axis=0), cmap='gray')
    plt.title('Maximum Intensity Projection of Segmented Vessels')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vessel_projection.png'))
    
    # # Plot growth analysis
    # time_points = np.linspace(5, np.max(distance_map[segmentation]), 20)
    # volumes = []
    
    # for t in time_points:
    #     volumes.append(np.sum(distance_map < t))
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(time_points, volumes)
    # plt.xlabel('Time Threshold')
    # plt.ylabel('Volume')
    # plt.title('Growth Analysis')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'growth_analysis.png'))