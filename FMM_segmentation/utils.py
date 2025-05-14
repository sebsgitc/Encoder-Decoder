import SimpleITK as sitk
import numpy as np

def load_image(file_path):
    """
    Load a TIFF image and return as numpy array with spacing information
    
    Parameters:
        file_path: Path to the TIFF file
        
    Returns:
        tuple: (image_data, spacing)
            - image_data: numpy array with image data
            - spacing: tuple with pixel/voxel spacing (default: (1.0, 1.0, 1.0) if not in metadata)
    """
    import tifffile
    import os
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the TIFF file
    try:
        with tifffile.TiffFile(file_path) as tif:
            image = tif.asarray()
            
            # Try to extract spacing information from metadata if available
            spacing = (1.0, 1.0, 1.0)  # Default spacing
            
            if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata is not None:
                # Try to get spacing from ImageJ metadata
                metadata = tif.imagej_metadata
                if 'spacing' in metadata:
                    spacing = (metadata['spacing'], 1.0, 1.0)
                    
            elif hasattr(tif, 'pages') and hasattr(tif.pages[0], 'tags'):
                # Try to get spacing from TIFF tags
                tags = tif.pages[0].tags
                
                # Look for resolution tags
                if 'XResolution' in tags and 'YResolution' in tags:
                    x_res = tags['XResolution'].value
                    y_res = tags['YResolution'].value
                    
                    # Convert resolution to spacing
                    if x_res[1] != 0 and y_res[1] != 0:
                        x_spacing = x_res[1] / x_res[0]
                        y_spacing = y_res[1] / y_res[0]
                        if x_spacing == y_spacing:
                            spacing = (y_spacing, y_spacing, y_spacing)
                        else:
                            spacing = (1.0, 1.0, 1.0)  # Z, Y, X order for numpy arrays
    
    except Exception as e:
        raise IOError(f"Error loading TIFF file: {e}")
    
    # Make sure we have a 3D array
    if len(image.shape) == 2:
        # Convert 2D image to 3D with single slice
        image = image[np.newaxis, :, :]
    
    return image, spacing

def save_result(segmentation, output_path, spacing):
    """Save segmentation result to file"""
    result_image = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    result_image.SetSpacing(spacing)
    sitk.WriteImage(result_image, output_path)

def extract_vesselness(image, sigma=1.0):
    """Calculate vesselness filter response for tubular structure enhancement"""
    sitk_image = sitk.GetImageFromArray(image)
    
    hessian_filter = sitk.HessianRecursiveGaussianImageFilter()
    hessian_filter.SetSigma(sigma)
    
    hessian_image = hessian_filter.Execute(sitk_image)
    
    vesselness_filter = sitk.HessianToObjectnessMeasureImageFilter()
    vesselness_filter.SetObjectDimension(1)  # Tubular structures
    vesselness_filter.SetBrightObject(True)
    vesselness_filter.SetAlpha(0.5)
    vesselness_filter.SetBeta(0.5)
    vesselness_filter.SetGamma(5.0)
    
    vesselness_image = vesselness_filter.Execute(hessian_image)
    
    return sitk.GetArrayFromImage(vesselness_image)

def detect_leakage(current_mask, new_mask, growth_rate_threshold=2.0):
    """Detect leakage based on volume/surface growth rate"""
    current_volume = np.sum(current_mask)
    new_volume = np.sum(new_mask)
    
    current_surface = np.sum(ndimage.binary_dilation(current_mask) & ~current_mask)
    new_surface = np.sum(ndimage.binary_dilation(new_mask) & ~new_mask)
    
    volume_growth = new_volume - current_volume
    surface_growth = new_surface - current_surface
    
    if surface_growth <= 0:
        return False
    
    growth_rate = volume_growth / surface_growth
    
    # If no previous mask for comparison, return False
    if current_volume == 0:
        return False
    
    # Calculate previous growth rate
    previous_growth_rate = current_volume / current_surface if current_surface > 0 else 0
    
    # Check if growth rate has increased significantly
    if previous_growth_rate > 0 and growth_rate > growth_rate_threshold * previous_growth_rate:
        return True
        
    return False