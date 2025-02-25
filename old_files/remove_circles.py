import numpy as np
import matplotlib.pyplot as plt
from skimage import color, img_as_uint
from skimage.filters import gaussian
from skimage.draw import disk
from skimage.transform import hough_circle, hough_circle_peaks

def remove_circles(image, min_radius=10, max_radius=100):
    """
    Detects and removes circles from a 16-bit grayscale image using Hough Circle Transform in scikit-image.
    
    Parameters:
        image (numpy array): Input 16-bit grayscale image.
        min_radius (int): Minimum radius of detected circles.
        max_radius (int): Maximum radius of detected circles.
    
    Returns:
        numpy array: Image with detected circles removed (still in 16-bit).
    """
    # Ensure image is not empty
    if image is None or image.size == 0:
        print("Warning: Received an empty image.")
        return image  # Return original image

    # Ensure image is 16-bit grayscale
    #if image.dtype != np.uint16:
    #    raise ValueError("Input image must be 16-bit grayscale (dtype=np.uint16)")

    # Convert to floating point (normalize to range 0-1) but keep original values
    image_float = image.astype(np.float32) / 65535.0

    # Apply Gaussian blur to reduce noise (uses float32)
    blurred = gaussian(image_float, sigma=1)

    # Define circle radii range
    hough_radii = np.arange(min_radius, max_radius, 2)

    # Detect circles using Hough Transform
    hough_res = hough_circle(blurred, hough_radii)

    # Extract detected circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=5)

    # Create a mask for detected circles
    mask = np.ones_like(image_float)

    for center_y, center_x, radius in zip(cy, cx, radii):
        rr, cc = disk((center_y, center_x), radius, shape=image.shape)
        mask[rr, cc] = 0  # Set detected circles to 0

    # Apply mask to remove detected circles
    cleaned_image = image_float * mask

    # Convert back to original 16-bit range
    cleaned_image = img_as_uint(cleaned_image)

    return cleaned_image
