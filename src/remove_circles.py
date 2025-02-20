import cv2
import numpy as np

def remove_circles(image, min_radius=10, max_radius=100, param1=50, param2=30, angular_range=(30, 250)):
    """
    Detects and removes circular arcs from an image using Hough Circle Transform.
    
    Parameters:
        image (numpy array): Input image (should be grayscale or RGB).
        min_radius (int): Minimum radius of detected circles.
        max_radius (int): Maximum radius of detected circles.
        param1 (int): First parameter for HoughCircles (Canny edge detection threshold).
        param2 (int): Second parameter for HoughCircles (accumulator threshold for circle detection).
        angular_range (tuple): Tuple (start_angle, end_angle) defining which part of the circle to remove.
        
    Returns:
        numpy array: Image with detected circles removed.
    """
    # Convert to grayscale if input is RGB
    if len(image.shape) == 3 and image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )

    # If no circles are detected, return the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Convert to integer values

        mask = np.ones_like(gray, dtype=np.uint8) * 255  # White mask

        for circle in circles[0, :]:
            x, y, radius = circle

            # Draw full circle mask
            cv2.circle(mask, (x, y), radius, 0, thickness=-1)

            # Remove only the specified angular section
            for angle in range(angular_range[0], angular_range[1]):
                angle_rad = np.deg2rad(angle)
                arc_x = int(x + radius * np.cos(angle_rad))
                arc_y = int(y + radius * np.sin(angle_rad))
                cv2.line(mask, (x, y), (arc_x, arc_y), 255, thickness=2)

        # Apply mask to remove the detected circles
        cleaned_image = cv2.bitwise_and(gray, gray, mask=mask)

        # If original image was RGB, convert back to RGB
        if len(image.shape) == 3 and image.shape[-1] == 3:
            cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2RGB)

        return cleaned_image

    return image  # Return original image if no circles were detected