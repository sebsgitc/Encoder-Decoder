from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import cv2  # Added for edge detection
import skimage
#skimage.rgb2gray(images)
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
NUM_EPOCHS = 5
NUM_BATCHSIZE = 6

def create_border_masks(images):
    """Creates binary masks highlighting only the white borders."""
    masks = []
    for img in images:
        # Convert to grayscale
        gray = cv2.cvtColor(img.astype("float32"), cv2.COLOR_RGB2GRAY)
        # Apply edge detection
        edges = cv2.Canny((gray * 255).astype("uint8"), 100, 200) / 255.0  # Normalize 0-1
        masks.append(np.expand_dims(edges, axis=-1))  # Ensure shape (H, W, 1)
    return np.array(masks)

def load_images(image_folder, color_mode="rgb"):  
    """Loads images and generates corresponding border masks."""
    images = []

    if not os.path.exists(image_folder):
        print(f"ERROR: The folder {image_folder} does not exist!")
        return np.array([]), np.array([])

    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)

        try:
            img = load_img(img_path, target_size=IMAGE_SIZE, color_mode=color_mode)  
            img = img_to_array(img) / 255.0  # Normalize to 0-1
            images.append(img)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    images = np.array(images)

    # If no images were loaded, return empty arrays
    if len(images) == 0:
        return np.array([]), np.array([])

    masks = create_border_masks(images)  # Generate edge-detected masks
    return images, masks  # Return both images and masks