from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import cv2  # Added for edge detection
import tensorflow as tf
import skimage

# Maybe try these two lines below to save memory
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy("mixed_float16")

#skimage.rgb2gray(images)
IMG_HEIGHT = 1024
IMG_WIDTH = 1024
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
NUM_EPOCHS = 2
NUM_BATCHSIZE = 6
NUM_CLASSES = 3

def create_border_mask(image):
    """Creates binary masks highlighting only the white borders dynamically."""
    image = tf.image.convert_image_dtype(image, tf.float32)  # Ensure correct dtype
    gray = tf.image.rgb_to_grayscale(image) if image.shape[-1] == 3 else image  # Convert to grayscale if needed
    gray = tf.expand_dims(gray, axis=0)  # Shape becomes (1, H, W, 1) or (1, H, W, 3)

    edges = tf.image.sobel_edges(gray)  # Edge detection (alternative to Canny)
    edges = tf.reduce_sum(edges, axis=-1)  # Sum gradient magnitudes
    edges = tf.squeeze(edges, axis=0)  # Remove the batch dimension, resulting in (H, W)

    edges = tf.where(edges > 0.3, 1.0, 0.0)  # Thresholding
    return edges



def process_image(image_path):
    """Loads a single image, ensures correct bit-depth, normalization, and shape."""
    image_path = image_path.decode("utf-8")  # Convert TF tensor to string

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Failed to load {image_path}. Image is None.")
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)  # Avoid crashing

    # Handle different bit depths
    if img.dtype == np.uint16:  
        img = img.astype(np.float32) / 65535.0  # Normalize 16-bit
    elif img.dtype == np.uint8:  
        img = img.astype(np.float32) / 255.0  # Normalize 8-bit

    # Resize image to (IMG_WIDTH, IMG_HEIGHT)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Ensure shape is (H, W, 1) for grayscale
    img = np.expand_dims(img, axis=-1)
    return img

def load_images(image_folder, batch_size=NUM_BATCHSIZE):
    """Creates a TensorFlow Dataset that loads images in batches on-demand."""
    if not os.path.exists(image_folder):
        print(f"ERROR: The folder {image_folder} does not exist!")
        return None

    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.tif', '.png', '.jpg'))]

    def load_and_preprocess(image_path):
        img = tf.numpy_function(process_image, [image_path], tf.float32)
        img.set_shape((IMG_HEIGHT, IMG_WIDTH, 1))  # Enforce shape
        mask = create_border_mask(img)  # Dynamically create mask
        return img, mask

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset