import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from PIL import Image
import numpy as np

# Paths
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "3d-stacks"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure output exists

# Image parameters
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 1024, 1024, 1024  # Target size
NUM_PARALLEL_CALLS = tf.data.AUTOTUNE  # Automatically optimize loading

def process_folder(subfolder):
    input_subfolder = os.path.join(INPUT_FOLDER, subfolder, "rec_16bit_Paganin_0")
    output_subfolder = os.path.join(OUTPUT_FOLDER, subfolder)
    os.makedirs(output_subfolder, exist_ok=True)  # Ensure output exists

    # List all image paths
    image_paths = sorted([os.path.join(input_subfolder, f) for f in os.listdir(input_subfolder) if f.endswith((".tif", ".png", ".jpg"))])

    # Ignore the first and last 56 images
    image_paths = image_paths[56:-56]

    # Select every second image to get 1024 images
    image_paths = image_paths[::2]

    # ** Step 1: TensorFlow Function to Load and Preprocess Images Efficiently**
    def load_and_preprocess_image(image_path):
        """Loads a 16-bit grayscale image, converts to 8-bit, resizes, and normalizes."""
        image_path = image_path.numpy().decode('utf-8')
        image = Image.open(image_path)
        image = np.array(image).astype(np.float32) / 65535.0  # Normalize to [0, 1]
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=-1)  # Add channel dimension
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])  # Resize to target size
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)  # Convert to 8-bit
        image = tf.squeeze(image, axis=-1)  # Remove channel dimension for saving
        return image

    def load_and_preprocess_image_wrapper(image_path):
        return tf.py_function(load_and_preprocess_image, [image_path], tf.uint8)

    # ** Step 2: Create TensorFlow Dataset for Fast Loading**
    def create_tf_dataset(image_paths):
        """Creates a TensorFlow dataset for fast parallelized image loading."""
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(load_and_preprocess_image_wrapper, num_parallel_calls=NUM_PARALLEL_CALLS)
        dataset = dataset.batch(1)  # Process images one by one
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    # ** Step 3: Process Images and Save**
    dataset = create_tf_dataset(image_paths)
    stacked_images = []

    for batch in dataset:
        img = batch[0]  # Extract the image
        stacked_images.append(img.numpy())

    # Stack all images along the depth axis
    final_stacked_img = np.stack(stacked_images, axis=-1)  # Stack into (H, W, D)

    # Option 1: Save as a single 3D volume
    output_path = os.path.join(output_subfolder, f"{subfolder}.8bit.tif")
    final_stacked_img = final_stacked_img.astype(np.uint8)
    final_stacked_img = np.transpose(final_stacked_img, (2, 0, 1))  # Transpose to (D, H, W)
    images = [Image.fromarray(final_stacked_img[i]) for i in range(final_stacked_img.shape[0])]
    images[0].save(output_path, save_all=True, append_images=images[1:])
    print(f"All images in {subfolder} processed and saved as a single 3D volume successfully!")
"""
    # Option 2: Save as individual 2D slices
    for i in range(final_stacked_img.shape[0]):
        slice_path = os.path.join(output_subfolder, f"{subfolder}_slice_{i:04d}.tif")
        Image.fromarray(final_stacked_img[i]).save(slice_path)
    print(f"All images in {subfolder} processed and saved as individual 2D slices successfully!")
"""
names_left = ["r014_", "r32b_", "r34_", "rL1_"]
# Process all subfolders in the input folder
#subfolders = [f for f in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, f))]
#for subfolder in subfolders:
for subfolder in names_left:
    process_folder(subfolder)

print("All subfolders processed and saved successfully!")