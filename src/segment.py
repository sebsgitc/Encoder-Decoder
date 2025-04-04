import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from sklearn.cluster import KMeans
from load_data import load_images, IMAGE_SIZE, NUM_CLASSES, NUM_BATCHSIZE, IMG_HEIGHT, IMG_WIDTH

# Maybe try these two lines below to save memory
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# Testar detta för att undvika krasch
tf.keras.backend.clear_session()

# Paths
USED_MODEL = "attention_resunet"
TEST_IMAGE_DIR = "3d-stacks/r01_"
MODEL_PATH = "models/" + USED_MODEL + "_model.h5"
#TEST_IMAGE_DIR = "images/r01_/rec_16bit_Paganin_0"
OUTPUT_DIR = "output/" + USED_MODEL + "_3d-test/"
#OUTPUT_DIR = "output/" + USED_MODEL + "/"

# Register custom loss function before loading
def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Load model with custom objects
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss for handling small objects in segmentation."""
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -tf.keras.backend.mean(alpha * tf.keras.backend.pow(1 - pt, gamma) * tf.keras.backend.log(pt))
    return loss

def contrastive_loss(y_true, y_pred):
    """
    Unsupervised loss that encourages distinct feature maps in the segmentation.
    - Uses pixel-wise contrastive clustering.
    """
    epsilon = 1e-6
    feature_mean = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    contrastive_term = tf.reduce_mean(tf.square(y_pred - feature_mean))  # Encourages cluster separation
    return contrastive_term

def cluster_segmentation(predictions, num_classes=NUM_CLASSES):
    """Uses OpenCV's highly optimized k-means implementation."""
    clustered_masks = []

    for feature_map in predictions:
        feature_flat = feature_map.reshape(-1, feature_map.shape[-1]).astype(np.float32)  # Convert to float32 for OpenCV
        _, labels, _ = cv2.kmeans(feature_flat, num_classes, None,
                                  criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                  attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
        clustered_img = labels.reshape(feature_map.shape[:2])
        clustered_masks.append(clustered_img)

    return np.array(clustered_masks)

# Visualize & Save results
def visualize_results(original, predicted, save_path=OUTPUT_DIR):
    j=0
    for i in range(len(original)):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Original Image
        axes[0].imshow(original[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Predicted Segmentation
        axes[1].imshow(predicted[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Predicted Segmentation")
        axes[1].axis("off")

        # Save figure
        output_file = os.path.join(save_path, f"segmented_{i}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved: {output_file}")
        j +=1
        if j == 10:
            break

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load model with correct loss function
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"focal_loss": focal_loss, "contrastive_loss": contrastive_loss})

print(f"Loaded model from {MODEL_PATH}")

# Load test images
test_images = load_images(TEST_IMAGE_DIR)

split = int(0.1 * len(list(test_images.as_numpy_iterator())))  # Calculate split index
original_images = test_images.take(split)   

original_images_list = []
for image in original_images:
    grayscale_image = tf.reduce_mean(image, axis=-1, keepdims=True)  # Average across the channels
    original_images_list.append(grayscale_image.numpy())  # Extract image tensor and convert to numpy array

# Convert list to numpy array and reshape to match model input
original_images_array = np.array(original_images_list)
original_images_array = original_images_array.reshape((-1, IMG_HEIGHT, IMG_WIDTH, 1))

# Run predictions
print("Running predictions...")
predictions = model.predict(original_images_array, batch_size=NUM_BATCHSIZE)
predictions = np.clip(predictions, 0, 1)  # Keeps grayscale range

# Apply clustering to segment the predictions
segmented_images = cluster_segmentation(predictions)

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

visualize_results(original_images_array, segmented_images)
print("Segmentation results saved in 'output/" + USED_MODEL + "/' folder.")