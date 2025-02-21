import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from load_data import load_images, IMAGE_SIZE

#circ shit
#from remove_circles import remove_circles

# Paths
USED_MODEL = "attention_unet"
#MODEL_PATH = "models/resunet_model.h5"
#MODEL_PATH = "models/unet_model.h5"
#MODEL_PATH = "models/fpn_model.h5"
#MODEL_PATH = "models/attention_unet_model.h5"
#MODEL_PATH = "models/attention_resunet_model.h5"
MODEL_PATH = "models/" + USED_MODEL + "_model.h5"

#TEST_IMAGE_DIR = "data/raw"
#TEST_IMAGE_DIR = "data/Excel cells"
TEST_IMAGE_DIR = "images/r01_/rec_16bit_Paganin_0"

OUTPUT_DIR = "output/" + USED_MODEL + "/"

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

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

# Load model with correct loss function
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"focal_loss": focal_loss})

print(f"Loaded model from {MODEL_PATH}")

# Load test images
test_images, _ = load_images(TEST_IMAGE_DIR, color_mode="rgb")  # Ignore masks
#circ sh
#test_images = np.array([remove_circles(img) for img in test_images])

# Run predictions
print("Running predictions...")
predictions = model.predict(test_images)

# Ensure output is in correct range
#predictions = np.clip(predictions, 0, 1)
# Convert predictions to binary mask (Thresholding)
threshold = 0.2 
predictions = (predictions > threshold).astype(np.float32)

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visualize & Save results
def visualize_results(original, predicted, save_path=OUTPUT_DIR):
    j=0
    for i in range(len(original)):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Original Image
        axes[0].imshow(original[i])
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Predicted Segmentation
        axes[1].imshow(predicted[i])
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

visualize_results(test_images, predictions)
print("Segmentation results saved in 'output/" + USED_MODEL + "/' folder.")