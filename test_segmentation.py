from tensorflow.keras.models import load_model
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from skimage.transform import resize
import os

# Load the trained model
model_path = "models/unet_segmentation.h5"
model = load_model(model_path)

project_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(project_dir, "images")

# Select an image for testing
test_image_name = sorted(os.listdir(image_dir))[0]  # Choose the first image
test_image_path = os.path.join(image_dir, test_image_name)

# Load and preprocess the test image
IMG_HEIGHT, IMG_WIDTH = 256, 256
test_image = iio.imread(test_image_path)
test_image_resized = resize(test_image, (IMG_HEIGHT, IMG_WIDTH))

# Expand dimensions to match model input shape
test_image_input = np.expand_dims(test_image_resized, axis=[0, -1])  # Shape: (1, 128, 128, 1)

# Predict segmentation mask
predicted_mask = model.predict(test_image_input)[0]  # Remove batch dimension

# Threshold the mask (convert probabilities to binary mask)
binary_mask = (predicted_mask > 0.5).astype(np.uint8)



# Display the results
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(test_image_resized, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Predicted Mask
plt.subplot(1, 3, 2)
plt.imshow(predicted_mask.squeeze(), cmap='gray')
plt.title("Predicted Mask (Raw)")
plt.axis("off")

# Thresholded Mask
plt.subplot(1, 3, 3)
plt.imshow(binary_mask.squeeze(), cmap='gray')
plt.title("Binary Segmentation Mask")
plt.axis("off")

plt.show()


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

dice_score = dice_coefficient(ground_truth_mask_resized, predicted_mask)
print(f"Dice Score: {dice_score:.4f}")    
