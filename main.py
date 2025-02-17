import os
import sys
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

#models from scripts
from scripts.unet_model import unet_model
from scripts.attention_unet_model import attention_unet_model
from scripts.resunet_model import resunet_model
from scripts.config import IMG_HEIGHT, IMG_WIDTH

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


project_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths to images & masks
image_dir = os.path.join(project_dir, "images")
mask_dir = os.path.join(project_dir, "masks")

# Get image file names
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])

# Image size controlled in scripts/config.py
#IMG_HEIGHT, IMG_WIDTH = 512, 512

# Load and preprocess images
X = np.array([resize(iio.imread(os.path.join(image_dir, f)), (IMG_HEIGHT, IMG_WIDTH)) for f in image_files])
Y = np.array([resize(iio.imread(os.path.join(mask_dir, f)), (IMG_HEIGHT, IMG_WIDTH)) for f in mask_files])

# Expand dimensions for TensorFlow
X = np.expand_dims(X, axis=-1)
Y = np.expand_dims(Y, axis=-1)

# Split dataset
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Load model
model = unet_model()
#model = attention_unet_model()
#model = resunet_model()

# Train model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=8, epochs=4)

# Save model
model.save("models/unet_segmentation.h5")

print("Training Complete. Model saved in 'models/unet_segmentation.h5'")