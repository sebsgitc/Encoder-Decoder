import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from resunet import resunet  # Import the ResUNet model
from load_data import load_images  # Ensure this loads preprocessed images

# Paths
IMAGE_DIR = "data/Excel cells"
MODEL_PATH = "models/resunet_model.h5"

# Load images
images, masks = load_images(IMAGE_DIR, color_mode="rgb")  # Use same setup as U-Net

# Train-test split
split = int(len(images) * 0.8)
X_train, X_val = images[:split], images[split:]
Y_train, Y_val = masks[:split], masks[split:]

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

# Define Focal Loss for Border Detection
import tensorflow.keras.backend as K

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss for handling small objects in segmentation."""
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -K.mean(alpha * K.pow(1 - pt, gamma) * K.log(pt))
    return loss

# Create & compile the model
model = resunet(input_shape=(256, 256, 3))  # Use same input shape as your U-Net
model.compile(optimizer=Adam(learning_rate=1e-4), loss=focal_loss, metrics=["accuracy"])

# Train Model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=6, epochs=50)

# Save Model
model.save(MODEL_PATH, save_format="h5")
print(f"Model saved to {MODEL_PATH}")