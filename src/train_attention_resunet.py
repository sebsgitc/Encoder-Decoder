import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from attention_resunet import attention_resunet  # Import Attention-ResUNet
from load_data import load_images, IMAGE_SIZE, IMG_HEIGHT, IMG_WIDTH  # Ensure this loads images correctly


#removing circles
from remove_circles import remove_circles
# Define Focal Loss BEFORE using it
import tensorflow.keras.backend as K

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss for handling small object segmentation."""
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -K.mean(alpha * K.pow(1 - pt, gamma) * K.log(pt))
    return loss

# Paths
#IMAGE_DIR = "data/Excel cells"
IMAGE_DIR = "images/r01_/rec_16bit_Paganin_0"
MODEL_PATH = "models/attention_resunet_model.h5"

# Load images & masks
images, masks = load_images(IMAGE_DIR, color_mode="rgb")

#circle shit
images = np.array([remove_circles(img) for img in images])


# Train-test split
split = int(len(images) * 0.8)
X_train, X_val = images[:split], images[split:]
Y_train, Y_val = masks[:split], masks[split:]

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"Sample min/max values - X_train: {X_train.min()} to {X_train.max()}, Y_train: {Y_train.min()} to {Y_train.max()}")

# Create & compile Attention ResUNet Model
model = attention_resunet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer=Adam(learning_rate=1e-4), loss=focal_loss, metrics=["accuracy"])

# Train Model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=6, epochs=5)

# Save Model
model.save(MODEL_PATH, save_format="h5")
print(f"Attention ResUNet model saved to {MODEL_PATH}")