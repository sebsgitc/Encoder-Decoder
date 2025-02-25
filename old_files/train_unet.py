import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model
from load_data import load_images

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from load_data import load_images  # Ensure this correctly loads images & masks

# Paths
IMAGE_DIR = "data/Excel cells"
MODEL_PATH = "models/unet_model.h5"

# Load images & border masks
images, masks = load_images(IMAGE_DIR, color_mode="rgb")

# Train-test split
split = int(len(images) * 0.8)
X_train, X_val = images[:split], images[split:]
Y_train, Y_val = masks[:split], masks[split:]

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"Sample min/max values - X_train: {X_train.min()} to {X_train.max()}, Y_train: {Y_train.min()} to {Y_train.max()}")

# Define Full U-Net Model
def unet_model(input_shape=(256, 256, 1)):
    """U-Net model for border detection."""
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(u5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Focal Loss for Better Segmentation
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss for handling small objects in segmentation."""
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -tf.keras.backend.mean(alpha * tf.keras.backend.pow(1 - pt, gamma) * tf.keras.backend.log(pt))
    return loss

# Create and Compile Model
model = unet_model(input_shape=(256, 256, 3))
model.compile(optimizer=Adam(learning_rate=1e-3), loss=focal_loss, metrics=["accuracy"])

# Train Model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=8, epochs=30)

# Save Model
model.save(MODEL_PATH, save_format="h5")
print(f"Model saved to {MODEL_PATH}")