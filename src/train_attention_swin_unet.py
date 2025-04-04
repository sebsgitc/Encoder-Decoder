import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from attention_swin_unet import attention_swin_unet  # Import the new model
from load_data import load_images, IMG_HEIGHT, IMG_WIDTH, NUM_EPOCHS, NUM_BATCHSIZE  # Ensure this loads images correctly

# Maybe try these two lines below to save memory
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
tf.keras.backend.clear_session()

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss for handling small object segmentation."""
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(tf.keras.backend.equal(y_true, 1), y_pred, 1 - y_pred)
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

# Paths
IMAGE_DIR = "images/r01_/rec_16bit_Paganin_0"
MODEL_PATH = "models/attention_swin_unet_model.h5"

# Load images & masks
dataset = load_images(IMAGE_DIR)

# Print and inspect the dataset
for image, mask in dataset.take(1):
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

# Train-test split
split = int(0.8 * len(list(dataset.as_numpy_iterator())))  # Calculate split index
train_dataset = dataset.take(split)   # 80% for training
val_dataset = dataset.skip(split)     # 20% for validation

# Create & compile Attention Swin U-Net Model
model = attention_swin_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer=Adam(learning_rate=1e-4), loss=contrastive_loss, metrics=["accuracy"])

# Train Model
model.fit(train_dataset, validation_data=val_dataset, epochs=NUM_EPOCHS)

# Save Model
model.save(MODEL_PATH, save_format="h5")
print(f"Attention Swin U-Net model saved to {MODEL_PATH}")