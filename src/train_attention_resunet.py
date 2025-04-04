import os
import numpy as np
import tensorflow as tf

# Limit GPU memory growth
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

#tf.config.optimizer.set_jit(True)  # Enable XLA, attempt to improve performance    # Tror detta var den problematiska raden

# Enable multi-GPU strategy
#strategy = tf.distribute.MirroredStrategy()
#strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")  # Force single GPU

# Check number of GPUs
#print("Number of devices:", strategy.num_replicas_in_sync)

from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from skimage import data
from skimage.color import rgb2gray
from attention_resunet import attention_resunet  # Import Attention-ResUNet
from load_data import load_images, IMAGE_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_EPOCHS, NUM_BATCHSIZE  # Ensure this loads images correctly

# Maybe try these two lines below to save memory
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
tf.keras.backend.clear_session()


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss for handling small object segmentation."""
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -K.mean(alpha * K.pow(1 - pt, gamma) * K.log(pt))
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

def apply_gradient_checkpointing(model):
    for layer in model.layers:
        if isinstance(layer, Model):  # Apply to sub-models if present
            apply_gradient_checkpointing(layer)
        layer.experimental_run_tf_function = False  # Enable checkpointing

# Paths
#IMAGE_DIR = "images/r01_/rec_16bit_Paganin_0"
IMAGE_DIR = "3d-stacks/r01_"
MODEL_PATH = "models/attention_resunet_model.h5"

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

# Create model inside strategy scope
#with strategy.scope():
#    model = attention_resunet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))  # Replace with your model
#    apply_gradient_checkpointing(model)  
#    model.compile(optimizer=Adam(learning_rate=1e-4), loss=focal_loss, metrics=["accuracy"])

# Create & compile Attention ResUNet Model
model = attention_resunet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))
#Row below added in attempt to save memory
#apply_gradient_checkpointing(model)  
model.compile(optimizer=Adam(learning_rate=1e-4), loss=focal_loss, metrics=[focal_loss])

# Train Model
model.fit(train_dataset, validation_data=val_dataset, epochs=NUM_EPOCHS)

# Save Model
model.save(MODEL_PATH, save_format="h5")
print(f"Attention ResUNet model saved to {MODEL_PATH}")