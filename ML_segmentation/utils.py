"""Utility functions for binary vessel segmentation using TensorFlow."""

import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add these utility functions to handle NaN values

def safe_mean(x, fallback=0.0):
    """
    Calculate mean with protection against NaN values.
    
    Args:
        x: Tensor or array to calculate mean from
        fallback: Value to return if all values are NaN
    
    Returns:
        Mean value, or fallback if mean would be NaN
    """
    try:
        import tensorflow as tf
        import numpy as np
        
        if isinstance(x, tf.Tensor):
            # For TensorFlow tensors
            mask = tf.logical_not(tf.math.is_nan(x))
            safe_x = tf.boolean_mask(x, mask)
            count = tf.reduce_sum(tf.cast(mask, tf.float32))
            
            if count > 0:
                return tf.reduce_sum(safe_x) / count
            return tf.constant(fallback, dtype=x.dtype)
        else:
            # For numpy arrays
            valid_values = x[~np.isnan(x)] if hasattr(x, '__len__') else x
            if len(valid_values) > 0:
                return np.mean(valid_values)
            return fallback
    except Exception:
        return fallback

# Update AverageMeter class to handle NaN values
class AverageMeter:
    """Computes and stores the average and current value with NaN protection."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update with NaN protection."""
        import numpy as np
        
        # Skip NaN values
        if np.isnan(val):
            return
        
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

def save_checkpoint(state, is_best, filename='checkpoints/checkpoint'):
    """Save checkpoint to disk."""
    # Save model weights - adding .weights.h5 extension as required by TensorFlow
    state['model'].save_weights(f"{filename}.weights.h5")
    
    # If this is the best model, save a copy
    if is_best:
        state['model'].save_weights('checkpoints/model_best.weights.h5')

def load_checkpoint(model, checkpoint_path):
    """Load checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load weights
    model.load_weights(checkpoint_path)
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model

def create_logger():
    """Create and configure logger."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('lung_segmentation')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'logs/training_{timestamp}.log')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def visualize_segmentation(image, mask, prediction=None, save_path=None):
    """
    Visualize image, ground truth mask and prediction for binary segmentation.
    
    Args:
        image: Input image
        mask: Ground truth mask
        prediction: Model prediction (optional)
        save_path: Path to save the visualization (optional)
    
    Returns:
        Path to saved image if save_path is provided, otherwise None
    """
    plt.figure(figsize=(15, 5))
    
    # Show original image
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    # Show ground truth mask - use binary colormap
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(mask, cmap='binary', vmin=0, vmax=1)
    plt.axis('off')
    
    # Show prediction if available
    if prediction is not None:
        plt.subplot(133)
        plt.title('Prediction')
        plt.imshow(prediction, cmap='binary', vmin=0, vmax=1)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()
        return None

def apply_2d_sliding_window_prediction(model, image, window_size=256, overlap=0.5):
    """Apply sliding window prediction on a 2D image for binary classification."""
    model.trainable = False
    
    h, w = image.shape
    stride = int(window_size * (1 - overlap))
    predictions = np.zeros((h, w), dtype=np.float32)  # For binary probabilities
    counts = np.zeros((h, w), dtype=np.float32)
    
    # Convert image to tensor and normalize if needed
    if isinstance(image, np.ndarray):
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        if tf.reduce_max(image_tensor) > 1.0:
            image_tensor = image_tensor / 255.0
    else:
        image_tensor = image
    
    # Apply sliding window
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            # Extract window
            window = image_tensor[y:y+window_size, x:x+window_size]
            window = tf.expand_dims(window, axis=0)  # Add batch dimension
            window = tf.expand_dims(window, axis=-1)  # Add channel dimension
            
            # Predict
            pred = model(window, training=False).numpy()[0].squeeze()  # Get sigmoid probabilities
            
            # Accumulate predictions and counts
            predictions[y:y+window_size, x:x+window_size] += pred
            counts[y:y+window_size, x:x+window_size] += 1
    
    # Average predictions by counts
    predictions /= np.maximum(counts, 1)
    
    # Get final binary segmentation
    segmentation = (predictions > 0.5).astype(np.int32)
    return segmentation

def dice_coefficient(y_true, y_pred, smooth=1):
    """Calculate Dice coefficient."""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def generate_confusion_matrix(y_true, y_pred, num_classes=2):
    """Generate confusion matrix for binary segmentation."""
    conf_matrix = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(num_classes):
            conf_matrix[i, j] = np.sum((y_true == i) & (y_pred == j))
    
    return conf_matrix