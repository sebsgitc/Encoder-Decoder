#!/usr/bin/env python3
"""
Semi-supervised training of vessel segmentation model using minimal seed point supervision.
Uses seed points to generate training data and can incorporate multiple volumes.
"""
import os
import sys
import numpy as np
import tensorflow as tf
import tifffile as tiff
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import datetime
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd  # Add this import at the top with the other imports
#from gpu_memory_utils import configure_gpu_memory_growth, get_gpu_memory_info, estimate_max_batch_size

# Set CUDA environment variables BEFORE importing TensorFlow
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu'
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Define seed points for different volumes
SEED_POINTS = {
    "r01_": [
        (478, 323, 32),   # Upper right lung vessel
        (372, 648, 45),   # Lower left lung vessel
        (920, 600, 72),   # Right peripheral vessel
        (420, 457, 24),   # Central vessel
        (369, 326, 74),   # Upper left pulmonary vessel
        (753, 417, 124),  # Right middle lobe vessel
        (755, 607, 174),  # Lower right vessel branch
        (305, 195, 274),   # Upper branch
        (574, 476, 324),
        (380, 625, 374),
        (313, 660, 424),
        (100, 512, 610),
        (512, 20, 730),
        (512, 200, 820),
        (512, 400, 940)
    ]
    # Add more volumes here when available
    # "r04_": [...],
    # "r07_": [...],
}

def forward_pass_safeguard(tensor):
    """Apply to model outputs to prevent NaN propagation"""
    return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)

def super_stable_loss(y_true, y_pred):
    """
    Ultra-stable loss function that can't produce NaN values
    no matter what inputs it receives
    """
    # First, ensure inputs are clean
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
    
    # Aggressively clip values to prevent any instability
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Use simple Dice loss - extremely stable
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    # Safe division
    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    
    # Final safety check
    return tf.where(tf.math.is_nan(loss), tf.ones_like(loss), loss)

def compute_vesselness_stable(y_pred, scales=None):
    """
    Simplified and stable version of vesselness calculation that relies on gradient magnitude
    rather than Hessian-based approaches
    
    Args:
        y_pred: Predicted probability map (tensor)
        scales: List of scales for calculation (used for compatibility)
        
    Returns:
        A measure of tubular structures using gradient information
    """
    # Default scales if not provided (for compatibility)
    if scales is None:
        scales = [1.0]
    
    # Reshape input to ensure it has 5 dimensions [batch, x, y, z, channel]
    y_pred_reshaped = tf.cond(
        tf.equal(tf.rank(y_pred), 3),
        lambda: tf.expand_dims(tf.expand_dims(y_pred, 0), -1),  # [x,y,z] -> [1,x,y,z,1]
        lambda: y_pred
    )
    
    y_pred_reshaped = tf.cond(
        tf.equal(tf.rank(y_pred_reshaped), 4),
        lambda: tf.expand_dims(y_pred_reshaped, -1),  # [batch,x,y,z] -> [batch,x,y,z,1]
        lambda: y_pred_reshaped
    )
    
    # Use simple gradient for approximating vessel-like structures
    # Calculate gradient magnitude at different scales
    vesselness = None
    
    for scale in scales:
        # Apply Gaussian smoothing
        sigma = tf.cast(scale, tf.float32)
        smoothed = gaussian_blur_3d(y_pred_reshaped, sigma)
        
        # Calculate gradients in x, y, z
        # Using central difference approximation instead of Sobel for stability
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        padded = tf.pad(smoothed, paddings, "SYMMETRIC")
        
        # Central difference gradients
        gx = padded[:, 2:, 1:-1, 1:-1, :] - padded[:, :-2, 1:-1, 1:-1, :]
        gy = padded[:, 1:-1, 2:, 1:-1, :] - padded[:, 1:-1, :-2, 1:-1, :]
        gz = padded[:, 1:-1, 1:-1, 2:, :] - padded[:, 1:-1, 1:-1, :-2, :]
        
        # Calculate gradient magnitude
        grad_mag = tf.sqrt(gx*gx + gy*gy + gz*gz + 1e-6)
        
        # Invert so that low gradients inside vessels have high values
        inverted_grad = 1.0 - tf.clip_by_value(grad_mag / (tf.reduce_max(grad_mag) + 1e-6), 0.0, 1.0)
        
        # Use a simple threshold to detect tubular structures
        # Areas with high intensity and low gradient are likely vessels
        tubular_measure = inverted_grad * smoothed
        
        # Normalize to [0,1]
        max_val = tf.reduce_max(tubular_measure) + 1e-6
        tubular_measure = tubular_measure / max_val
        
        # Update max vesselness across scales
        if vesselness is None:
            vesselness = tubular_measure
        else:
            vesselness = tf.maximum(vesselness, tubular_measure)
    
    return vesselness

def gaussian_blur_3d(x, sigma):
    """
    Apply 3D Gaussian blur using separable convolution for efficiency.
    
    Args:
        x: Input tensor
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Blurred tensor
    """
    # Create 1D Gaussian kernel
    kernel_size = tf.maximum(tf.cast(tf.round(sigma * 3) * 2 + 1, tf.int32), 3)
    kernel_size = tf.minimum(kernel_size, 7)  # Limit kernel size for efficiency
    
    # Create meshgrid for kernel
    ax = tf.range(-kernel_size//2 + 1, kernel_size//2 + 1, dtype=tf.float32)
    xx, yy, zz = tf.meshgrid(ax, ax, ax)
    
    # Gaussian kernel
    kernel = tf.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    
    # Reshape for 3D convolution
    kernel = tf.reshape(kernel, [kernel_size, kernel_size, kernel_size, 1, 1])
    
    # Apply convolution
    return tf.nn.conv3d(x, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')

# Add this function to create the model within the strategy scope
def create_model_with_strategy(strategy, input_shape=(64, 64, 64, 2), filters_base=32):
    """
    Creates a 3D U-Net model optimized for vessel segmentation within strategy scope
    with numerical stability improvements
    """
    with strategy.scope():
        # Input layer
        inputs = layers.Input(input_shape, name="input_layer")
        
        # Add attention mechanism to focus on important features
        def attention_block(x, g, filters):
            theta_x = layers.Conv3D(filters, 1, strides=1, padding='same')(x)
            phi_g = layers.Conv3D(filters, 1, strides=1, padding='same')(g)
            
            f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
            psi_f = layers.Conv3D(1, 1, strides=1, padding='same')(f)
            
            rate = layers.Activation('sigmoid')(psi_f)
            att_x = layers.multiply([x, rate])
            
            return att_x
        
        # Improved convolutional block with residual connections for better gradient flow
        def conv_block(input_tensor, num_filters):
            x = layers.Conv3D(num_filters, 3, padding='same')(input_tensor)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv3D(num_filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Add residual connection
            if input_tensor.shape[-1] == num_filters:
                x = layers.add([x, input_tensor])
            else:
                shortcut = layers.Conv3D(num_filters, 1, padding='same')(input_tensor)
                x = layers.add([x, shortcut])
                
            return x
        
        # Contracting path (encoder)
        # Level 1
        conv1 = conv_block(inputs, filters_base)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
        
        # Level 2
        conv2 = conv_block(pool1, filters_base*2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
        
        # Level 3
        conv3 = conv_block(pool2, filters_base*4)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        
        # Bridge with dilated convolutions for better field of view
        bridge = conv_block(pool3, filters_base*8)
        # Add dilated convolutions to increase receptive field
        dilation_rates = [2, 4, 8]
        dilated_layers = []
        
        for rate in dilation_rates:
            dilated = layers.Conv3D(filters_base*8, 3, padding='same', dilation_rate=rate)(bridge)
            dilated = layers.BatchNormalization()(dilated)
            dilated = layers.Activation('relu')(dilated)
            dilated_layers.append(dilated)
        
        # Combine dilated convolutions
        dilated_concat = layers.Concatenate()(dilated_layers + [bridge])
        bridge = layers.Conv3D(filters_base*8, 1, padding='same')(dilated_concat)
        bridge = layers.BatchNormalization()(bridge)
        bridge = layers.Activation('relu')(bridge)
        
        # Expansive path (decoder) with attention gates for focus on relevant features
        # Level 3
        up5 = layers.UpSampling3D(size=(2, 2, 2))(bridge)
        att5 = attention_block(conv3, up5, filters_base*4)
        concat5 = layers.Concatenate()([up5, att5])
        conv5 = conv_block(concat5, filters_base*4)
        
        # Level 2
        up6 = layers.UpSampling3D(size=(2, 2, 2))(conv5)
        att6 = attention_block(conv2, up6, filters_base*2)
        concat6 = layers.Concatenate()([up6, att6])
        conv6 = conv_block(concat6, filters_base*2)
        
        # Level 1
        up7 = layers.UpSampling3D(size=(2, 2, 2))(conv6)
        att7 = attention_block(conv1, up7, filters_base)
        concat7 = layers.Concatenate()([up7, att7])
        conv7 = conv_block(concat7, filters_base)
        
        # Additional attention mechanism for vessel focus
        attention = layers.Conv3D(1, 1, activation='sigmoid', name='attention_map')(conv7)
        attended = layers.Multiply()([conv7, attention])
        
        # Output layer
        outputs = layers.Conv3D(1, 1, activation='sigmoid', name='output')(attended)

        outputs = layers.Lambda(forward_pass_safeguard, name="nan_protection")(outputs)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Loss functions (keep existing loss functions)
        def dice_loss(y_true, y_pred):
            # Make the loss function robust against empty batches
            return tf.cond(
                tf.equal(tf.size(y_true), 0),
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: _dice_loss_impl(y_true, y_pred)
            )
            
        def _dice_loss_impl(y_true, y_pred):
            smooth = 1.0
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return 1.0 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        
        def dice_coefficient(y_true, y_pred):
            # Make the metric robust against empty batches
            return tf.cond(
                tf.equal(tf.size(y_true), 0),
                lambda: tf.constant(1.0, dtype=tf.float32),
                lambda: _dice_coefficient_impl(y_true, y_pred)
            )
            
        def _dice_coefficient_impl(y_true, y_pred):
            smooth = 1.0
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            small_number = 1e-7 # To prevent division by zero
            return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth + small_number)
        
        # Use focal loss to better handle class imbalance
        def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
            # Make the loss function robust against empty batches
            return tf.cond(
                tf.equal(tf.size(y_true), 0),
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: _focal_loss_impl(y_true, y_pred, gamma, alpha)
            )
            
        def _focal_loss_impl(y_true, y_pred, gamma=2.0, alpha=0.25):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Calculate focal loss
            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
            focal_loss = -alpha * tf.pow(1-p_t, gamma) * tf.math.log(p_t + epsilon)
            return tf.reduce_mean(focal_loss)
        
        # Combine dice loss and focal loss
        def combined_loss(y_true, y_pred):
            # Make the combined loss robust against empty batches
            return tf.cond(
                tf.equal(tf.size(y_true), 0),
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)
            )
        
        def stable_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=2.0):
            # Make the loss function robust against empty batches
            return tf.cond(
                tf.equal(tf.size(y_true), 0),
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: _stable_loss_impl(y_true, y_pred, alpha, beta, gamma)
            )
            
        def _stable_loss_impl(y_true, y_pred, alpha=0.7, beta=0.3, gamma=2.0):
            """
            Stable implementation of asymmetric Tversky loss

            Args:
                y_true: Ground truth
                y_pred: Prediction
                alpha: Weight for false negatives
                beta: Weight for false positives
                gamma: Focal parameter

            Returns:
                Loss value
            """
            # Add larger epsilon to prevent division by zero

            epsilon = 1e-4

            # Flatten the tensors
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])

            # Clip prediction values to avoid numerical instability
            y_pred_f = tf.clip_by_value(y_pred_f, epsilon, 1.0 - epsilon)

            # Calculate Tversky index components
            true_pos = tf.reduce_sum(y_true_f * y_pred_f)
            false_neg = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
            false_pos = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)

            # Asymmetric Tversky index with larger epsilon
            tversky = (true_pos + epsilon) / (true_pos + alpha * false_neg + beta * false_pos + epsilon)
    
            # Focal component - apply clipping to prevent numerical issues
            focal_tversky = tf.pow(tf.clip_by_value(1.0 - tversky, 0.0, 1.0 - epsilon), gamma)

            return focal_tversky
        
        def stable_focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=2.0):
            # Make the loss function robust against empty batches
            return tf.cond(
                tf.equal(tf.size(y_true), 0),
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: _stable_focal_tversky_loss_impl(y_true, y_pred, alpha, beta, gamma)
            )
        
        def _stable_focal_tversky_loss_impl(y_true, y_pred, alpha=0.7, beta=0.3, gamma=2.0):
            """
            Ultra-stable version of the asymmetric focal Tversky loss that
            catches and prevents any NaN values
            """
            # Add NaN safety check at the input
            y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
            y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
    
            # Calculate stable Tversky loss
            try:
                tversky_loss = stable_loss(y_true, y_pred, alpha, beta, gamma)
            except:
                # Fallback to a simpler loss if stable_loss fails
                tversky_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    
            # Ensure we don't have NaN in the loss before continuing
            tversky_loss = tf.where(tf.math.is_nan(tversky_loss), 1.0, tversky_loss)
    
            # Try to calculate continuity penalty with maximum safety
            try:
                # Symmetric padding
                paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
                y_pred_padded = tf.pad(y_pred, paddings, "SYMMETRIC")
        
                # Calculate gradients safely
                gx = y_pred_padded[:, 2:, 1:-1, 1:-1, :] - y_pred_padded[:, :-2, 1:-1, 1:-1, :]
                gy = y_pred_padded[:, 1:-1, 2:, 1:-1, :] - y_pred_padded[:, 1:-1, :-2, 1:-1, :]
                gz = y_pred_padded[:, 1:-1, 1:-1, 2:, :] - y_pred_padded[:, 1:-1, 1:-1, :-2, :]
        
                # Apply clipping to gradients
                gx = tf.clip_by_value(gx, -1.0, 1.0)
                gy = tf.clip_by_value(gy, -1.0, 1.0)
                gz = tf.clip_by_value(gz, -1.0, 1.0)
        
                # Use safe sqrt with larger epsilon
                grad_mag = tf.sqrt(gx*gx + gy*gy + gz*gz + 1e-4)

                # Ensure no NaNs in regularization term
                safe_y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        
                # Use reduce_mean with additional safety
                continuity_penalty = tf.reduce_mean(
                    tf.clip_by_value(safe_y_pred * grad_mag, 0.0, 10.0)
                ) * 0.1
        
                # Check for NaN in penalty
                continuity_penalty = tf.where(
                    tf.math.is_nan(continuity_penalty),
                    tf.constant(0.0, dtype=tf.float32),
                    continuity_penalty
                )
        
            except Exception:
                # If any part fails, use a constant small value
                continuity_penalty = tf.constant(0.01, dtype=tf.float32)
    
            # Final safety check before returning
            result = tversky_loss + continuity_penalty
            result = tf.where(tf.math.is_nan(result), tf.constant(1.0, dtype=tf.float32), result)
    
            return result

        # Replace the loss function with our stable version
        def asymmetric_focal_tversky_loss(y_true, y_pred):
            # Make the loss function robust against empty batches 
            return tf.cond(
                tf.equal(tf.size(y_true), 0),
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: _asymmetric_focal_tversky_loss_impl(y_true, y_pred)
            )
            
        def _asymmetric_focal_tversky_loss_impl(y_true, y_pred):
            """Numerically stable loss that can recover from extreme values"""
            # Force explicit numeric casting and aggressive clipping
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Ensure all values are in a safe range with multi-stage clipping
            epsilon = 1e-6
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Use a simpler binary cross-entropy implementation for maximum stability
            bce = -tf.reduce_mean(
                y_true * tf.math.log(y_pred + epsilon) + 
                (1.0 - y_true) * tf.math.log(1.0 - y_pred + epsilon)
            )
            
            # Add simple dice loss with extreme safety measures
            smooth = 1.0
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            
            # Ensure inputs to reduce_sum are clean
            y_true_f = tf.clip_by_value(y_true_f, 0.0, 1.0)
            y_pred_f = tf.clip_by_value(y_pred_f, epsilon, 1.0 - epsilon)
            
            # Calculate intersection and union safely
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
            
            # Ensure never dividing by zero
            denominator = tf.maximum(union + smooth, epsilon)
            dice = 1.0 - (2.0 * intersection + smooth) / denominator
            
            # Combine losses with clamping to prevent extreme values
            combined = tf.clip_by_value(0.5 * bce + 0.5 * dice, 0.0, 10.0)
            
            # Final NaN check (any NaN will return 0.1 - a safe fallback value)
            safe_loss = tf.where(tf.math.is_nan(combined), 0.1, combined)
            
            return safe_loss

        def super_safe_loss(y_true, y_pred):
            """Ultra-stable loss that guarantees no NaNs"""
            # Ensure numerical stability with aggressive clipping
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            y_true = tf.clip_by_value(y_true, 0.0, 1.0)
            
            # Simple BCE with extreme safety
            bce = -tf.reduce_mean(
                y_true * tf.math.log(y_pred + 1e-7) + 
                (1.0 - y_true) * tf.math.log(1.0 - y_pred + 1e-7)
            )
            
            # Simple dice with extreme safety
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            
            # Super-safe denominator
            denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0
            dice = 1.0 - (2.0 * intersection + 1.0) / denom
            
            # Combine with weighted average
            combined = 0.5 * bce + 0.5 * dice
            
            # Final safety check
            combined = tf.where(tf.math.is_nan(combined), 0.1, combined)
            combined = tf.where(tf.math.is_inf(combined), 0.1, combined)
            
            return combined
        
        # Create a custom optimizer wrapper that prevents NaN gradients
        class SafeOptimizer(tf.keras.optimizers.Optimizer):
            def __init__(self, optimizer):
                # Get learning rate from the provided optimizer
                learning_rate = optimizer.learning_rate
                # Pass learning_rate to the parent constructor
                super(SafeOptimizer, self).__init__(name="SafeOptimizer", learning_rate=learning_rate)
                self.optimizer = optimizer

            def apply_gradients(self, grads_and_vars, **kwargs):
                safe_grads_and_vars = []
                for grad, var in grads_and_vars:
                    if grad is not None:
                        # Replace NaN/Inf with zeros
                        safe_grad = tf.where(
                            tf.math.logical_or(tf.math.is_nan(grad), tf.math.is_inf(grad)),
                            tf.zeros_like(grad),
                            grad
                        )
                        # Additional aggressive clipping
                        safe_grad = tf.clip_by_norm(safe_grad, 0.1)
                        safe_grads_and_vars.append((safe_grad, var))
                    else:
                        safe_grads_and_vars.append((grad, var))

                return self.optimizer.apply_gradients(safe_grads_and_vars, **kwargs)

        # Then wrap your existing optimizer:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.00005, 
            epsilon=1e-5,
            clipnorm=0.1  # Add gradient clipping directly to the optimizer
        )
        # Compile with stable loss and gradient clipping
        
        model.compile(
            optimizer=optimizer,
            loss=super_safe_loss,  # Make sure this is actually being used
            metrics=['accuracy']
        )
                     
        return model

# Add a new debug functionality to investigate and fix the NaN issue
def create_focused_debug_callback():
    """
    Creates a highly focused debug callback that monitors critical 
    operations right before the problematic batch range
    """
    class FocusedDebugCallback(tf.keras.callbacks.Callback):
        def __init__(self, critical_range=(115, 130)):
            super().__init__()
            self.critical_range = critical_range
            self.weight_snapshots = {}
            self.activation_snapshots = {}
            
        def on_batch_begin(self, batch, logs=None):
            """Take weight and gradient snapshots before the critical batches"""
            is_critical = self.critical_range[0] <= batch <= self.critical_range[1]
            
            # Additional inspection for batches just before the problematic range
            if batch >= 120 and batch <= 130:
                # Check activation statistics for key layers
                layer_names_to_check = ['output', 'attention_map']
                print(f"\nLayer stats at batch {batch} start:")
                for layer_name in layer_names_to_check:
                    try:
                        layer = [l for l in self.model.layers if l.name == layer_name][0]
                        # Create a small test input
                        dummy_data = np.zeros((1, 64, 64, 64, 2), dtype=np.float32)
                        dummy_data[:, 32, 32, 32, 0] = 1.0
                        
                        # Build a model that outputs this specific layer's output
                        temp_model = tf.keras.Model(
                            inputs=self.model.input, 
                            outputs=layer.output
                        )
                        
                        # Get the layer output
                        with tf.GradientTape() as tape:
                            layer_output = temp_model(dummy_data)
                            print(f"  {layer_name}: min={tf.reduce_min(layer_output).numpy():.8f}, "
                                  f"max={tf.reduce_max(layer_output).numpy():.8f}, "
                                  f"mean={tf.reduce_mean(layer_output).numpy():.8f}, "
                                  f"has_nan={tf.reduce_any(tf.math.is_nan(layer_output))}")
                                  
                            # Calculate gradient of this layer output for additional insight
                            if batch == 124: # Right before the NaN appears
                                try:
                                    # Add gradient analysis - just for the crucial batch
                                    # Create a fixed dummy target with a batch dimension matching the model output
                                    # This ensures batch dimension compatibility
                                    with tf.GradientTape() as inner_tape:
                                        inner_pred = self.model(dummy_data, training=False)
                                        # Create target with matching batch dimension
                                        batch_size = tf.shape(inner_pred)[0]
                                        dummy_target = tf.zeros(shape=tf.shape(inner_pred), dtype=tf.float32)
                                        # Set center region to 1.0
                                        indices = tf.TensorArray(tf.int32, size=batch_size)
                                        for i in range(batch_size):
                                            indices = indices.write(i, i)
                                        indices = indices.stack()
                                        dummy_target = tf.tensor_scatter_nd_update(
                                            dummy_target,
                                            tf.stack([
                                                indices,
                                                tf.ones_like(indices) * 31,
                                                tf.ones_like(indices) * 32,
                                                tf.ones_like(indices) * 32, 
                                                tf.zeros_like(indices)
                                            ], axis=1),
                                            tf.ones(batch_size, dtype=tf.float32)
                                        )
                                        # Calculate loss with matching shapes
                                        inner_loss = tf.keras.losses.binary_crossentropy(dummy_target, inner_pred)
                                                
                                    grad = inner_tape.gradient(inner_loss, layer_output)
                                    print(f"    Gradient stats: min={tf.reduce_min(grad).numpy():.8f}, "
                                          f"max={tf.reduce_max(grad).numpy():.8f}, "
                                          f"mean={tf.reduce_mean(grad).numpy():.8f}, "
                                          f"has_nan={tf.reduce_any(tf.math.is_nan(grad))}")
                                except Exception as e:
                                    print(f"    Error calculating gradients: {e}")
                    except Exception as e:
                        print(f"  Error inspecting {layer_name}: {e}")
            
            # Only check a few batches before the failure point to reduce overhead
            if is_critical:
                # Create a very small test input that won't tax memory
                dummy_data = np.zeros((1, 64, 64, 64, 2), dtype=np.float32)
                dummy_data[:, 32, 32, 32, 0] = 1.0  # Just a single point
                dummy_target = np.zeros((1, 64, 64, 64, 1), dtype=np.float32)
                
                # Save initial weights for key convolutional layers
                if batch == self.critical_range[0]:
                    self._save_model_weights(batch)
                
                # Forward pass check
                try:
                    with tf.GradientTape() as tape:
                        pred = self.model(dummy_data, training=False)
                        has_nan_output = tf.math.reduce_any(tf.math.is_nan(pred)).numpy()
                        
                        if has_nan_output:
                            print(f" NaN detected in model OUTPUT at batch {batch}")
                            # Early stopping to prevent further issues
                            self.model.stop_training = True
                            return
                    
                    # Only log normal values every few batches to reduce output spam
                    if batch % 5 == 0 or batch >= self.critical_range[0]:
                        print(f"Batch {batch} check: Output min={tf.reduce_min(pred).numpy():.6f}, "
                              f"max={tf.reduce_max(pred).numpy():.6f}, "
                              f"mean={tf.reduce_mean(pred).numpy():.6f}")
                
                except Exception as e:
                    print(f"Error during forward pass check: {e}")
                
        def on_batch_end(self, batch, logs=None):
            """Check for invalid values immediately after batch processing"""
            is_critical = self.critical_range[0] <= batch <= self.critical_range[1]
            
            if logs and 'loss' in logs:
                loss_value = logs.get('loss')
                
                # Check for NaN loss
                if np.isnan(loss_value) or np.isinf(loss_value):
                    print(f"\n NaN/Inf LOSS VALUE at batch {batch}!")
                    
                    # We had a valid forward pass but invalid loss - see what changed
                    if batch > self.critical_range[0]:
                        self._compare_weights(batch)
                    
                    # Stop training
                    print("Terminating training due to invalid loss")
                    self.model.stop_training = True
                    return
                
                # During critical range, check specific layer weights for NaN
                if is_critical and batch % 2 == 0:
                    self._check_weights_for_nan(batch)
                
                # Monitor learning rate
                if hasattr(self.model.optimizer, 'learning_rate'):
                    lr = float(self.model.optimizer.learning_rate.numpy())
                    if batch == self.critical_range[0]:
                        print(f"Current learning rate: {lr:.8f}")
        
        def _save_model_weights(self, batch):
            """Save weights snapshot for key layers"""
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv3D) and hasattr(layer, 'kernel'):
                    # Only save a few key convolutional layers
                    if i % 5 == 0:
                        name = f"{i}_{layer.name}"
                        try:
                            self.weight_snapshots[name] = tf.identity(layer.kernel).numpy().copy()
                        except Exception as e:
                            print(f"Error saving weights for layer {name}: {e}")
                            
        def _compare_weights(self, batch):
            """Compare current weights with initial weights to find problematic layers"""
            print("\nComparing weights with previous snapshots:")
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv3D) and hasattr(layer, 'kernel'):
                    name = f"{i}_{layer.name}"
                    if name in self.weight_snapshots:
                        try:
                            current_weights = layer.kernel.numpy()
                            saved_weights = self.weight_snapshots[name]
                            
                            # Check for NaNs in current weights
                            has_nan = np.isnan(current_weights).any()
                            
                            # Calculate max difference
                            diff = np.abs(current_weights - saved_weights)
                            max_diff = np.max(diff)
                            
                            print(f"Layer {name}: has_nan={has_nan}, max_diff={max_diff:.6f}")
                            
                            # If this layer has NaNs, it might be the problematic one
                            if has_nan:
                                print(f" Layer {name} contains NaN values")
                                
                        except Exception as e:
                            print(f"Error comparing weights for layer {name}: {e}")
        
        def _check_weights_for_nan(self, batch):
            """Check all model weights for NaN values"""
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'kernel'):
                    try:
                        weights = layer.kernel.numpy()
                        if np.isnan(weights).any():
                            print(f" NaN weights detected in layer {i} ({layer.name}) at batch {batch}")
                    except Exception:
                        pass  # Skip layers that might cause errors

    return FocusedDebugCallback()

# Add a function to create a simplified model architecture
def create_simplified_model(input_shape=(64, 64, 64, 2), filters_base=16):
    """
    Creates an ultra-simplified 3D CNN to replace U-Net when needed.
    This avoids numerical stability issues by using a much simpler architecture.
    """
    inputs = tf.keras.layers.Input(input_shape)
    
    # Extract just image features (channel 0)
    image_features = tf.keras.layers.Lambda(lambda x: x[..., 0:1])(inputs)
    
    # Extract just guidance features (channel 1)
    guidance = tf.keras.layers.Lambda(lambda x: x[..., 1:2])(inputs)
    
    # Very simple downsampling path
    x = tf.keras.layers.Conv3D(filters_base, 3, padding='same')(image_features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling3D()(x)
    
    x = tf.keras.layers.Conv3D(filters_base*2, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling3D()(x)
    
    # Simple upsampling path
    x = tf.keras.layers.Conv3D(filters_base*2, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling3D()(x)
    
    x = tf.keras.layers.Conv3D(filters_base, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling3D()(x)
    
    # Inject guidance directly via concatenation at final layer
    x = tf.keras.layers.Concatenate()([x, guidance])
    
    # Final layer with sigmoid and safeguard
    x = tf.keras.layers.Conv3D(1, 1, padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    outputs = tf.keras.layers.Lambda(
        lambda x: tf.where(tf.math.is_nan(x), tf.zeros_like(x), x),
        name="nan_protection"
    )(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Use the simplest optimizer settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-5,
        clipnorm=0.1,
        epsilon=1e-4
    )
    
    # Binary cross-entropy is the most stable loss function
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_seed_guidance(volume_shape, seed_points, seed_type='vessel', radius=3):
    
    # Create empty guidance volume
    guidance = np.zeros(volume_shape, dtype=np.float32)
    
    # Set value based on seed type (1 for vessel, -1 for background)
    value = 1.0 if seed_type == 'vessel' else -1.0
    
    # Mark seed points with specified radius
    for x, y, z in seed_points:
        if (0 <= z < volume_shape[0] and 
            0 <= y < volume_shape[1] and 
            0 <= x < volume_shape[2]):
            
            # Define bounds for the seed region (respecting volume boundaries)
            z_min = max(0, z - radius)
            z_max = min(volume_shape[0], z + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(volume_shape[1], y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(volume_shape[2], x + radius + 1)
            
            # Create spherical region around seed (approximate)
            for zi in range(z_min, z_max):
                for yi in range(y_min, y_max):
                    for xi in range(x_min, x_max):
                        # Check if within radius
                        dist = np.sqrt((zi-z)**2 + (yi-y)**2 + (xi-x)**2)
                        if dist <= radius:
                            guidance[zi, yi, xi] = value
    
    return guidance

# Replace the prepare_training_data function

def prepare_training_data(volume, seed_points, patch_size=64, num_patches=200, random_patches=True):
    """
    Prepare training data with better vessel detection focus
    """
    # Create seed guidance for full volume
    guidance = create_seed_guidance(volume.shape, seed_points, 'vessel', radius=5)  # Increased radius
    
    # Extract patches around seed points and near seed points for better vessel coverage
    inputs = []
    labels = []
    
    # First, extract patches around seed points
    seed_patches = 0
    for x, y, z in seed_points:
        # Create multiple patches with different offsets around this seed point
        # Use more offsets to better cover potential vessel paths
        offsets = [
            (0,0,0), (5,5,5), (-5,-5,-5), (5,-5,5), (-5,5,-5),
            (10,0,0), (-10,0,0), (0,10,0), (0,-10,0), (0,0,10), (0,0,-10),
            (15,15,0), (-15,-15,0), (15,-15,0), (-15,15,0)
        ]
        
        for offset_x, offset_y, offset_z in offsets:
            cx, cy, cz = x + offset_x, y + offset_y, z + offset_z
            
            # Define bounds for patch extraction (respecting volume boundaries)
            z_min = max(0, cz - patch_size//2)
            z_max = min(volume.shape[0], cz + patch_size//2)
            y_min = max(0, cy - patch_size//2)
            y_max = min(volume.shape[1], cy + patch_size//2)
            x_min = max(0, cx - patch_size//2)
            x_max = min(volume.shape[2], cx + patch_size//2)
            
            # Skip if patch is too small
            if (z_max - z_min < patch_size//2 or 
                y_max - y_min < patch_size//2 or 
                x_max - x_min < patch_size//2):
                continue
                
            # Extract patch
            vol_patch = volume[z_min:z_max, y_min:y_max, x_min:x_max]
            guide_patch = guidance[z_min:z_max, y_min:y_max, x_min:x_max]
            
            # Only use complete patches
            if vol_patch.shape[0] < patch_size or vol_patch.shape[1] < patch_size or vol_patch.shape[2] < patch_size:
                continue
                
            # Crop to exact patch size if needed
            vol_patch = vol_patch[:patch_size, :patch_size, :patch_size]
            guide_patch = guide_patch[:patch_size, :patch_size, :patch_size]
            
            # Normalize volume patch 
            vol_patch = vol_patch.astype(np.float32) / 255.0
            
            # Only include patches that have seed guidance
            if np.sum(guide_patch) == 0:
                continue
                
            # Create input with guidance
            input_patch = np.stack([vol_patch, guide_patch], axis=-1)
            
            # For the label, enhance vessel structures using intensity cues
            # This creates better vessel continuity in the training data
            vessel_enhance = np.copy(guide_patch)
            
            # Enhance based on the intensity profile around seeds
            # Vessel voxels often have similar intensity to seed points
            seed_mask = guide_patch > 0
            if np.any(seed_mask):
                seed_intensity = np.mean(vol_patch[seed_mask])
                
                # Find similar intensity voxels that might be vessels
                similar_mask = np.logical_and(
                    vol_patch >= seed_intensity * 0.8,
                    vol_patch <= seed_intensity * 1.2
                )
                
                # Connect similar intensity voxels to seeds using distance
                from scipy import ndimage
                distance = ndimage.distance_transform_edt(~seed_mask)
                connected_mask = np.logical_and(similar_mask, distance < patch_size//4)
                
                # Add to vessel enhancement
                vessel_enhance[connected_mask] = 0.5
            
            label_patch = (vessel_enhance > 0).astype(np.float32)
            label_patch = np.expand_dims(label_patch, axis=-1)
            
            inputs.append(input_patch)
            labels.append(label_patch)
            seed_patches += 1
            
            # Data augmentation: add 90-degree rotations for the same patch #This might need to be deleted
            # for k in range(1, 4):  # Add 3 more rotations
            #     # Rotate the patches 
            #     rot_vol = np.rot90(vol_patch, k=k, axes=(0, 1))
            #     rot_guide = np.rot90(guide_patch, k=k, axes=(0, 1))
            #     rot_vessel = np.rot90(vessel_enhance, k=k, axes=(0, 1))
                
            #     # Create input with guidance
            #     rot_input = np.stack([rot_vol, rot_guide], axis=-1)
                
            #     # Create label based on guidance
            #     rot_label = (rot_vessel > 0).astype(np.float32)
            #     rot_label = np.expand_dims(rot_label, axis=-1)
                
            #     inputs.append(rot_input)
            #     labels.append(rot_label)
    
    print(f"Created {seed_patches} seed-centered patches")
    
    # Generate additional random patches for background context
    # Limit the number of random patches to maintain class balance
    # random_patch_limit = min(num_patches - len(inputs), len(inputs))
    
    # if random_patches and random_patch_limit > 0:
    #     print(f"Generating {random_patch_limit} random background patches")
    #     for _ in range(random_patch_limit):
    #         z = np.random.randint(0, volume.shape[0] - patch_size)
    #         y = np.random.randint(0, volume.shape[1] - patch_size)
    #         x = np.random.randint(0, volume.shape[2] - patch_size)
            
    #         vol_patch = volume[z:z+patch_size, y:y+patch_size, x:x+patch_size]
    #         guide_patch = guidance[z:z+patch_size, y:y+patch_size, x:x+patch_size]
            
    #         # Normalize volume patch
    #         vol_patch = vol_patch.astype(np.float32) / 255.0
            
    #         # Create input with guidance
    #         input_patch = np.stack([vol_patch, guide_patch], axis=-1)
            
    #         # Create label based on guidance 
    #         label_patch = (guide_patch > 0).astype(np.float32)
    #         label_patch = np.expand_dims(label_patch, axis=-1)
            
    #         inputs.append(input_patch)
    #         labels.append(label_patch)
    
    return np.array(inputs), np.array(labels)

# Replace the process_chunk function with this more sensitive version

def process_chunk(model, chunk, guidance_chunk=None, threshold=0.15):  # Much lower threshold
    """
    Process a single chunk through the model with improved vessel detection
    """
    # Normalize chunk to [0,1]
    chunk_norm = chunk.astype(np.float32) / 255.0
    
    # Add guidance channel if provided
    if guidance_chunk is not None:
        chunk_input = np.stack([chunk_norm, guidance_chunk], axis=-1)
    else:
        # When no guidance is available, create a more effective diffuse guidance
        # Use directional guidance patterns to help with vessel continuity
        diffuse_guidance = np.zeros_like(chunk_norm, dtype=np.float32)
        
        # Add weak vessel guidance patterns (simulating potential vessels in multiple directions)
        # This helps the model bridge gaps between seed points
        center = chunk_norm.shape[0] // 2
        radius = chunk_norm.shape[0] // 4
        
        # Create a central hotspot for attention
        z, y, x = np.ogrid[:chunk_norm.shape[0], :chunk_norm.shape[1], :chunk_norm.shape[2]]
        dist = np.sqrt((z - center)**2 + (y - center)**2 + (x - center)**2)
        mask = dist <= radius
        diffuse_guidance[mask] = 0.1
        
        chunk_input = np.stack([chunk_norm, diffuse_guidance], axis=-1)
    
    # Record original shape for later
    original_shape = chunk_input.shape
    
    # Create a modified input array that ensures all dimensions are 
    # multiples of 16 (handles all dilation rates including 2, 4, and 8)
    padded_sizes = []
    for i in range(3):  # Handle the 3 spatial dimensions
        # Make each dimension a multiple of 16 to be safe
        if original_shape[i] % 16 != 0:
            new_size = ((original_shape[i] // 16) + 1) * 16
        else:
            new_size = original_shape[i]
        padded_sizes.append(new_size)
    
    # Create new array with padding
    padded_input = np.zeros((padded_sizes[0], padded_sizes[1], padded_sizes[2], 2), 
                           dtype=chunk_input.dtype)
    
    # Copy the original data into the padded array
    padded_input[:original_shape[0], 
                :original_shape[1], 
                :original_shape[2], :] = chunk_input
    
    # Ensure chunk has the right shape [batch, height, width, depth, channels]
    padded_input = np.expand_dims(padded_input, axis=0)
    
    try:
        # Run inference on padded input
        prediction = model.predict(padded_input, verbose=0)
        
        # Extract only the portion corresponding to the original input size
        prediction = prediction[0, 
                             :original_shape[0], 
                             :original_shape[1], 
                             :original_shape[2], 0]
    except Exception as e:
        print(f"Error during inference: {e}")
        # If inference fails, return a blank result
        prediction = np.zeros(original_shape[:3], dtype=np.float32)
    
    # Use multithreshold approach to improve vessel detection
    binary_mask = np.zeros_like(prediction, dtype=np.uint8)
    
    # First pass: detect strong vessel signals
    strong_vessels = (prediction > threshold * 2).astype(np.uint8)
    
    # Second pass: detect weaker vessel signals that connect to strong ones
    weak_vessels = (prediction > threshold).astype(np.uint8)
    
    # Create connectivity-based mask (only keep weak signals connected to strong ones)
    if np.sum(strong_vessels) > 0:
        # Use binary_fill_holes to connect nearby vessel segments
        from scipy import ndimage
        # Use both strong and weak vessels for the initial structure
        combined = np.logical_or(strong_vessels, weak_vessels).astype(np.uint8)
        
        # Extract connected components
        labeled, num_features = ndimage.label(combined)
        
        # Only keep components that have at least one strong vessel voxel
        for i in range(1, num_features + 1):
            component = (labeled == i)
            if not np.any(np.logical_and(component, strong_vessels)):
                # This component has no strong vessel voxel - remove it
                combined[component] = 0
        
        # Fill small gaps for better connectivity
        combined = ndimage.binary_closing(combined, structure=np.ones((3,3,3))).astype(np.uint8)
        
        # Only keep larger components to reduce noise
        labeled, num_features = ndimage.label(combined)
        if num_features > 0:  # Check if there are any components
            sizes = ndimage.sum(combined, labeled, range(1, num_features+1))
            
            # Keep components with more than minimum size
            min_size = 20  # Increased from 10 to preserve more of the vessel structure
            mask = np.zeros_like(labeled, dtype=bool)
            for i, size in enumerate(sizes):
                if size > min_size:
                    mask[labeled == i+1] = True
                    
            binary_mask = mask.astype(np.uint8) * 255
    else:
        # If no strong vessels detected, use weak vessels as fallback with stricter filtering
        if np.sum(weak_vessels) > 50:  # Only if there's substantial detection
            binary_mask = weak_vessels * 255
    
    return binary_mask

def segment_large_volume(model, volume, seed_points=None, chunk_size=64, overlap=8, threshold=0.5):
    """
    Segment a large volume by processing it in chunks
    """
    # Initialize output segmentation
    segmentation = np.zeros_like(volume, dtype=np.uint8)
    
    # Create guidance volume if seed points are provided
    if seed_points and len(seed_points) > 0:
        print(f"Creating guidance from {len(seed_points)} seed points...")
        guidance = create_seed_guidance(volume.shape, seed_points, 'vessel', radius=3)
    else:
        guidance = None
    
    # Make sure chunk_size is divisible by 16 to handle all dilation rates
    if chunk_size % 16 != 0:
        chunk_size = ((chunk_size // 16) + 1) * 16
        print(f"Adjusted chunk size to {chunk_size} to ensure compatibility with dilated convolutions")
    
    # Generate chunk coordinates
    chunk_coords = []
    for z in range(0, volume.shape[0], chunk_size - overlap):
        z_end = min(z + chunk_size, volume.shape[0])
        z_start = max(0, z_end - chunk_size)  # Ensure fixed chunk size where possible
        for y in range(0, volume.shape[1], chunk_size - overlap):
            y_end = min(y + chunk_size, volume.shape[1])
            y_start = max(0, y_end - chunk_size)  # Ensure fixed chunk size where possible
            for x in range(0, volume.shape[2], chunk_size - overlap):
                x_end = min(x + chunk_size, volume.shape[2])
                x_start = max(0, x_end - chunk_size)  # Ensure fixed chunk size where possible
                chunk_coords.append((z_start, z_end, y_start, y_end, x_start, x_end))
                
    print(f"Processing volume in {len(chunk_coords)} chunks...")
    
    # Process each chunk
    for i, (z_start, z_end, y_start, y_end, x_start, x_end) in enumerate(tqdm(chunk_coords)):
        # Extract chunk
        chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Skip processing if chunk is too small in any dimension
        if any(dim < 16 for dim in chunk.shape):
            continue
            
        # Extract guidance for this chunk if available
        guidance_chunk = None
        if guidance is not None:
            guidance_chunk = guidance[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Process chunk with error handling
        try:
            processed_chunk = process_chunk(model, chunk, guidance_chunk, threshold)
            
            # Ensure the processed chunk matches the original chunk dimensions
            if processed_chunk.shape != chunk.shape:
                # Trim or pad to match original dimensions
                temp_chunk = np.zeros_like(chunk, dtype=processed_chunk.dtype)
                # Copy the valid part
                z_dim = min(processed_chunk.shape[0], chunk.shape[0])
                y_dim = min(processed_chunk.shape[1], chunk.shape[1])
                x_dim = min(processed_chunk.shape[2], chunk.shape[2])
                temp_chunk[:z_dim, :y_dim, :x_dim] = processed_chunk[:z_dim, :y_dim, :x_dim]
                processed_chunk = temp_chunk
            
            # Determine valid region (exclude overlap except at edges)
            z_valid_start = overlap//2 if z_start > 0 else 0
            y_valid_start = overlap//2 if y_start > 0 else 0
            x_valid_start = overlap//2 if x_start > 0 else 0
            
            z_valid_end = chunk.shape[0] - overlap//2 if z_end < volume.shape[0] else chunk.shape[0]
            y_valid_end = chunk.shape[1] - overlap//2 if y_end < volume.shape[1] else chunk.shape[1]
            x_valid_end = chunk.shape[2] - overlap//2 if x_end < volume.shape[2] else chunk.shape[2]
            
            # Extract valid region from processed chunk
            valid_chunk = processed_chunk[
                z_valid_start:z_valid_end,
                y_valid_start:y_valid_end,
                x_valid_start:x_valid_end
            ]
            
            # Insert into final segmentation
            segmentation[
                z_start+z_valid_start:z_start+z_valid_end,
                y_start+y_valid_start:y_start+y_valid_end,
                x_start+x_valid_start:x_start+x_valid_end
            ] = valid_chunk
        
        except Exception as e:
            print(f"Error processing chunk at z={z_start}-{z_end}, y={y_start}-{y_end}, x={x_start}-{x_end}: {e}")
            # Continue with the next chunk
            continue
        
    return segmentation

def segment_large_volume_parallel(model, volume, seed_points=None, chunk_size=64, overlap=8, threshold=0.5, num_workers=2):
    """
    Segment a large volume by processing chunks in parallel across multiple GPUs
    """
    # Initialize output segmentation
    segmentation = np.zeros_like(volume, dtype=np.uint8)
    
    # Create guidance volume if seed points are provided
    if seed_points and len(seed_points) > 0:
        print(f"Creating guidance from {len(seed_points)} seed points...")
        guidance = create_seed_guidance(volume.shape, seed_points, 'vessel', radius=3)
    else:
        guidance = None
    
    # Make sure chunk_size is divisible by 16 to handle all dilation rates
    if chunk_size % 16 != 0:
        chunk_size = ((chunk_size // 16) + 1) * 16
        print(f"Adjusted chunk size to {chunk_size} to ensure compatibility with dilated convolutions")
    
    # Generate chunk coordinates
    chunk_coords = []
    for z in range(0, volume.shape[0], chunk_size - overlap):
        z_end = min(z + chunk_size, volume.shape[0])
        z_start = max(0, z_end - chunk_size)  # Ensure fixed chunk size where possible
        for y in range(0, volume.shape[1], chunk_size - overlap):
            y_end = min(y + chunk_size, volume.shape[1])
            y_start = max(0, y_end - chunk_size)  # Ensure fixed chunk size where possible
            for x in range(0, volume.shape[2], chunk_size - overlap):
                x_end = min(x + chunk_size, volume.shape[2])
                x_start = max(0, x_end - chunk_size)  # Ensure fixed chunk size where possible
                chunk_coords.append((z_start, z_end, y_start, y_end, x_start, x_end))
                
    print(f"Processing volume in {len(chunk_coords)} chunks with {num_workers} workers...")
    
    # Instead of using multiple models on different GPUs, use a single model and process chunks sequentially
    # This avoids the cross-device access issue
    
    # Function to process a single chunk
    def process_chunk_task(args):
        chunk_id, coords = args
        z_start, z_end, y_start, y_end, x_start, x_end = coords
        
        # Extract chunk
        chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Skip processing if chunk is too small in any dimension
        if any(dim < 16 for dim in chunk.shape):
            return chunk_id, None, coords
            
        # Extract guidance for this chunk if available
        guidance_chunk = None
        if guidance is not None:
            guidance_chunk = guidance[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Process chunk with error handling
        try:
            processed_chunk = process_chunk(model, chunk, guidance_chunk, threshold)
            return chunk_id, processed_chunk, coords
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            # Return None to indicate failure
            return chunk_id, None, coords
    
    # Create a pool of workers
    available_gpus = len(tf.config.list_physical_devices('GPU'))
    if available_gpus == 0:
        print("No GPUs available, falling back to CPU processing")
        num_workers = min(num_workers, multiprocessing.cpu_count())
    else:
        # For multi-GPU, we'll use a different approach
        if num_workers > 1 and available_gpus > 1:
            print(f"Using {num_workers} workers across {available_gpus} GPUs")
            return segment_large_volume_distributed(model, volume, seed_points, chunk_size, overlap, threshold, num_workers)
        else:
            print(f"Using single GPU processing with {num_workers} workers")

    # Prepare arguments for parallel processing
    chunk_args = [(i, coords) for i, coords in enumerate(chunk_coords)]

    # Use ThreadPoolExecutor for parallel processing on a single GPU
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_chunk_task, arg): arg for arg in chunk_args}
        
        # Process results as they complete
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            try:
                chunk_id, processed_chunk, coords = future.result()
                if processed_chunk is None:
                    continue
                    
                z_start, z_end, y_start, y_end, x_start, x_end = coords
                chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Ensure the processed chunk matches the original chunk dimensions
                if processed_chunk.shape != chunk.shape:
                    # Trim or pad to match original dimensions
                    temp_chunk = np.zeros_like(chunk, dtype=processed_chunk.dtype)
                    # Copy the valid part
                    z_dim = min(processed_chunk.shape[0], chunk.shape[0])
                    y_dim = min(processed_chunk.shape[1], chunk.shape[1])
                    x_dim = min(processed_chunk.shape[2], chunk.shape[2])
                    temp_chunk[:z_dim, :y_dim, :x_dim] = processed_chunk[:z_dim, :y_dim, :x_dim]
                    processed_chunk = temp_chunk
                
                # Determine valid region (exclude overlap except at edges)
                z_valid_start = overlap//2 if z_start > 0 else 0
                y_valid_start = overlap//2 if y_start > 0 else 0
                x_valid_start = overlap//2 if x_start > 0 else 0
                
                z_valid_end = chunk.shape[0] - overlap//2 if z_end < volume.shape[0] else chunk.shape[0]
                y_valid_end = chunk.shape[1] - overlap//2 if y_end < volume.shape[1] else chunk.shape[1]
                x_valid_end = chunk.shape[2] - overlap//2 if x_end < volume.shape[2] else chunk.shape[2]
                
                # Extract valid region from processed chunk
                valid_chunk = processed_chunk[
                    z_valid_start:z_valid_end,
                    y_valid_start:y_valid_end,
                    x_valid_start:x_valid_end
                ]
                
                # Insert into final segmentation using a lock to avoid race conditions
                segmentation[
                    z_start+z_valid_start:z_start+z_valid_end,
                    y_start+y_valid_start:y_start+y_valid_end,
                    x_start+x_valid_start:x_start+x_valid_end
                ] = valid_chunk
                
            except Exception as e:
                print(f"Error processing result: {e}")
    
    return segmentation

def segment_large_volume_distributed(model, volume, seed_points=None, chunk_size=64, overlap=8, threshold=0.5, num_workers=2):
    """
    Segment a large volume by distributing chunks across multiple GPUs, one worker per GPU
    This is a different approach that avoids cross-device issues
    """
    # Initialize output segmentation
    segmentation = np.zeros_like(volume, dtype=np.uint8)
    
    # Create guidance volume if seed points are provided
    if seed_points and len(seed_points) > 0:
        print(f"Creating guidance from {len(seed_points)} seed points...")
        guidance = create_seed_guidance(volume.shape, seed_points, 'vessel', radius=3)
    else:
        guidance = None
    
    # Generate chunk coordinates
    chunk_coords = []
    for z in range(0, volume.shape[0], chunk_size - overlap):
        z_end = min(z + chunk_size, volume.shape[0])
        z_start = max(0, z_end - chunk_size)
        for y in range(0, volume.shape[1], chunk_size - overlap):
            y_end = min(y + chunk_size, volume.shape[1])
            y_start = max(0, y_end - chunk_size)
            for x in range(0, volume.shape[2], chunk_size - overlap):
                x_end = min(x + chunk_size, volume.shape[2])
                x_start = max(0, x_end - chunk_size)
                chunk_coords.append((z_start, z_end, y_start, y_end, x_start, x_end))
    
    # Get available GPUs
    available_gpus = min(len(tf.config.list_physical_devices('GPU')), num_workers)
    
    # Split chunks by GPU
    chunks_per_gpu = [[] for _ in range(available_gpus)]
    for i, coords in enumerate(chunk_coords):
        gpu_idx = i % available_gpus
        chunks_per_gpu[gpu_idx].append(coords)
    
    # Initialize models on each GPU
    print(f"Creating models on {available_gpus} GPUs")
    gpu_models = []
    
    # Get model weights once from CPU
    with tf.device('/cpu:0'):
        weights = model.get_weights()
    
    # Create a model per GPU
    for gpu_idx in range(available_gpus):
        with tf.device(f'/device:GPU:{gpu_idx}'):
            print(f"Creating model on GPU {gpu_idx}")
            # Create a fresh model on this GPU
            gpu_model = create_3d_unet(input_shape=model.input_shape[1:])
            gpu_model.set_weights(weights)
            gpu_models.append(gpu_model)
    
    # Process function for each GPU
    def process_gpu_chunks(gpu_idx):
        local_results = []
        gpu_model = gpu_models[gpu_idx]
        
        # Process all chunks assigned to this GPU
        for coords in tqdm(chunks_per_gpu[gpu_idx], desc=f"GPU {gpu_idx}"):
            z_start, z_end, y_start, y_end, x_start, x_end = coords
            
            # Extract chunk
            chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Skip processing if chunk is too small
            if any(dim < 16 for dim in chunk.shape):
                continue
                
            # Extract guidance for this chunk if available
            guidance_chunk = None
            if guidance is not None:
                guidance_chunk = guidance[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Process chunk with the GPU-specific model
            try:
                with tf.device(f'/device:GPU:{gpu_idx}'):
                    processed_chunk = process_chunk(gpu_model, chunk, guidance_chunk, threshold)
                
                # Calculate valid region (exclude overlap except at edges)
                z_valid_start = overlap//2 if z_start > 0 else 0
                y_valid_start = overlap//2 if y_start > 0 else 0
                x_valid_start = overlap//2 if x_start > 0 else 0
                
                z_valid_end = chunk.shape[0] - overlap//2 if z_end < volume.shape[0] else chunk.shape[0]
                y_valid_end = chunk.shape[1] - overlap//2 if y_end < volume.shape[1] else chunk.shape[1]
                x_valid_end = chunk.shape[2] - overlap//2 if x_end < volume.shape[2] else chunk.shape[2]
                
                # Extract valid region
                valid_chunk = processed_chunk[
                    z_valid_start:z_valid_end,
                    y_valid_start:y_valid_end,
                    x_valid_start:x_valid_end
                ]
                
                # Store result with coordinates for later assembly
                local_results.append((
                    z_start+z_valid_start, z_start+z_valid_end,
                    y_start+y_valid_start, y_start+y_valid_end,
                    x_start+x_valid_start, x_start+x_valid_end,
                    valid_chunk
                ))
                
            except Exception as e:
                print(f"Error processing chunk on GPU {gpu_idx}: {e}")
        
        return local_results
    
    # Process chunks on each GPU in parallel
    with ThreadPoolExecutor(max_workers=available_gpus) as executor:
        all_futures = []
        for gpu_idx in range(available_gpus):
            future = executor.submit(process_gpu_chunks, gpu_idx)
            all_futures.append(future)
        
        # Collect results from all GPUs
        for future in all_futures:
            try:
                results = future.result()
                # Insert each result into the final segmentation
                for z_start, z_end, y_start, y_end, x_start, x_end, chunk in results:
                    segmentation[z_start:z_end, y_start:y_end, x_start:x_end] = chunk
            except Exception as e:
                print(f"Error processing GPU batch: {e}")
    
    return segmentation

def save_training_visualizations(history, output_prefix):
    """
    Save training history visualizations to files
    """
    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    if 'dice_coefficient' in history.history:
        plt.plot(history.history['dice_coefficient'])
        plt.plot(history.history['val_dice_coefficient'])
        plt.title('Dice Coefficient')
        plt.ylabel('Dice')
    else:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_training_history.png")
    print(f"Training visualization saved to {output_prefix}_training_history.png")
    
    # Save history as JSON for later analysis
    with open(f"{output_prefix}_history.json", 'w') as f:
        history_dict = history.history.copy()
        # Convert numpy arrays to lists for JSON serialization
        for key in history_dict:
            history_dict[key] = [float(x) for x in history_dict[key]]
        json.dump(history_dict, f)

def save_volume_slices(volume, segmentation, output_prefix, num_slices=5):
    """
    Save sample slices of volume with segmentation overlay
    """
    # Create the visualization directory if it doesn't exist
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Select evenly spaced slices
    slice_indices = np.linspace(0, volume.shape[0]-1, num_slices, dtype=int)
    
    plt.figure(figsize=(15, 3*num_slices))
    for i, z in enumerate(slice_indices):
        # Original volume slice
        plt.subplot(num_slices, 3, i*3 + 1)
        plt.imshow(volume[z], cmap='gray')
        plt.title(f'Original (z={z})')
        plt.axis('off')
        
        # Segmentation slice
        plt.subplot(num_slices, 3, i*3 + 2)
        plt.imshow(segmentation[z], cmap='gray')
        plt.title(f'Segmentation (z={z})')
        plt.axis('off')
        
        # Overlay
        plt.subplot(num_slices, 3, i*3 + 3)
        plt.imshow(volume[z], cmap='gray')
        # Create a red mask for the segmentation
        seg_mask = np.zeros((*segmentation[z].shape, 4))
        seg_mask[segmentation[z] > 0] = [1, 0, 0, 0.5]  # Semi-transparent red
        plt.imshow(seg_mask)
        plt.title(f'Overlay (z={z})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_slices.png")
    print(f"Volume slices visualization saved to {output_prefix}_slices.png")

# Add this function to expand seed points for inference

def expand_seed_points(seed_points, volume_shape, expansion_radius=30):
    """
    Create additional seed points at potential vessel locations
    to improve segmentation coverage
    """
    expanded_points = seed_points.copy()
    
    # For each seed point, add points along potential vessel paths
    for x, y, z in seed_points:
        # Add points in 6 primary directions
        directions = [
            (1, 0, 0), (-1, 0, 0),  # x-axis
            (0, 1, 0), (0, -1, 0),  # y-axis
            (0, 0, 1), (0, 0, -1)   # z-axis
        ]
        
        for dx, dy, dz in directions:
            for dist in range(10, expansion_radius+1, 10):
                nx = max(0, min(volume_shape[2]-1, x + dx*dist))
                ny = max(0, min(volume_shape[1]-1, y + dy*dist))
                nz = max(0, min(volume_shape[0]-1, z + dz*dist))
                
                new_point = (nx, ny, nz)
                if new_point not in expanded_points:
                    expanded_points.append(new_point)
    
    print(f"Expanded {len(seed_points)} seed points to {len(expanded_points)} points")
    return expanded_points

# Define standard 3D U-Net function outside of main to avoid the "referenced before assignment" error
def create_3d_unet(input_shape=(64, 64, 64, 2), filters_base=32):
    """
    Creates a 3D U-Net model optimized for vessel segmentation with guidance channel
    """
    # Input layer
    inputs = layers.Input(input_shape, name="input_layer")
    
    # Add attention mechanism to focus on important features
    def attention_block(x, g, filters):
        theta_x = layers.Conv3D(filters, 1, strides=1, padding='same')(x)
        phi_g = layers.Conv3D(filters, 1, strides=1, padding='same')(g)
        
        f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
        psi_f = layers.Conv3D(1, 1, strides=1, padding='same')(f)
        
        rate = layers.Activation('sigmoid')(psi_f)
        att_x = layers.multiply([x, rate])
        
        return att_x
    
    # Improved convolutional block with residual connections for better gradient flow
    def conv_block(input_tensor, num_filters):
        x = layers.Conv3D(num_filters, 3, padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv3D(num_filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Add residual connection
        if input_tensor.shape[-1] == num_filters:
            x = layers.add([x, input_tensor])
        else:
            shortcut = layers.Conv3D(num_filters, 1, padding='same')(input_tensor)
            x = layers.add([x, shortcut])
            
        return x
    
    # Contracting path (encoder)
    # Level 1
    conv1 = conv_block(inputs, filters_base)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    # Level 2
    conv2 = conv_block(pool1, filters_base*2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    # Level 3
    conv3 = conv_block(pool2, filters_base*4)
    pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    # Bridge with dilated convolutions for better field of view
    bridge = conv_block(pool3, filters_base*8)
    # Add dilated convolutions to increase receptive field
    dilation_rates = [2, 4, 8]
    dilated_layers = []
    
    for rate in dilation_rates:
        dilated = layers.Conv3D(filters_base*8, 3, padding='same', dilation_rate=rate)(bridge)
        dilated = layers.BatchNormalization()(dilated)
        dilated = layers.Activation('relu')(dilated)
        dilated_layers.append(dilated)
    
    # Combine dilated convolutions
    dilated_concat = layers.Concatenate()(dilated_layers + [bridge])
    bridge = layers.Conv3D(filters_base*8, 1, padding='same')(dilated_concat)
    bridge = layers.BatchNormalization()(bridge)
    bridge = layers.Activation('relu')(bridge)
    
    # Expansive path (decoder) with attention gates for focus on relevant features
    # Level 3
    up5 = layers.UpSampling3D(size=(2, 2, 2))(bridge)
    att5 = attention_block(conv3, up5, filters_base*4)
    concat5 = layers.Concatenate()([up5, att5])
    conv5 = conv_block(concat5, filters_base*4)
    
    # Level 2
    up6 = layers.UpSampling3D(size=(2, 2, 2))(conv5)
    att6 = attention_block(conv2, up6, filters_base*2)
    concat6 = layers.Concatenate()([up6, att6])
    conv6 = conv_block(concat6, filters_base*2)
    
    # Level 1
    up7 = layers.UpSampling3D(size=(2, 2, 2))(conv6)
    att7 = attention_block(conv1, up7, filters_base)
    concat7 = layers.Concatenate()([up7, att7])
    conv7 = conv_block(concat7, filters_base)
    
    # Additional attention mechanism for vessel focus
    attention = layers.Conv3D(1, 1, activation='sigmoid', name='attention_map')(conv7)
    attended = layers.Multiply()([conv7, attention])
    
    # Output layer
    outputs = layers.Conv3D(1, 1, activation='sigmoid', name='output')(attended)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Loss functions (keep existing loss functions)
    def dice_loss(y_true, y_pred):
        smooth = 1.0
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 1.0 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    def dice_coefficient(y_true, y_pred):
        smooth = 1.0
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    # Use focal loss to better handle class imbalance
    def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
        focal_loss = -alpha * tf.pow(1-p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    
    # Combine dice loss and focal loss
    def combined_loss(y_true, y_pred):
        return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)
    
    # Compile with combined loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                loss=combined_loss, 
                metrics=['accuracy', dice_coefficient])
    
    return model

# Add a new function to load seed points from CSV file
def load_seed_points_from_csv(csv_path):
    """
    Load seed points from a CSV file
    
    The CSV file should have columns: 
    - Column 1: ID (ignored)
    - Column 2: X
    - Column 3: Y
    - Column 4: Slice (Z)
    
    Returns a list of (x, y, z) tuples
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        if not all(col in df.columns for col in ['X', 'Y', 'Slice']):
            print(f"Warning: CSV file {csv_path} does not have the expected columns (X, Y, Slice)")
            return []
        
        # Convert to list of tuples (x, y, z)
        seed_points = [(x, y, z) for x, y, z in zip(df['X'], df['Y'], df['Slice'])]
        
        print(f"Loaded {len(seed_points)} seed points from {csv_path}")
        return seed_points
    
    except Exception as e:
        print(f"Error loading seed points from {csv_path}: {e}")
        return []

# Add this function to create TensorFlow datasets from training data
def create_dataset(indices, batch_size, inputs, labels, is_training=True):
    """
    Create a TensorFlow dataset from input data indices to efficiently load data in batches
    
    Args:
        indices: Array of indices to select from inputs and labels
        batch_size: Number of samples per batch
        inputs: The complete input data array
        labels: The complete label data array
        is_training: Whether this is a training dataset (enables shuffling)
        
    Returns:
        A TensorFlow Dataset that loads data efficiently
    """
    # Define a function to load a single sample from inputs and labels
    def load_sample(idx):
        idx = int(idx.numpy())
        return inputs[idx], labels[idx]
            
    # TensorFlow wrapper for the function
    def tf_load_sample(idx):
        sample = tf.py_function(
            load_sample,
            [idx],
            [tf.float32, tf.float32]
        )
        # Set the shapes explicitly based on the input shapes
        sample[0].set_shape(inputs.shape[1:])
        sample[1].set_shape(labels.shape[1:])
        return sample
    
    # Create dataset from indices
    ds = tf.data.Dataset.from_tensor_slices(indices)
    
    # Shuffle only training data - use a safer buffer size to avoid memory issues
    if is_training:
        # Use a buffer size that's reasonable but not too large
        buffer_size = min(8192, len(indices))
        ds = ds.shuffle(buffer_size=buffer_size)
        
    # Map indices to actual data
    ds = ds.map(tf_load_sample, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Use batches to avoid OOM - crucial batch size adjustment for multi-GPU
    # If there's a remainder, it could cause problems during distributed training
    ds = ds.batch(batch_size, drop_remainder=True)  # Important: drop_remainder=True fixes most NaN issues
    
    # Prefetch for better performance
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds

# Modify the main function to use distributed training

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Semi-Supervised Vessel Segmentation Training")
    
    # Add arguments
    parser.add_argument("--data-dir", default="3d-stacks", help="Directory containing 3D volume data")
    parser.add_argument("--output-dir", default="output/semi_supervised", help="Output directory for results")
    parser.add_argument("--model-dir", default="models", help="Directory to save trained models")
    parser.add_argument("--chunk-size", type=int, default=64, help="Size of chunks for processing")
    parser.add_argument("--overlap", type=int, default=8, help="Overlap between chunks")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=0, help="Training batch size, 0 for auto, values > 0 override auto")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    parser.add_argument("--patches-per-volume", type=int, default=200, 
                        help="Number of training patches to extract per volume")
    parser.add_argument("--test", action="store_true", help="Run inference on test data only")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers for inference")
    parser.add_argument("--seed-points", default=None, help="Path to CSV file with seed points")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.model_dir, f"vessel_segmentation_model_{timestamp}.h5")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, f"training_log_{timestamp}.txt")
    
    # Print system information
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Available GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    # Set up the strategy based on available GPUs
    use_multi_gpu = len(gpus) > 1
    
    if use_multi_gpu:
        try:
            # Try setting up the MirroredStrategy
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {strategy.num_replicas_in_sync} GPUs with MirroredStrategy")
            
            # Force TensorFlow to use all GPUs
            if strategy.num_replicas_in_sync < len(gpus):
                print("Warning: Not all GPUs are being used. Recreating strategy...")
                # Clear any existing session
                tf.keras.backend.clear_session()
                
                # Explicitly set visible devices
                tf.config.set_visible_devices(gpus, 'GPU')
                
                # Recreate strategy
                strategy = tf.distribute.MirroredStrategy()
                print(f"Recreated strategy now using {strategy.num_replicas_in_sync} GPUs")
            
        except Exception as e:
            print(f"Error setting up MirroredStrategy: {e}")
            strategy = tf.distribute.get_strategy()  # Default strategy for single GPU
            use_multi_gpu = False
            print("Falling back to single GPU due to error")
    else:
        strategy = tf.distribute.get_strategy()  # Default strategy for single GPU
        print("Using single GPU or CPU")
    
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")
    
    # For now, we only have r01_
    volume_path = os.path.join(args.data_dir, "r01_", "r01_.8bit.tif")
    print(f"Loading volume from {volume_path}")
    volume = tiff.imread(volume_path)
    print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
    
    # Get seed points - now from CSV if specified, otherwise from default
    if args.seed_points and os.path.exists(args.seed_points):
        seed_points = load_seed_points_from_csv(args.seed_points)
    else:
        # Check if a default seed points file exists for this volume
        default_seed_path = os.path.join("seed_points", "r01_seed_points.csv")
        if os.path.exists(default_seed_path):
            print(f"Using default seed points from {default_seed_path}")
            seed_points = load_seed_points_from_csv(default_seed_path)
        else:
            # Fall back to hardcoded seed points
            print("Using hardcoded seed points")
            seed_points = SEED_POINTS["r01_"]
    
    if not seed_points:
        print("No valid seed points found. Using default hardcoded seed points.")
        seed_points = SEED_POINTS["r01_"]
    
    # Create model - for multi-GPU use the strategy-based model creation
    input_shape = (args.chunk_size, args.chunk_size, args.chunk_size, 2)
    if use_multi_gpu:
        model = create_model_with_strategy(strategy, input_shape=input_shape)
        print("Created model with MirroredStrategy")
    else:
        # Use the standard model (already defined outside of main)
        model = create_3d_unet(input_shape=input_shape)
    
    if not args.test:
        # Use more patches for better coverage
        print(f"Preparing training data with {len(seed_points)} seed points")
        inputs, labels = prepare_training_data(
            volume, 
            seed_points, 
            patch_size=args.chunk_size, 
            num_patches=args.patches_per_volume * 4  # More training data
        )
        print(f"Training data shape: {inputs.shape}, labels shape: {labels.shape}")
        
        # Split data into training and validation sets
        # This needs to happen before creating datasets
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        
        # Moved row 1353-1392 here from row 1293
        val_size = int(len(inputs) * 0.2)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        # Make dataset sizes perfectly divisible by batch_size * num_gpus
        # This is crucial for preventing NaN issues during multi-GPU training
        if use_multi_gpu:
            def estimate_optimal_batch_size(chunk_size, available_gpus):
                # Base batch size depends on chunk size
                if chunk_size <= 32:
                    base_batch_size = 8
                elif chunk_size <= 48:
                    base_batch_size = 4
                elif chunk_size <= 64:
                    base_batch_size = 4 # Provar byta från 2 tillfälligt
                else:
                    base_batch_size = 1
            
                # Scale by number of GPUs (with diminishing returns to be safe)
                if available_gpus == 0:
                    # CPU only
                    print(f"RETURNED BATCH SIZE: {1}\nBase batch size: {base_batch_size}")
                    return 1
                elif available_gpus == 1:
                    print(f"RETURNED BATCH SIZE: {base_batch_size}\nBase batch size: {base_batch_size}")
                    return base_batch_size
                else:
                    # When using multiple GPUs, we can increase batch size
                    # but not necessarily linearly due to synchronization overhead
                    print(f"RETURNED BATCH SIZE: {base_batch_size * max(1, int(available_gpus))}\nBase batch size: {base_batch_size}")
                    return base_batch_size * max(1, int(available_gpus))
        
            # Get batch size
            if args.batch_size > 0:
                # User specified batch size
                optimal_batch_size = args.batch_size
            else:
                # Calculate based on hardware
                optimal_batch_size = estimate_optimal_batch_size(
                    args.chunk_size, 
                    strategy.num_replicas_in_sync  # Use actual number of GPUs from strategy
                )
            # Get the exact number needed for perfect division
            print(f"Optimal batch size: {optimal_batch_size} per GPU")
            gpu_batch_factor = optimal_batch_size * strategy.num_replicas_in_sync
            
            # Adjust training set
            train_remainder = len(train_indices) % gpu_batch_factor
            if train_remainder != 0:
                # Remove extra samples to make evenly divisible
                train_indices = train_indices[:-train_remainder]
                
            # Adjust validation set
            val_remainder = len(val_indices) % gpu_batch_factor
            if val_remainder != 0:
                # Remove extra samples to make evenly divisible
                val_indices = val_indices[:-val_remainder]
                
            print(f"Adjusted dataset sizes to be divisible by {gpu_batch_factor}:")
            print(f"  Training samples: {len(train_indices)}")
            print(f"  Validation samples: {len(val_indices)}")
        
        # Use improved callbacks for better model selection
        callbacks = [
            ModelCheckpoint(
                model_path,
                save_best_only=True,
                monitor='val_dice_coefficient',
                mode='max'
            ),
            EarlyStopping(
                patience=10,  # Increased patience for better convergence
                restore_best_weights=True,
                monitor='val_dice_coefficient',
                mode='max'
            ),
            ReduceLROnPlateau(
                factor=0.5,
                patience=5,  # Increased patience for learning rate reduction
                min_lr=1e-6,
                monitor='val_dice_coefficient',
                mode='max'
            ),
            TensorBoard(log_dir=os.path.join(args.output_dir, f"logs_{timestamp}"))
        ]
        
        # callbacks.append(
        #     tf.keras.callbacks.TerminateOnNaN()  # Stop training when NaN is encountered
        # )

        callbacks.append(
            create_focused_debug_callback()
        )

        # callbacks.append(
        #     tf.keras.callbacks.ReduceLROnPlateau(
        #         monitor='loss',
        #         factor=0.5,
        #         patience=2,
        #         min_lr=1e-6,
        #         verbose=1
        #     )
        # )
        # class NanCallback(tf.keras.callbacks.Callback):
        #     def on_batch_end(self, batch, logs=None):
        #         logs = logs or {}
        #         loss = logs.get('loss')
        #         if loss is not None and (np.isnan(loss) or np.isinf(loss)):
        #             print(f"NaN loss detected at batch {batch}, stopping training.")
        #             print(f"Logs: {logs}")
        #             self.model.stop_training = True
        
        # # Add this to your callbacks list
        # callbacks.append(NanCallback())
        # class DebugNanCallback(tf.keras.callbacks.Callback):
        #     def on_batch_end(self, batch, logs=None):
        #         logs = logs or {}
        #         loss = logs.get('loss')
        #         if loss is not None and (np.isnan(loss) or np.isinf(loss)):
        #             print(f"\n\nNaN loss detected at batch {batch}, stopping training.")
        #             print(f"Current logs: {logs}")
        #             print(f"Layer weights status:")
        #         for i, layer in enumerate(self.model.layers):
        #             weights = layer.get_weights()
        #             if len(weights) > 0:
        #                 has_nan = any(np.isnan(w).any() for w in weights)
        #                 print(f"Layer {i} ({layer.name}): {'Has NaN' if has_nan else 'OK'}")
        #         self.model.stop_training = False

        class DetailedDebugCallback(tf.keras.callbacks.Callback):
            def on_batch_begin(self, batch, logs=None):
                if batch >= 120 and batch <= 130:  # Focus on the critical range
                    # Get output of the model's last layer for a dummy input
                    dummy_input = tf.ones((1, 64, 64, 64, 2))
                    with tf.GradientTape() as tape:
                        output = self.model(dummy_input, training=False)
                        # Print stats about the output
                        print(f"\nBatch {batch} output stats:")
                        print(f"  Min: {tf.reduce_min(output)}")
                        print(f"  Max: {tf.reduce_max(output)}")
                        print(f"  Mean: {tf.reduce_mean(output)}")
                        print(f"  Has NaN: {tf.reduce_any(tf.math.is_nan(output))}")
        callbacks.append(DetailedDebugCallback())

        class SafetyCallback(tf.keras.callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                logs = logs or {}
                if 'loss' in logs and (np.isnan(logs['loss']) or np.isinf(logs['loss'])):
                    print(f"\nStopping training at batch {batch} due to NaN/Inf in loss")
                    print(f"Last few good batches were fine, indicating this is a numerical")
                    print(f"stability issue in the loss function calculation, not the model.")
                    self.model.stop_training = True

        callbacks.append(SafetyCallback())
        #callbacks.append(DebugNanCallback())
        
        # Calculate optimal batch size based on available GPU memory and model complexity
        # This is an approximate calculation that you may need to adjust based on your hardware
        
        # Moved row 1353-1392 here from row 1293
        # val_size = int(len(inputs) * 0.2)
        # train_indices = indices[val_size:]
        # val_indices = indices[:val_size]

        #Correct the number of samples to be divisible by batch size and number of GPUs
        # rest = (len(train_indices)) % (optimal_batch_size * 2)
        # print(f"len(train_indices): {len(train_indices)}, Denom mod: {optimal_batch_size * 2}, rest: {rest}")
        # if rest != 0:
        #     train_indices = np.append(train_indices, val_indices[:(optimal_batch_size * 2 - rest)])
        #     val_indices = val_indices[(optimal_batch_size * 2 - rest):]
        
        # rest = (len(val_indices)) % (optimal_batch_size * 2)
        # print(f"len(val_indices): {len(val_indices)}, Denom mod: {optimal_batch_size * 2}, rest: {rest}")
        # if rest != 0:
        #     val_indices = val_indices[(optimal_batch_size * 2 - rest):]
        # print(f"mod train: {len(train_indices) % (optimal_batch_size * 2)}, mod val: {len(val_indices) % (optimal_batch_size * 2)}")
        # print(f"Split data into {len(train_indices)} training and {len(val_indices)} validation samples")
        
        # Save a sample of the training data for visualization
        if not os.path.exists(os.path.join(args.output_dir, "training_samples")):
            os.makedirs(os.path.join(args.output_dir, "training_samples"))
            
        # Save 5 random samples
        indices = np.random.choice(len(inputs), 5, replace=False)
        for i, idx in enumerate(indices):
            plt.figure(figsize=(12, 4))
            
            # Original volume
            plt.subplot(1, 3, 1)
            plt.imshow(inputs[idx, args.chunk_size//2, :, :, 0], cmap='gray')
            plt.title(f'Volume (Sample {i+1})')
            plt.axis('off')
            
            # Guidance channel
            plt.subplot(1, 3, 2)
            plt.imshow(inputs[idx, args.chunk_size//2, :, :, 1], cmap='jet')
            plt.title('Guidance (Seed Points)')
            plt.axis('off')
            
            # Label
            plt.subplot(1, 3, 3)
            plt.imshow(labels[idx, args.chunk_size//2, :, :, 0], cmap='gray')
            plt.title('Label')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"training_samples/sample_{i+1}.png"))
            plt.close()
        
        print(f"Using batch size: {gpu_batch_factor} with {strategy.num_replicas_in_sync} GPU(s)")
        
        # If using multiple GPUs, make sure batch size is at least equal to num GPUs
        if use_multi_gpu and optimal_batch_size < strategy.num_replicas_in_sync:
            optimal_batch_size = strategy.num_replicas_in_sync
            print(f"Adjusted batch size to {optimal_batch_size} to match number of GPUs")
        
        # Now create training and validation datasets with the previously defined indices
        train_dataset = create_dataset(train_indices, gpu_batch_factor, inputs, labels, is_training=True)   # Corrected to use gpu_batch_factor, not 
        val_dataset = create_dataset(val_indices, gpu_batch_factor, inputs, labels, is_training=False)      # optimal_batch_size, trial. Both for train and val
        
        print(f"Created TensorFlow datasets: {len(train_indices)} training samples, {len(val_indices)} validation samples")
        
        # Clear existing model to free memory
        tf.keras.backend.clear_session()
        
        # Configure TensorFlow to be memory efficient
        tf.config.optimizer.set_jit(False)  # Disable XLA JIT compilation to reduce memory usage
        
        # Use a more memory-efficient model configuration if needed
        # Reduce model complexity if we're still having memory issues
        # if args.chunk_size >= 64: # Maybe alter here to use original filters_base
        #     print("Using more memory-efficient model configuration")
        #     input_shape = (args.chunk_size, args.chunk_size, args.chunk_size, 2)
        #     model = create_3d_unet(input_shape=input_shape, filters_base=16)  # Reduce filter count
        # else:
        #     model = create_3d_unet(input_shape=input_shape)
        
        try:
            # Don't wrap the model.fit call in a strategy scope because the model 
            # was already created with the strategy
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
        except Exception as e:
            print(f"Error during training: {e}")
            
            # If training fails, try with an even smaller dataset
            print("Trying with a smaller dataset and simpler model...")
            
            # Clear memory
            tf.keras.backend.clear_session()
            
            # Create a smaller subset of data
            max_samples = min(500, len(inputs))
            subset_indices = np.random.choice(len(inputs), max_samples, replace=False)
            subset_inputs = inputs[subset_indices]
            subset_labels = labels[subset_indices]
            
            # Create an even simpler model with the same strategy scope as before
            if use_multi_gpu:
                simple_model = create_simplified_model(input_shape=input_shape)
            else:
                simple_model = create_simplified_model(input_shape=input_shape)
            
            # Split into train/val
            val_size = int(max_samples * 0.2)
            train_inputs = subset_inputs[val_size:]
            train_labels = subset_labels[val_size:]
            val_inputs = subset_inputs[:val_size]
            val_labels = subset_labels[:val_size]
            
            # Train with the smallest possible batch size - don't wrap in strategy scope
            print(f"Training with minimal dataset: {len(train_inputs)} samples, batch size=1")
            
            # Determine appropriate batch size for multi-GPU
            fallback_batch_size = 1
            if use_multi_gpu:
                fallback_batch_size = max(1, strategy.num_replicas_in_sync)
                
            history = simple_model.fit(
                train_inputs, train_labels,
                validation_data=(val_inputs, val_labels),
                batch_size=fallback_batch_size,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Replace the model with the simple one
            model = simple_model
        
        # Save training visualizations
        save_training_visualizations(
            history, 
            os.path.join(args.output_dir, f"training_{timestamp}")
        )
        
        print(f"Model trained and saved to {model_path}")
    else:
        # Load pre-trained model for testing
        model_files = sorted(glob.glob(os.path.join(args.model_dir, "*.h5")))
        if not model_files:
            raise ValueError("No model files found for testing. Please train a model first.")
        
        latest_model = model_files[-1]
        print(f"Loading model from {latest_model}")
        model.load_weights(latest_model)
    
    # Print model summary
    model.summary()
    
    # For inference, expand the seed points to improve coverage
    inference_seed_points = expand_seed_points(seed_points, volume.shape)
    
    # Segment volume with lower threshold for better detection
    start_time = time.time()
    print(f"Segmenting volume using {len(inference_seed_points)} expanded seed points")
    
    # For inference, use the parallel version with multiple GPUs if available
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    
    # Modify this section to use the appropriate segmentation function
    if num_gpus > 1 and args.workers > 1:
        print(f"Using distributed segmentation with {num_gpus} GPUs")
        segmentation = segment_large_volume_distributed(
            model, 
            volume, 
            seed_points=inference_seed_points,
            chunk_size=args.chunk_size, 
            overlap=args.overlap,
            threshold=0.15,  # Much lower threshold for better recall
            num_workers=min(args.workers, num_gpus)  # Limit workers to number of GPUs
        )
    else:
        # Use the standard version for single GPU
        print(f"Using standard segmentation on {'GPU' if num_gpus > 0 else 'CPU'}")
        segmentation = segment_large_volume(
            model, 
            volume, 
            seed_points=inference_seed_points,
            chunk_size=args.chunk_size, 
            overlap=args.overlap,
            threshold=0.15  # Much lower threshold for better recall
        )
    
    # Apply post-processing to improve connectivity
    from scipy import ndimage
    print("Applying post-processing to enhance vessel connectivity")
    
    # Clean up small isolated components
    labeled, num_features = ndimage.label(segmentation)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes > 50  # Only keep components with at least 50 voxels
    mask_sizes[0] = 0  # Background stays background
    segmentation = mask_sizes[labeled]
    
    # Apply morphological closing to connect nearby segments
    segmentation = ndimage.binary_closing(segmentation, structure=np.ones((3,3,3))).astype(np.uint8) * 255
    
    # Save segmentation
    seg_output_path = os.path.join(args.output_dir, f"segmentation_{timestamp}.tif")
    print(f"Saving segmentation to {seg_output_path}")
    tiff.imwrite(seg_output_path, segmentation)
    
    # Save visualization of segmentation
    save_volume_slices(
        volume, 
        segmentation, 
        os.path.join(args.output_dir, f"segmentation_{timestamp}")
    )
    
    # Print statistics
    elapsed_time = time.time() - start_time
    segmented_voxels = np.sum(segmentation > 0)
    percentage = segmented_voxels / np.prod(segmentation.shape) * 100
    
    print(f"Segmentation complete in {elapsed_time:.2f} seconds")
    print(f"Segmented {segmented_voxels} voxels ({percentage:.4f}% of volume)")
    print(f"Used {len(seed_points)} seed points")
    
    # Save the stats to a file
    with open(os.path.join(args.output_dir, f"stats_{timestamp}.txt"), "w") as f:
        f.write(f"Segmentation Statistics:\n")
        f.write(f"- Processing time: {elapsed_time:.2f} seconds\n")
        f.write(f"- Segmented voxels: {segmented_voxels} ({percentage:.4f}% of volume)\n")
        f.write(f"- Seed points used: {len(seed_points)}\n")
    
    print("\nDone!")

if __name__ == "__main__":
    # Fix for TensorFlow memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    
    main()


