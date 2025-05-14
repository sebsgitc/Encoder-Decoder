"""Implementation of ResUNet with attention gates for binary vessel segmentation using TensorFlow."""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, Add, Multiply, Reshape
from configuration import IMAGE_SIZE, FILTERS  # Import IMAGE_SIZE and FILTERS from configuration
import numpy as np

class AttentionGate(tf.keras.layers.Layer):
    """
    Memory-optimized Attention Gate module for focusing on relevant features.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # Reduce intermediate feature dimension for better memory efficiency
        F_int = max(2, F_int // 4)  # Further reduce to 1/4 to save memory
        
        self.W_g = tf.keras.Sequential([
            Conv2D(F_int, kernel_size=1, strides=1, padding='same', use_bias=False),
            BatchNormalization()
        ])
        
        self.W_x = tf.keras.Sequential([
            Conv2D(F_int, kernel_size=1, strides=1, padding='same', use_bias=False),
            BatchNormalization()
        ])
        
        self.psi = tf.keras.Sequential([
            Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=False),
            BatchNormalization(),
            Activation('sigmoid')
        ])
        
        self.relu = Activation('relu')
        
    def call(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return Multiply()([x, psi])

def ResidualBlock(inputs, out_channels):
    """
    Residual block with two convolutional layers.
    """
    in_channels = inputs.shape[-1]
    
    x = Conv2D(out_channels, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(out_channels, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    
    if in_channels != out_channels:
        skip = Conv2D(out_channels, kernel_size=1, strides=1, padding='same', use_bias=False)(inputs)
        skip = BatchNormalization()(skip)
    else:
        skip = inputs
    
    x = Add()([x, skip])
    x = Activation('relu')(x)
    return x

def EncoderBlock(inputs, out_channels):
    """
    Encoder block with residual connection and max pooling.
    """
    skip = ResidualBlock(inputs, out_channels)
    x = MaxPooling2D(pool_size=2, strides=2)(skip)
    return x, skip

def DecoderBlock(inputs, skip, out_channels):
    """
    Decoder block with upsampling and attention gate.
    """
    x = Conv2DTranspose(out_channels, kernel_size=2, strides=2, padding='same')(inputs)
    
    # Attention gate
    attention_gate = AttentionGate(out_channels, out_channels, out_channels//2)
    skip_attention = attention_gate(x, skip)
    
    x = Concatenate()([x, skip_attention])
    x = ResidualBlock(x, out_channels)
    return x

def ResUNetWithAttention(input_shape=(None, None, 1), num_classes=2, filters=[64, 128, 256, 512, 1024]):
    """
    ResUNet with attention gates for lung segmentation.
    """
    inputs = Input(shape=input_shape)
    
    # Initial block
    x = ResidualBlock(inputs, filters[0])
    
    # Add spatial dropout after first block to reduce overfitting and memory usage
    x = tf.keras.layers.SpatialDropout2D(0.1)(x)
    
    # Encoder path - use max pooling more aggressively to reduce tensor sizes
    x1, skip1 = EncoderBlock(x, filters[1])
    x1 = tf.keras.layers.SpatialDropout2D(0.1)(x1)
    
    x2, skip2 = EncoderBlock(x1, filters[2])
    x2 = tf.keras.layers.SpatialDropout2D(0.2)(x2)
    
    x3, skip3 = EncoderBlock(x2, filters[3])
    x3 = tf.keras.layers.SpatialDropout2D(0.3)(x3)
    
    x4, skip4 = EncoderBlock(x3, filters[4])
    
    # Bridge
    x = ResidualBlock(x4, filters[4])
    
    # Decoder path
    x = DecoderBlock(x, skip4, filters[3])
    x = DecoderBlock(x, skip3, filters[2])
    x = DecoderBlock(x, skip2, filters[1])
    x = DecoderBlock(x, skip1, filters[0])
    
    # Output
    outputs = Conv2D(num_classes, kernel_size=1, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def call(self, y_true, y_pred):
        # For binary classification, the model outputs a single channel with sigmoid activation
        # Make sure we don't try to access an out-of-bounds index
        
        # If y_pred has a channel dimension with size > 1, take class 1 (vessels)
        if len(y_pred.shape) > 3 and y_pred.shape[-1] > 1:
            y_pred = y_pred[..., 1]
        elif len(y_pred.shape) > 3:
            # If there's only one channel, just use it directly (squeeze)
            y_pred = tf.squeeze(y_pred, axis=-1)
        
        if len(y_true.shape) > 2:
            # Convert sparse to binary
            y_true = tf.cast(tf.equal(y_true, 1), tf.float32)
        
        # Flatten label and prediction tensors
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])
        
        # Check for NaN values and replace with zeros - this prevents NaN propagation
        y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
        y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
        
        # Clip predictions to avoid numerical issues
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate intersection and dice score with increased smoothing factor
        intersection = tf.reduce_sum(y_true * y_pred)
        dice = (2. * intersection + self.smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth)
        
        # Check if result is NaN and return a safe value if so
        result = 1 - dice
        return tf.where(tf.math.is_nan(result), tf.ones_like(result) * 0.5, result)

class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights=None, use_focal=True, focal_gamma=2.0,
                 background_boost=1.0, border_weight=1.0, target_vessel_percentage=None,
                 use_proximity_weighting=True, threshold_encourage=5, threshold_discourage=20,
                 encourage_factor=0.8, discourage_factor=1.5, vessel_priority_factor=3.0):
        super(CombinedLoss, self).__init__()
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.dice_loss = DiceLoss()
        self.background_boost = background_boost  # Additional weight for background
        self.border_weight = border_weight  # Weight for boundary regions
        self.target_vessel_percentage = target_vessel_percentage  # Target vessel percentage
        
        # Proximity weighting parameters
        self.use_proximity_weighting = use_proximity_weighting
        self.threshold_encourage = threshold_encourage  # Distance threshold for encouraging pixels
        self.threshold_discourage = threshold_discourage  # Distance threshold for discouraging pixels
        self.encourage_factor = encourage_factor  # Factor for pixels close to annotations (< 1)
        self.discourage_factor = discourage_factor  # Factor for pixels far from annotations (> 1)
        
        # New parameter for prioritizing vessel accuracy over background
        self.vessel_priority_factor = vessel_priority_factor  # Weight multiplier for false negatives
    
    def calculate_distance_weights(self, y_true):
        """
        Calculate distance-based weight map to encourage predictions near existing annotations.
        
        Args:
            y_true: Ground truth mask [batch_size, height, width]
            
        Returns:
            Weight map based on distance from annotations
        """
        # Handle potential shape issues - ensure y_true has proper batch dimension
        if len(tf.shape(y_true)) < 3:
            y_true = tf.expand_dims(y_true, 0)  # Add batch dimension if missing
            
        # Get batch size and dimensions safely
        shape = tf.shape(y_true)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        
        # Initialize weight map with ones
        weight_map = tf.ones_like(y_true, dtype=tf.float32)
        
        # Handle empty batch case
        if batch_size == 0:
            return weight_map
            
        # Use a more efficient implementation for calculating all distance transforms at once
        # to avoid the loop which could cause shape issues
        try:
            # Convert to numpy for the distance transform operation
            y_true_np = y_true.numpy()
            
            # Calculate distances for all slices in the batch
            distances = np.zeros_like(y_true_np, dtype=np.float32)
            
            for b in range(batch_size):
                # Calculate distance transform for this slice
                mask = y_true_np[b]
                distances[b] = self._distance_transform(mask)
                
            # Convert back to tensor
            distances = tf.convert_to_tensor(distances, dtype=tf.float32)
            
            # Create masks for different distance regions
            close_mask = tf.cast(distances < self.threshold_encourage, tf.float32)
            far_mask = tf.cast(distances > self.threshold_discourage, tf.float32)
            middle_mask = 1.0 - close_mask - far_mask  # Neither close nor far
            
            # Combine weights
            weight_map = (close_mask * self.encourage_factor + 
                          middle_mask * 1.0 + 
                          far_mask * self.discourage_factor)
            
        except Exception:
            # Fallback to a simpler implementation if the numpy approach fails
            # Just return a uniform weight map
            print("Warning: Distance transform calculation failed, using uniform weights")
            weight_map = tf.ones_like(y_true, dtype=tf.float32)
            
        return weight_map
    
    def _distance_transform(self, binary_mask):
        """
        Calculate distance transform from annotated pixels.
        Uses SciPy's distance_transform_edt for exact Euclidean distances.
        
        Args:
            binary_mask: Binary mask where 1 indicates annotated vessel pixels
        
        Returns:
            Distance map with distances from annotated pixels
        """
        from scipy.ndimage import distance_transform_edt
        
        # For vessel class, calculate distance from positive pixels
        vessel_distances = distance_transform_edt(1 - binary_mask)
        return vessel_distances
    
    def call(self, y_true, y_pred):
        # Check for empty slices (all background) and handle specially
        is_empty_slice = tf.equal(tf.reduce_sum(y_true), 0)
        
        # Handle binary case properly
        if len(y_pred.shape) > 3 and y_pred.shape[-1] > 1:
            # Multi-class case
            ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()(
                y_true, y_pred
            )
        else:
            # Binary case
            # Ensure consistent data types and prevent NaN
            y_true_binary = tf.cast(tf.equal(y_true, 1), tf.float32)
            y_pred_binary = tf.squeeze(y_pred, axis=-1) if len(y_pred.shape) > 3 else y_pred
            y_pred_binary = tf.cast(y_pred_binary, tf.float32)  # Ensure float32 type
            
            # For empty slices (all background), we want to encourage predicting all background
            # If the prediction is already all background, return a very small loss
            if is_empty_slice:
                # If prediction is mostly background (mean < 0.1), return small loss
                if tf.reduce_mean(y_pred_binary) < 0.1:
                    return tf.constant(0.01, dtype=tf.float32)  # Small constant loss
                # Otherwise use normal loss calculation to encourage learning
            
            # Clip predictions to avoid log(0) or division by zero
            epsilon = 1e-7
            y_pred_binary = tf.clip_by_value(y_pred_binary, epsilon, 1.0 - epsilon)
            
            # Check for NaN values and replace with safe values
            y_pred_binary = tf.where(tf.math.is_nan(y_pred_binary), tf.ones_like(y_pred_binary) * 0.5, y_pred_binary)
            y_true_binary = tf.where(tf.math.is_nan(y_true_binary), tf.zeros_like(y_true_binary), y_true_binary)
            
            # Calculate proximity-based weights if enabled - with extra shape safety
            if self.use_proximity_weighting and not is_empty_slice:
                try:
                    # Only apply proximity weighting if we have a non-empty batch
                    if tf.shape(y_true)[0] > 0:
                        proximity_weights = self.calculate_distance_weights(y_true)
                        
                        # Ensure weights have the same shape as the predictions for safe multiplication
                        if tf.shape(proximity_weights)[0] != tf.shape(y_true_binary)[0]:
                            # If shapes don't match, fallback to ones
                            proximity_weights = tf.ones_like(y_true_binary, dtype=tf.float32)
                    else:
                        proximity_weights = tf.ones_like(y_true_binary, dtype=tf.float32)
                except Exception:
                    # In case of any error, use uniform weights
                    proximity_weights = tf.ones_like(y_true_binary, dtype=tf.float32)
            else:
                proximity_weights = tf.ones_like(y_true_binary, dtype=tf.float32)
            
            # New: Create vessel priority weights to give higher importance to vessel pixels
            # This makes false negatives (missing vessels) more costly than false positives
            vessel_weights = tf.ones_like(y_true_binary, dtype=tf.float32)
            vessel_weights = tf.where(tf.equal(y_true_binary, 1), 
                                     vessel_weights * self.vessel_priority_factor,  # Vessel pixels weighted higher
                                     vessel_weights)  # Background pixels unchanged
            
            # Add vessel percentage constraint based on target percentage
            if self.target_vessel_percentage is not None and not is_empty_slice:
                # Calculate current vessel percentage in prediction (0-100 scale)
                current_vessel_percentage = tf.reduce_mean(y_pred_binary) * 100
                
                # Convert target percentage to proportion (0-1 scale)
                target_proportion = self.target_vessel_percentage / 100.0
                
                # NEW: Penalize more if predictions are below target (to prevent vessel under-prediction)
                percentage_error = tf.cond(
                    tf.reduce_mean(y_pred_binary) < target_proportion,
                    lambda: (target_proportion - tf.reduce_mean(y_pred_binary)) * 1.5,  # Higher penalty for under-prediction
                    lambda: (tf.reduce_mean(y_pred_binary) - target_proportion) * 0.8   # Lower penalty for over-prediction
                )
                
                # Create a scaling factor for the loss based on how close we are to the target
                # As we get closer to the target percentage, this factor decreases
                percentage_scale = tf.clip_by_value(percentage_error * 5.0, 0.1, 2.0)
            else:
                percentage_scale = 1.0
            
            # Continue with existing focal loss or BCE calculation
            if self.use_focal:
                # Get probabilities for the positive class - ensure float32 and handle NaN
                p_t = tf.cast(y_true_binary * y_pred_binary + (1 - y_true_binary) * (1 - y_pred_binary), tf.float32)
                p_t = tf.clip_by_value(p_t, epsilon, 1.0 - epsilon)  # Avoid numerical issues
                
                # For slices with all background, modify focal weight to avoid instability
                if is_empty_slice:
                    focal_weight = tf.pow(y_pred_binary, self.focal_gamma)  # Focus on false positives
                else:
                    # Apply class weights with background boost - ensure float32
                    alpha = 0.25
                    if self.class_weights is not None and len(self.class_weights) >= 2:
                        # Safe access to class weights
                        # Apply background boost to class weights
                        bg_weight = tf.cast(self.class_weights[0] * self.background_boost, tf.float32)
                        vessel_weight = tf.cast(self.class_weights[1], tf.float32)
                        
                        # Create the alpha factor with the boosted background weight
                        alpha_factor = tf.cast(y_true_binary * vessel_weight + (1 - y_true_binary) * bg_weight, tf.float32)
                    else:
                        alpha_factor = tf.cast(y_true_binary * alpha + (1 - y_true_binary) * (1 - alpha) * self.background_boost, tf.float32)
                    
                    # Calculate focal weight - higher gamma means more focus on hard examples
                    focal_weight = tf.cast(tf.pow(1.0 - p_t, self.focal_gamma), tf.float32)
                
                    # Calculate focal loss with safe log
                    log_pt = tf.math.log(p_t)  # p_t is already clipped above
                    focal_loss = tf.cast(-alpha_factor * focal_weight * log_pt, tf.float32)
                    
                    # Apply proximity weights to focal loss
                    focal_loss = focal_loss * tf.cast(proximity_weights, tf.float32)
                    
                    # NEW: Apply vessel priority weights to focal loss
                    focal_loss = focal_loss * tf.cast(vessel_weights, tf.float32)
                    
                    # Check for NaN in focal loss
                    focal_loss = tf.where(tf.math.is_nan(focal_loss), tf.zeros_like(focal_loss), focal_loss)
                    
                    # Identify border regions - ensure float32 consistency
                    border_regions = tf.cast(
                        tf.logical_and(
                            tf.greater(y_pred_binary, 0.3),
                            tf.less(y_pred_binary, 0.7)
                        ),
                        tf.float32
                    )
                    
                    # Apply additional weight to border regions
                    border_weight_factor = tf.ones_like(focal_loss, dtype=tf.float32) + (border_regions * (self.border_weight - 1.0))
                    focal_loss = tf.cast(focal_loss * border_weight_factor, tf.float32)
                    
                    # Reduce mean safely
                    non_nan_elements = tf.cast(tf.logical_not(tf.math.is_nan(focal_loss)), tf.float32)
                    safe_sum = tf.reduce_sum(tf.where(tf.math.is_nan(focal_loss), tf.zeros_like(focal_loss), focal_loss))
                    safe_count = tf.reduce_sum(non_nan_elements) + epsilon
                    ce_loss = safe_sum / safe_count
                
                # If the slice is empty, use binary cross entropy focused on false positives
                if is_empty_slice:
                    # For empty slices, use simple BCE with focus on reducing false positives
                    bce = tf.keras.losses.BinaryCrossentropy(
                        reduction=tf.keras.losses.Reduction.NONE
                    )(y_true_binary, y_pred_binary)
                    
                    # Fix: Handle shape mismatch between focal_weight and bce tensors
                    # Instead of reshaping (which causes dimension issues), broadcast multiply properly
                    # First make sure both tensors have compatible shapes
                    bce_shape = tf.shape(bce)
                    
                    # Create a compatible focal weight with proper broadcasting
                    # Use simple loss weighting for empty slices instead of the complicated focal weight
                    # This avoids the need for reshaping entirely
                    weighted_bce = bce * 0.7  # Use constant weight for empty slices
                    
                    # Add a check for NaN values
                    weighted_bce = tf.where(tf.math.is_nan(weighted_bce), 
                                           tf.zeros_like(weighted_bce), 
                                           weighted_bce)
                    
                    # Safe reduction
                    ce_loss = tf.reduce_mean(weighted_bce)
                
                # Apply percentage constraint scale to final loss
                ce_loss = ce_loss * percentage_scale
                
            else:
                # Standard binary cross entropy with class weights
                if self.class_weights is not None and len(self.class_weights) >= 2:
                    # Apply class weights with background boost
                    pos_weight = self.class_weights[1] / (self.class_weights[0] * self.background_boost)
                    bce = tf.keras.losses.BinaryCrossentropy(
                        reduction=tf.keras.losses.Reduction.NONE  # Get per-element losses
                    )(y_true_binary, y_pred_binary)
                    
                    # Apply class weights manually and handle NaN
                    weighted_bce = tf.where(
                        tf.equal(y_true_binary, 1),
                        bce * pos_weight,
                        bce
                    )
                    
                    # NEW: Apply vessel priority weights
                    weighted_bce = weighted_bce * vessel_weights
                    
                    # Apply proximity weighting to BCE loss
                    weighted_bce = weighted_bce * tf.cast(proximity_weights, tf.float32)
                    
                    # Safe reduction
                    ce_loss = tf.reduce_mean(tf.where(tf.math.is_nan(weighted_bce), tf.zeros_like(weighted_bce), weighted_bce))
                else:
                    bce = tf.keras.losses.BinaryCrossentropy(
                        reduction=tf.keras.losses.Reduction.NONE
                    )(y_true_binary, y_pred_binary)
                    
                    # NEW: Apply vessel priority weights
                    bce = bce * vessel_weights
                    
                    # Apply proximity weighting to BCE loss
                    weighted_bce = bce * tf.cast(proximity_weights, tf.float32)
                    ce_loss = tf.reduce_mean(weighted_bce)
        
        # For empty slices, we can skip the dice loss which would be unstable
        if is_empty_slice:
            # Use only cross-entropy for empty slices
            return ce_loss
        
        # Calculate dice loss with vessel priority
        # We use vanilla dice loss but increase weight for vessels
        dice_loss = self.dice_loss(y_true, y_pred)
        
        # Handle NaN in total loss
        ce_loss = tf.where(tf.math.is_nan(ce_loss), tf.zeros_like(ce_loss) + 0.5, ce_loss)
        dice_loss = tf.where(tf.math.is_nan(dice_loss), tf.zeros_like(dice_loss) + 0.5, dice_loss)
        
        # Weighted combination of losses - cast to float32 to ensure consistency
        # Use a higher weight for Dice loss to further emphasize vessel segmentation
        total_loss = tf.cast(ce_loss, tf.float32) + tf.cast(dice_loss * 1.2, tf.float32)  # Higher weight for Dice
        
        # Final NaN check
        return tf.where(tf.math.is_nan(total_loss), tf.constant(1.0, dtype=tf.float32), total_loss)

# Add a new function to create a model with better background handling
def get_model_with_improved_background(in_channels=1, num_classes=2, filters=None):
    """Create a model with improved background handling capabilities."""
    if filters is None:
        filters = FILTERS
    
    # Create the attention model with explicit input shape
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, in_channels))
    
    # Initial block with increased receptive field for better context
    x = Conv2D(filters[0], kernel_size=5, padding='same')(inputs)  # Larger kernel for more context
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip0 = x
    
    # Encoder path with attention for better feature extraction
    x1, skip1 = EncoderBlock(x, filters[1])
    x1 = tf.keras.layers.SpatialDropout2D(0.1)(x1)
    
    x2, skip2 = EncoderBlock(x1, filters[2])
    x2 = tf.keras.layers.SpatialDropout2D(0.1)(x2)
    
    x3, skip3 = EncoderBlock(x2, filters[3])
    x3 = tf.keras.layers.SpatialDropout2D(0.2)(x3)
    
    x4, skip4 = EncoderBlock(x3, filters[4])
    
    # Bridge
    x = ResidualBlock(x4, filters[4])
    
    # Decoder path with background-focused attention
    x = DecoderBlock(x, skip4, filters[3])
    x = DecoderBlock(x, skip3, filters[2])
    x = DecoderBlock(x, skip2, filters[1])
    x = DecoderBlock(x, skip1, filters[0])
    
    # Add an extra background-focused block
    x = Concatenate()([x, skip0])  # Connect with initial features for better background detail
    x = Conv2D(filters[0], kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Output layer with sigmoid activation for binary segmentation
    outputs = Conv2D(1, kernel_size=1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Update the main get_model function to use the improved model
def get_model(in_channels=1, num_classes=2, filters=None):
    """Create a model with customizable filters to control memory usage."""
    if filters is None:
        # Use the default filters defined in configuration
        filters = FILTERS
    
    # Use the improved background model instead of the standard attention model
    use_improved_background_model = True
    
    if use_improved_background_model:
        return get_model_with_improved_background(in_channels, num_classes, filters)
    else:
        model = ResUNetWithAttention(input_shape=(IMAGE_SIZE, IMAGE_SIZE, in_channels), 
                                     num_classes=num_classes, 
                                     filters=filters)
        
        # Use sigmoid activation for binary classification if num_classes == 2
        if num_classes == 2:
            inputs = model.inputs
            x = model.layers[-2].output  # Get the output before the final activation
            outputs = Conv2D(1, kernel_size=1, activation='sigmoid')(x)
            model = Model(inputs=inputs, outputs=outputs)
    
    # For improved performance with large inputs, use lower precision
    if IMAGE_SIZE >= 512:
        model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return model