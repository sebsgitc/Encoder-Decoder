"""Training script for binary vessel segmentation model using TensorFlow."""

import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from configuration import *
from model import get_model, CombinedLoss
from data_loader import prepare_data
from utils import save_checkpoint, load_checkpoint, create_logger, AverageMeter
from evaluate import evaluate_model

# Set GPU device at the module level BEFORE any other TensorFlow operations
# This needs to happen before any other TensorFlow code runs
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    print(f"Found {len(gpus)} GPUs, restricting to GPU:0") #Normaly we want to use GPU:1, trying for GPU:0 right now
    # Set only GPU:1 as visible (index 1 is the second GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # This clears the GPU memory cache and resets device settings
    tf.keras.backend.clear_session()
    # Get updated list of available GPUs after setting environment variable
    gpus = tf.config.list_physical_devices('GPU')
else:
    print(f"Only {len(gpus)} GPU found, using it with memory growth enabled")

if gpus:
    # Enable memory growth for all available GPUs
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Error configuring GPU memory growth: {e}")

# Set up mixed precision policy if enabled - update to handle loss computation correctly
if USE_MIXED_PRECISION:
    # Set mixed precision policy but keep losses in float32
    policy = tf.keras.mixed_precision.Policy(MIXED_PRECISION_DTYPE)
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision enabled with dtype: {policy.compute_dtype}, loss dtype: float32")

def train_one_epoch(model, dataset, optimizer, loss_fn):
    """Train the model for one epoch with background-focused improvements."""
    epoch_loss = AverageMeter()
    background_loss = AverageMeter()  # Track background loss separately
    vessel_loss = AverageMeter()      # Track vessel loss separately
    nan_batch_count = 0  # Track batches containing NaN values
    
    progress_bar = tqdm(dataset, desc="Training")
    
    # Enable memory growth during training to avoid running out of GPU memory
    tf.config.experimental.enable_tensor_float_32_execution(True)
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # Handle both regular data and boundary-weighted data
        if len(batch_data) == 3:  # With boundary weights
            images, masks, weights = batch_data
        else:  # Regular data
            images, masks = batch_data
            weights = None
        
        # Enhanced NaN detection and handling at the batch level
        if tf.reduce_any(tf.math.is_nan(images)) or tf.reduce_any(tf.math.is_inf(images)):
            # Count NaNs for monitoring
            nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(images), tf.int32)).numpy()
            inf_count = tf.reduce_sum(tf.cast(tf.math.is_inf(images), tf.int32)).numpy()
            nan_batch_count += 1
            
            # Replace NaN values with zeros and Inf with ones
            print(f"Warning: NaN detected in input images in batch {batch_idx}. Fixing...")
            images = tf.where(tf.math.is_nan(images), tf.zeros_like(images), images)
            images = tf.where(tf.math.is_inf(images), tf.ones_like(images), images)
        
        # Forward pass and compute loss using gradient tape
        with tf.GradientTape() as tape:
            # Apply try-except around the model call for robustness
            try:
                outputs = model(images, training=True)
                
                # Check for NaN in model outputs
                if tf.reduce_any(tf.math.is_nan(outputs)) or tf.reduce_any(tf.math.is_inf(outputs)):
                    print(f"Warning: NaN detected in model outputs in batch {batch_idx}. Using fallback...")
                    # Create a fallback output that doesn't have NaN values
                    outputs = tf.where(tf.math.is_nan(outputs), tf.ones_like(outputs) * 0.5, outputs)
                    outputs = tf.where(tf.math.is_inf(outputs), tf.ones_like(outputs) * 0.5, outputs)
                
                # Apply main loss - ensure outputs are cast to float32 for loss computation
                outputs_float32 = tf.cast(outputs, tf.float32)
                # Clip outputs more aggressively to prevent extreme values
                outputs_float32 = tf.clip_by_value(outputs_float32, 1e-7, 1.0 - 1e-7)
                
                try:
                    # Calculate loss with additional safeguards
                    loss = loss_fn(masks, outputs_float32)
                    
                    # Check if loss is NaN and use a fallback value
                    if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                        print(f"Warning: NaN/Inf loss detected in batch {batch_idx}. Using fallback loss value.")
                        loss = tf.constant(1.0, dtype=tf.float32)  # Fallback to a reasonable loss value
                except Exception as loss_err:
                    print(f"Error calculating loss in batch {batch_idx}: {str(loss_err)}")
                    loss = tf.constant(1.0, dtype=tf.float32)  # Fallback loss
            except Exception as model_err:
                print(f"Error in model forward pass for batch {batch_idx}: {str(model_err)}")
                # Skip this batch if model call fails
                continue
        
        # Backward pass and optimize
        try:
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Check for NaN gradients and replace them with zeros
            has_nan_gradients = False
            patched_gradients = []
            for grad in gradients:
                if grad is not None and tf.reduce_any(tf.math.is_nan(grad)):
                    has_nan_gradients = True
                    patched_grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
                    patched_gradients.append(patched_grad)
                else:
                    patched_gradients.append(grad)
                    
            if has_nan_gradients:
                print(f"Warning: NaN gradients detected in batch {batch_idx}. Patching gradient values.")
                gradients = patched_gradients
            
            # Clip gradients to prevent explosion with mixed precision
            gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
            
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
        except Exception as e:
            print(f"Error during gradient update: {str(e)}")
            continue  # Skip this batch
        
        # Update metrics, handle NaN in loss value
        try:
            if not tf.math.is_nan(loss):
                epoch_loss.update(loss.numpy())
            else:
                print(f"Skipping NaN loss in metrics for batch {batch_idx}")
                
            # Track background/foreground metrics separately for monitoring
            if len(outputs.shape) > 3:
                predictions = tf.cast(outputs > 0.5, tf.float32).numpy()
                masks_np = masks.numpy()
                
                # Calculate separate metrics for background and vessels
                for i in range(len(masks_np)):
                    mask_flat = masks_np[i].flatten()
                    pred_flat = predictions[i].squeeze().flatten()
                    
                    # Background accuracy
                    bg_indices = (mask_flat == 0)
                    if np.sum(bg_indices) > 0:
                        bg_acc = np.mean((pred_flat[bg_indices] == 0).astype(float))
                        if not np.isnan(bg_acc):  # Skip NaN values
                            background_loss.update(1.0 - bg_acc)
                    
                    # Vessel accuracy
                    vessel_indices = (mask_flat == 1)
                    if np.sum(vessel_indices) > 0:
                        vessel_acc = np.mean((pred_flat[vessel_indices] == 1).astype(float))
                        if not np.isnan(vessel_acc):  # Skip NaN values
                            vessel_loss.update(1.0 - vessel_acc)
        except Exception as metric_err:
            print(f"Error updating metrics: {str(metric_err)}")
        
        # Update progress bar with non-NaN values
        progress_bar.set_postfix(
            loss=f"{epoch_loss.avg:.4f}" if not np.isnan(epoch_loss.avg) else "NaN",
            bg_err=f"{background_loss.avg:.4f}" if not np.isnan(background_loss.avg) else "NaN",
            vs_err=f"{vessel_loss.avg:.4f}" if not np.isnan(vessel_loss.avg) else "NaN"
        )
        
    # Print summary of NaN occurrences
    if nan_batch_count > 0:
        print(f"\nWARNING: Encountered {nan_batch_count} batches with NaN values during training")
    
    return epoch_loss.avg, background_loss.avg, vessel_loss.avg

def validate(model, dataset, loss_fn):
    """Validate the model with additional vessel preservation metrics."""
    val_loss = AverageMeter()
    
    # Add confusion matrix to track class distribution - counting pixels instead of batches
    conf_matrix = np.zeros((2, 2), dtype=np.int64)  # Use int64 to handle large pixel counts
    
    # NEW: Add tracking for vessel preservation
    true_vessel_pixels = 0
    detected_vessel_pixels = 0
    correctly_detected_vessels = 0
    
    progress_bar = tqdm(dataset, desc="Validating")
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        # Forward pass
        outputs = model(images, training=False)
        loss = loss_fn(masks, outputs)
        
        # Update metrics
        val_loss.update(loss.numpy())
        
        # Calculate confusion matrix for monitoring class balance
        predictions = (tf.cast(outputs > 0.5, tf.int32)).numpy()
        masks_np = masks.numpy()
        
        # Handle different shape cases
        if len(predictions.shape) > 3:  # Has channel dimension
            predictions = predictions.squeeze(axis=-1)
        
        # Count all pixels for confusion matrix instead of batch-level classification
        for i in range(len(masks_np)):
            # Flatten masks and predictions to count all pixels
            true_flat = masks_np[i].flatten()
            pred_flat = predictions[i].flatten()
            
            # NEW: Track vessel preservation metrics
            true_vessel_mask = (true_flat == 1)
            pred_vessel_mask = (pred_flat == 1)
            
            # Update vessel counts
            true_vessel_pixels += np.sum(true_vessel_mask)
            detected_vessel_pixels += np.sum(pred_vessel_mask)
            correctly_detected_vessels += np.sum(true_vessel_mask & pred_vessel_mask)
            
            # Update confusion matrix with pixel-level counts
            # [0,0]: true negative (correctly predicted background)
            # [0,1]: false positive (predicted vessel but is background)
            # [1,0]: false negative (predicted background but is vessel)
            # [1,1]: true positive (correctly predicted vessel)
            for t in [0, 1]:
                for p in [0, 1]:
                    conf_matrix[t, p] += np.sum((true_flat == t) & (pred_flat == p))
    
    # NEW: Calculate vessel preservation metrics
    vessel_preservation_data = {
        'true_vessel_pixels': true_vessel_pixels,
        'detected_vessel_pixels': detected_vessel_pixels,
        'correctly_detected_vessels': correctly_detected_vessels,
    }
    
    if true_vessel_pixels > 0:
        vessel_preservation_data['vessel_recall'] = correctly_detected_vessels / true_vessel_pixels
        vessel_preservation_data['vessel_precision'] = correctly_detected_vessels / max(detected_vessel_pixels, 1)
        vessel_preservation_data['vessel_detection_ratio'] = detected_vessel_pixels / true_vessel_pixels
    else:
        vessel_preservation_data['vessel_recall'] = 0
        vessel_preservation_data['vessel_precision'] = 0
        vessel_preservation_data['vessel_detection_ratio'] = 0
            
    return val_loss.avg, conf_matrix, vessel_preservation_data

def train_model():
    """Main training function for binary vessel segmentation."""
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Setup logger
    logger = create_logger()
    
    # Get data loaders
    logger.info("Loading datasets...")
    
    # Use the batch size from configuration
    requested_batch_size = BATCH_SIZE
    logger.info(f"Using batch size: {requested_batch_size}")
    
    # Log multi-dimensional slicing settings
    logger.info("Using only Z-axis slices for stable processing")
    
    # Initialize performance monitor if configured
    using_perf_monitor = False
    if ENABLE_PERFORMANCE_MONITORING:
        try:
            from performance_monitor import TrainingPerformanceMonitor
            perf_monitor = TrainingPerformanceMonitor(log_interval=30)
            perf_monitor.start()
            using_perf_monitor = True
            logger.info("Performance monitoring started")
        except (ImportError, ModuleNotFoundError):
            logger.info("Performance monitoring not available - continuing without it")
    
    try:
        # Use all available slices without explicitly passing slice_range
        # Now returns target_vessel_percentage as well
        train_dataset, val_dataset, target_vessel_percentage = prepare_data(batch_size=requested_batch_size)
        logger.info("Datasets loaded successfully")
        logger.info(f"Target vessel percentage: {target_vessel_percentage:.4f}%")
        
        # Calculate and log dataset size and expected steps per epoch
        train_size = 0
        for _ in train_dataset:
            train_size += 1
        logger.info(f"Training dataset has {train_size} batches (steps per epoch)")
        logger.info(f"With batch size {BATCH_SIZE}, this corresponds to approximately {train_size * BATCH_SIZE} slices")
        
        # Move NaN checking here, after datasets are loaded
        try:
            from nan_detector import check_dataset_for_nans
            
            # Only check a few batches to see if there's an immediate problem
            print("Running pre-training dataset NaN check...")
            nan_stats = check_dataset_for_nans(train_dataset, num_batches=5)
            
            if nan_stats['nan_count'] > 0:
                logger.warning(f"Detected {nan_stats['nan_count']} NaN values in dataset sample!")
                logger.warning("Consider running fix_nans.py before continuing training.")
                
                # Ask if user wants to continue
                print("\nNaN values detected in the dataset. Continue training anyway? (y/n)")
                response = input("> ")
                
                if response.lower() != 'y':
                    print("Training aborted. Run fix_nans.py to clean dataset.")
                    return
                
                print("Continuing with training despite NaN values...")
        except ImportError:
            logger.warning("NaN detector module not available. Continuing without NaN checks.")
    
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise
    
    # Initialize model with larger capacity to utilize available GPU memory
    logger.info("Initializing model...")
    # Increase filter sizes to use more GPU memory
    filters = FILTERS  # Use the updated filter sizes from configuration
    logger.info(f"Using model with filter sizes: {filters}")
    model = get_model(in_channels=1, num_classes=NUM_CLASSES, filters=filters)
    
    # Compile the model with focal loss and proximity weighting
    if USE_FOCAL_LOSS:
        logger.info(f"Using Focal Loss with gamma={FOCAL_LOSS_GAMMA} to handle class imbalance")
        
        # Define proximity weighting parameters (these could be added to configuration.py)
        use_proximity_weighting = True  # Enable proximity weighting
        threshold_encourage = 1         # Distance threshold for encouraging predictions near vessels
        threshold_discourage = 300      # Distance threshold for discouraging predictions far from vessels
        encourage_factor = 0.8          # Weight multiplier for close pixels (< 1 to reduce loss)
        discourage_factor = 15          # Weight multiplier for far pixels (> 1 to increase loss)
        
        # NEW: Add vessel priority factor - higher value means more importance given to vessel pixels
        vessel_priority_factor = 4.0    # Weight multiplier for false negatives (missed vessels)
        
        # NEW: Update class weights to favor vessels even more
        vessel_class_weight = 0.75
        bg_class_weight = 0.25  
        class_weights_balanced = [bg_class_weight, vessel_class_weight]
        
        loss_fn = CombinedLoss(
            class_weights=class_weights_balanced, 
            use_focal=True, 
            focal_gamma=FOCAL_LOSS_GAMMA,
            background_boost=BACKGROUND_FOCUS['background_boost_factor'],
            border_weight=BACKGROUND_FOCUS['border_weight_factor'],
            target_vessel_percentage=target_vessel_percentage,  # Pass the target percentage
            use_proximity_weighting=use_proximity_weighting,
            threshold_encourage=threshold_encourage,
            threshold_discourage=threshold_discourage,
            encourage_factor=encourage_factor,
            discourage_factor=discourage_factor,
            vessel_priority_factor=vessel_priority_factor  # Add vessel priority factor
        )
        logger.info(f"Using vessel percentage constraint with target: {target_vessel_percentage:.4f}%")
        logger.info(f"Using vessel priority factor of {vessel_priority_factor} to prioritize vessel detection")
        if use_proximity_weighting:
            logger.info(f"Using proximity-based loss weighting to encourage nearby vessel detection")
            logger.info(f"  Encourage threshold: {threshold_encourage} pixels, factor: {encourage_factor}")
            logger.info(f"  Discourage threshold: {threshold_discourage} pixels, factor: {discourage_factor}")
    else:
        loss_fn = CombinedLoss(
            class_weights=CLASS_WEIGHTS,
            target_vessel_percentage=target_vessel_percentage,  # Pass the target percentage
            vessel_priority_factor=4.0  # Also use vessel priority with non-focal loss
        )
    
    # Define metrics for binary classification
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC()
    ]
    
    # Use a learning rate schedule for better convergence
    initial_lr = LEARNING_RATE
    logger.info(f"Initial learning rate: {initial_lr}")
    
    # Create optimizer with background-focused improvements
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    
    # Setup learning rate scheduler
    lr_factor = 0.5
    lr_patience = 5
    min_delta = 0.0001
    min_lr = 1e-6
    
    # Log the model summary to see parameter count
    model.summary(print_fn=logger.info)
    
    # Log GPU memory information before training
    gpu_info = {}
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        for device in gpu_devices:
            gpu_info_dict = tf.config.experimental.get_memory_info(device)
            gpu_info[device.name] = gpu_info_dict
        logger.info(f"GPU memory info before training: {gpu_info}")
    except:
        logger.info("Could not get detailed GPU memory information")
    
    # Training loop
    start_epoch = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    lr_history = []
    no_improvement_count = 0
    current_lr = initial_lr
    
    # NEW: Track vessel preservation metrics
    best_vessel_recall = 0.0
    best_vessel_model_saved = False
    
    logger.info("Starting training for binary vessel segmentation...")
    
    try:
        for epoch in range(start_epoch, EPOCHS):
            logger.info(f"Epoch {epoch+1}/{EPOCHS}")
            
            # Train for one epoch with background metrics
            train_loss, bg_loss, vs_loss = train_one_epoch(model, train_dataset, optimizer, loss_fn)
            
            # Validate with confusion matrix and vessel preservation metrics
            val_loss, conf_matrix, vessel_metrics = validate(model, val_dataset, loss_fn)
            
            # Log expanded metrics including background accuracy
            if np.sum(conf_matrix) > 0:
                # Calculate true positive rate (vessel recall)
                vessel_recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) if (conf_matrix[1, 1] + conf_matrix[1, 0]) > 0 else 0
                
                # Calculate true negative rate (background recall)
                bg_recall = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
                
                # Calculate precision for vessels and background
                vessel_precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) if (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0 else 0
                bg_precision = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0]) if (conf_matrix[0, 0] + conf_matrix[1, 0]) > 0 else 0
                
                # Calculate F1 scores for each class
                vessel_f1 = 2 * vessel_precision * vessel_recall / (vessel_precision + vessel_recall) if (vessel_precision + vessel_recall) > 0 else 0
                bg_f1 = 2 * bg_precision * bg_recall / (bg_precision + bg_recall) if (bg_precision + bg_recall) > 0 else 0
                
                # Log detailed metrics
                logger.info(f"Background metrics - Recall: {bg_recall:.4f}, Precision: {bg_precision:.4f}, F1: {bg_f1:.4f}")
                logger.info(f"Vessel metrics    - Recall: {vessel_recall:.4f}, Precision: {vessel_precision:.4f}, F1: {vessel_f1:.4f}")
                
                # Log vessel preservation metrics
                vessel_detection_ratio = vessel_metrics['vessel_detection_ratio']
                logger.info(f"Vessel preservation - Detection ratio: {vessel_detection_ratio:.4f}")
                logger.info(f"                    - True pixels: {vessel_metrics['true_vessel_pixels']}, Detected: {vessel_metrics['detected_vessel_pixels']}")
                
                # Special warning if vessel detection ratio drops too low
                if vessel_detection_ratio < 0.75:
                    logger.warning(f"WARNING: Vessel detection ratio is low at {vessel_detection_ratio:.4f}. Need to prioritize vessel detection more.")
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr}")
            logger.info(f"Confusion Matrix:\n{conf_matrix}")
            
            # Check if validation loss improved
            is_best_loss = val_loss < best_val_loss - min_delta
            if is_best_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                early_stopping_counter += 1
            
            # NEW: Check if vessel recall improved
            is_best_vessel_recall = vessel_recall > best_vessel_recall
            if is_best_vessel_recall:
                best_vessel_recall = vessel_recall
                # Save a special checkpoint for the best vessel recall model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model,
                    'optimizer': optimizer,
                    'vessel_recall': vessel_recall,
                }, False, filename='checkpoints/best_vessel_recall')
                best_vessel_model_saved = True
                logger.info(f"Saved new best vessel recall model with recall: {vessel_recall:.4f}")
            
            # Reduce learning rate if needed
            if no_improvement_count >= lr_patience:
                new_lr = max(current_lr * lr_factor, min_lr)
                if new_lr != current_lr:
                    logger.info(f"Reducing learning rate from {current_lr} to {new_lr}")
                    current_lr = new_lr
                    optimizer.learning_rate.assign(current_lr)
                    no_improvement_count = 0  # Reset counter after reducing LR
            
            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model,
                'optimizer': optimizer,
                'best_val_loss': best_val_loss,
            }, is_best_loss)
            
            # NEW: If we're risking under-segmenting vessels, adjust the loss function
            if epoch > 5 and vessel_detection_ratio < 0.8:
                # Increase vessel priority to recover lost vessel detections
                current_vessel_priority = loss_fn.vessel_priority_factor
                new_vessel_priority = min(current_vessel_priority * 1.2, 8.0)  # Increase by 20%, max 8.0
                loss_fn.vessel_priority_factor = new_vessel_priority
                logger.info(f"Adjusting vessel priority from {current_vessel_priority:.2f} to {new_vessel_priority:.2f} to increase vessel detection")
            
            # Early stopping - check vessel recall stability
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                # NEW: If vessel recall is too low, continue training with stronger vessel priority
                if vessel_recall < 0.65 and epoch < EPOCHS - 5:  # Still have epochs left
                    logger.info(f"Vessel recall too low ({vessel_recall:.4f}), resetting early stopping and boosting vessel priority")
                    early_stopping_counter = 0
                    loss_fn.vessel_priority_factor = 6.0  # Strong boost to vessel priority
                    # Also reduce learning rate
                    current_lr = current_lr * 0.5
                    optimizer.learning_rate.assign(current_lr)
                    logger.info(f"Reduced learning rate to {current_lr} and increased vessel priority to 6.0")
                else:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Final evaluation on best models
        logger.info("Training completed. Loading best models for evaluation...")
        best_model = get_model(in_channels=1, num_classes=NUM_CLASSES)
        load_checkpoint(best_model, 'checkpoints/model_best.weights.h5')
        
        # Evaluate with both best loss model and best vessel recall model
        metrics = evaluate_model(best_model, val_dataset, visualize=SAVE_EVALUATION_IMAGES)
        logger.info(f"Final evaluation metrics (best loss model):")
        logger.info(f"Dice: {metrics['dice']:.4f}")
        logger.info(f"IoU: {metrics['iou']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1: {metrics['f1']:.4f}")
        
        # Also evaluate best vessel recall model if we saved one
        if best_vessel_model_saved:
            best_recall_model = get_model(in_channels=1, num_classes=NUM_CLASSES)
            load_checkpoint(best_recall_model, 'checkpoints/best_vessel_recall.weights.h5')
            
            metrics_recall_model = evaluate_model(best_recall_model, val_dataset, 
                                               visualize=SAVE_EVALUATION_IMAGES, 
                                               output_suffix="_best_vessel")
            logger.info(f"Final evaluation metrics (best vessel recall model):")
            logger.info(f"Dice: {metrics_recall_model['dice']:.4f}")
            logger.info(f"IoU: {metrics_recall_model['iou']:.4f}")
            logger.info(f"Accuracy: {metrics_recall_model['accuracy']:.4f}")
            logger.info(f"Precision: {metrics_recall_model['precision']:.4f}")
            logger.info(f"Recall: {metrics_recall_model['recall']:.4f}")
            logger.info(f"F1: {metrics_recall_model['f1']:.4f}")
            
            # Compare models and choose the final one to save
            if metrics_recall_model['recall'] > metrics['recall'] * 1.1:  # If recall is 10% better
                logger.info("Best vessel recall model has significantly better recall. Using it as the final model.")
                best_model = best_recall_model
                best_model.save_weights('checkpoints/final_model.weights.h5')
        
        # Log the path to evaluation images
        if SAVE_EVALUATION_IMAGES:
            logger.info(f"Evaluation images saved to: {EVALUATION_OUTPUT_DIR}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        # Stop performance monitoring if it was started
        if using_perf_monitor:
            perf_monitor.stop()
            logger.info("Performance monitoring stopped")

if __name__ == "__main__":
    print("TensorFlow devices after configuration:", tf.config.list_physical_devices())
    train_model()