"""Evaluation functions for binary vessel segmentation model using TensorFlow."""

import tensorflow as tf
import numpy as np
import matplotlib
# Force matplotlib to not use X server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, jaccard_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os
from datetime import datetime
from configuration import EVALUATION_OUTPUT_DIR

from utils import dice_coefficient, generate_confusion_matrix, visualize_segmentation

def evaluate_model(model, dataset, visualize=False, output_suffix=""):
    """
    Evaluate the model on the validation set for binary classification.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to evaluate
    dataset : tf.data.Dataset
        Validation dataset
    visualize : bool
        Whether to visualize predictions
    output_suffix : str
        Suffix to append to output filenames for differentiating between models
        
    Returns:
    --------
    dict
        Dictionary containing metrics
    """
    model.trainable = False
    
    all_dices = []
    all_ious = []
    all_accuracies = []
    confusion = np.zeros((2, 2))
    
    # Collect predictions and true labels for ROC and PR curves
    all_preds_raw = []
    all_labels = []
    
    # Generate timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_output_dir = os.path.join(EVALUATION_OUTPUT_DIR, f"eval_{timestamp}{output_suffix}")
    if visualize:
        os.makedirs(eval_output_dir, exist_ok=True)
    
    # Set up progress bar
    progress_bar = tqdm(dataset, desc="Evaluating")
    sample_count = 0
    
    # NEW: Add vessel-specific tracking
    vessel_pixel_stats = {
        'total_true_vessels': 0,
        'total_predicted_vessels': 0, 
        'correctly_detected_vessels': 0
    }
    
    try:
        for batch_idx, (images, masks) in enumerate(progress_bar):
            # Forward pass
            outputs = model(images, training=False)
            
            # Get raw predictions (for ROC) and binary predictions
            try:
                preds_raw = outputs.numpy().squeeze()
                predictions = tf.cast(outputs > 0.5, tf.int32).numpy().squeeze()
                
                # Convert to numpy for evaluation
                masks_np = masks.numpy()
                
                # Process batch items individually to handle shape issues
                batch_size = images.shape[0] if len(images.shape) > 3 else 1
                
                # Create arrays if they don't exist
                if not isinstance(all_preds_raw, list):
                    all_preds_raw = []
                if not isinstance(all_labels, list):
                    all_labels = []
                
                # Handle single vs. batch predictions
                for i in range(batch_size):
                    # Extract single image and mask
                    if batch_size > 1:
                        img = images[i]
                        mask = masks_np[i]
                        
                        # Get prediction for this image
                        if len(preds_raw.shape) > 2:
                            pred = preds_raw[i]
                            binary_pred = predictions[i]
                        else:
                            # Single prediction for multiple images (shouldn't happen)
                            pred = preds_raw
                            binary_pred = predictions
                    else:
                        img = images
                        mask = masks_np
                        pred = preds_raw
                        binary_pred = predictions
                    
                    # Convert to flat arrays for metrics
                    mask_flat = mask.reshape(-1).astype(np.int32)
                    if len(binary_pred.shape) > 2:
                        pred_flat = binary_pred.reshape(-1).astype(np.int32)
                    else:
                        pred_flat = binary_pred.reshape(-1).astype(np.int32)
                    
                    # Skip if shapes don't match
                    if mask_flat.shape[0] != pred_flat.shape[0]:
                        continue
                    
                    # Calculate metrics
                    try:
                        dice = dice_coefficient(mask_flat, pred_flat)
                        all_dices.append(dice)
                        
                        # Only compute metrics if there's meaningful data
                        if np.sum(mask_flat) > 0 or np.sum(pred_flat) > 0:
                            iou = jaccard_score(mask_flat, pred_flat, average='binary', zero_division=0)
                            all_ious.append(iou)
                            
                            acc = accuracy_score(mask_flat, pred_flat)
                            all_accuracies.append(acc)
                            
                            # Update confusion matrix
                            conf = generate_confusion_matrix(mask_flat, pred_flat, num_classes=2)
                            confusion += conf
                            
                            # NEW: Update vessel pixel stats
                            vessel_pixel_stats['total_true_vessels'] += np.sum(mask_flat == 1)
                            vessel_pixel_stats['total_predicted_vessels'] += np.sum(pred_flat == 1)
                            vessel_pixel_stats['correctly_detected_vessels'] += np.sum((mask_flat == 1) & (pred_flat == 1))
                        
                            # Collect raw predictions and labels for curves
                            if pred_flat.size > 0 and mask_flat.size > 0:
                                all_preds_raw.extend(pred.reshape(-1))
                                all_labels.extend(mask_flat)
                    except Exception as metric_error:
                        # Skip this sample if metrics calculation fails
                        continue
                    
                    # Visualize and save samples
                    if visualize and sample_count % 50 == 0:  # Save every 50th sample, fix this. it only saves 1 sample
                        try:
                            # Get image data
                            img_np = img.numpy() if hasattr(img, 'numpy') else img
                            
                            # Remove channel dimension if present
                            if len(img_np.shape) > 2 and img_np.shape[-1] == 1:
                                img_np = img_np[:, :, 0]
                            
                            # Save visualization
                            image_path = os.path.join(eval_output_dir, f"sample_{sample_count}_visualization.png")
                            save_prediction_visualization(img_np, mask, binary_pred, image_path)
                        except Exception as vis_error:
                            # Continue if visualization fails
                            pass
                    
                    # Increment sample counter for all samples, not just visualized ones
                    sample_count += 1
            
            except Exception as batch_error:
                # Continue to next batch if processing fails
                continue
                
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    metrics['dice'] = np.mean(all_dices) if all_dices else 0.0
    metrics['iou'] = np.mean(all_ious) if all_ious else 0.0
    metrics['accuracy'] = np.mean(all_accuracies) if all_accuracies else 0.0
    
    # Calculate precision, recall, and F1 from confusion matrix
    if confusion[1, 0] + confusion[1, 1] > 0:  # Check if there are any vessel pixels
        precision_val = confusion[1, 1] / (confusion[0, 1] + confusion[1, 1] + 1e-8)
        recall_val = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1] + 1e-8)
        f1_val = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-8)
    else:
        precision_val = recall_val = f1_val = 0.0
    
    metrics['precision'] = precision_val
    metrics['recall'] = recall_val
    metrics['f1'] = f1_val
    metrics['confusion_matrix'] = confusion
    
    # NEW: Add vessel pixel stats
    metrics['vessel_stats'] = vessel_pixel_stats
    metrics['vessel_detection_ratio'] = (vessel_pixel_stats['total_predicted_vessels'] / 
                                      max(vessel_pixel_stats['total_true_vessels'], 1))
    
    # Try to calculate ROC and PR curves if we have enough data
    try:
        if len(all_preds_raw) > 0 and len(all_labels) > 0:
            # Convert lists to arrays
            all_preds_raw = np.array(all_preds_raw)
            all_labels = np.array(all_labels)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(all_labels, all_preds_raw)
            metrics['roc_auc'] = auc(fpr, tpr)
            
            # PR curve
            precision, recall, _ = precision_recall_curve(all_labels, all_preds_raw)
            metrics['pr_auc'] = average_precision_score(all_labels, all_preds_raw)
            
            # Save curves if visualization is enabled
            if visualize:
                save_roc_curve(fpr, tpr, metrics['roc_auc'], os.path.join(eval_output_dir, f"roc_curve{output_suffix}.png"))
                save_pr_curve(precision, recall, metrics['pr_auc'], os.path.join(eval_output_dir, f"pr_curve{output_suffix}.png"))
    except Exception as curve_error:
        # Skip curve metrics if calculation fails
        pass
    
    # Save confusion matrix visualization
    if visualize and 'confusion_matrix' in metrics:
        save_confusion_matrix(metrics['confusion_matrix'], os.path.join(eval_output_dir, f"confusion_matrix{output_suffix}.png"))
    
    return metrics

def save_prediction_visualization(image, mask, prediction, save_path):
    """Save visualization of image, ground truth and prediction."""
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot ground truth mask
    axes[1].imshow(mask, cmap='binary')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot prediction
    axes[2].imshow(prediction, cmap='binary')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def save_roc_curve(fpr, tpr, roc_auc, save_path):
    """Save ROC curve plot."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return save_path

def save_pr_curve(precision, recall, pr_auc, save_path):
    """Save precision-recall curve plot."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return save_path

def save_confusion_matrix(confusion_matrix, save_path):
    """Save confusion matrix visualization."""
    # Normalize matrix
    norm_confusion = confusion_matrix / np.maximum(confusion_matrix.sum(axis=1, keepdims=True), 1)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(norm_confusion, cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Background', 'Vessels'])
    plt.yticks([0, 1], ['Background', 'Vessels'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{norm_confusion[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if norm_confusion[i, j] < 0.7 else "white")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return save_path

def predict_on_volume(model, input_volume):
    """Apply model prediction on a 3D volume for binary classification."""
    model.trainable = False
    
    # Initialize output volume
    output_volume = np.zeros_like(input_volume, dtype=np.int64)
    
    # Predict slice by slice
    for z in tqdm(range(input_volume.shape[0]), desc="Predicting on volume"):
        # Extract slice and normalize
        slice_img = input_volume[z]
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        
        # Convert to tensor
        slice_tensor = tf.convert_to_tensor(slice_img, dtype=tf.float32)
        slice_tensor = tf.expand_dims(slice_tensor, axis=0)  # Add batch dimension
        slice_tensor = tf.expand_dims(slice_tensor, axis=-1)  # Add channel dimension
        
        # Predict
        output = model(slice_tensor, training=False)
        prediction = tf.cast(output > 0.5, tf.int32).numpy()[0].squeeze()
        
        output_volume[z] = prediction
    
    return output_volume

if __name__ == "__main__":
    # This code will run if the file is executed directly
    pass