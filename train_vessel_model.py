#!/usr/bin/env python3
"""
Train a 3D U-Net model for vessel segmentation using prepared patch data.
Uses CPU by default to avoid CUDA compatibility issues.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import argparse
import glob
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import datetime

# Force CPU usage to avoid CUDA compatibility issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_dataset(data_dir, batch_size=8, val_split=0.2):
    """
    Load dataset from prepared patches
    
    Args:
        data_dir: Directory containing input/labels folders
        batch_size: Batch size for training
        val_split: Portion of data to use for validation
        
    Returns:
        train_dataset, val_dataset
    """
    print(f"Loading dataset from {data_dir}")
    
    # Get file lists
    input_files = sorted(glob.glob(os.path.join(data_dir, "inputs", "*.npy")))
    label_files = sorted(glob.glob(os.path.join(data_dir, "labels", "*.npy")))
    
    print(f"Found {len(input_files)} input files and {len(label_files)} label files")
    
    if len(input_files) == 0 or len(label_files) == 0:
        raise ValueError("No training files found. Please run prepare_training_data.py first.")
    
    # Split into training and validation
    split_idx = int(len(input_files) * (1 - val_split))
    train_inputs = input_files[:split_idx]
    train_labels = label_files[:split_idx]
    val_inputs = input_files[split_idx:]
    val_labels = label_files[split_idx:]
    
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(val_inputs)}")
    
    # Create data generators to load efficiently
    def data_generator(input_files, label_files, batch_size):
        num_samples = len(input_files)
        indices = np.arange(num_samples)
        
        while True:
            # Shuffle at epoch start
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Load batch data
                batch_inputs = []
                batch_labels = []
                
                for idx in batch_indices:
                    # Load arrays
                    x = np.load(input_files[idx])
                    y = np.load(label_files[idx])
                    
                    # Normalize inputs to [0,1]
                    x = x.astype(np.float32) / 255.0
                    
                    # Add channel dimension if needed
                    if x.ndim == 3:
                        x = np.expand_dims(x, axis=-1)
                    if y.ndim == 3:
                        y = np.expand_dims(y, axis=-1)
                    
                    batch_inputs.append(x)
                    batch_labels.append(y)
                
                yield np.array(batch_inputs), np.array(batch_labels)
    
    # Create datasets
    train_steps = len(train_inputs) // batch_size
    val_steps = len(val_inputs) // batch_size
    
    if train_steps == 0 or val_steps == 0:
        raise ValueError(f"Batch size ({batch_size}) too large for dataset size")
    
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_inputs, train_labels, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, None, None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None, None, 1), dtype=tf.float32)
        )
    )
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(val_inputs, val_labels, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, None, None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None, None, 1), dtype=tf.float32)
        )
    )
    
    return train_dataset, val_dataset, train_steps, val_steps

def create_3d_unet(input_shape=(64, 64, 64, 1), filters_base=16):
    """
    Create a 3D U-Net model for vessel segmentation
    """
    print(f"Creating 3D U-Net with input shape {input_shape}")
    
    # Input layer
    inputs = layers.Input(input_shape)
    
    # Contracting path (encoder)
    # Level 1
    conv1 = layers.Conv3D(filters_base, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(filters_base, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    # Level 2
    conv2 = layers.Conv3D(filters_base*2, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(filters_base*2, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    # Level 3
    conv3 = layers.Conv3D(filters_base*4, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(filters_base*4, 3, activation='relu', padding='same')(conv3)
    
    # Expansive path (decoder)
    # Level 2
    up2 = layers.UpSampling3D(size=(2, 2, 2))(conv3)
    concat2 = layers.Concatenate()([up2, conv2])
    conv4 = layers.Conv3D(filters_base*2, 3, activation='relu', padding='same')(concat2)
    conv4 = layers.Conv3D(filters_base*2, 3, activation='relu', padding='same')(conv4)
    
    # Level 1
    up1 = layers.UpSampling3D(size=(2, 2, 2))(conv4)
    concat1 = layers.Concatenate()([up1, conv1])
    conv5 = layers.Conv3D(filters_base, 3, activation='relu', padding='same')(concat1)
    conv5 = layers.Conv3D(filters_base, 3, activation='relu', padding='same')(conv5)
    
    # Output layer
    outputs = layers.Conv3D(1, 1, activation='sigmoid')(conv5)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def dice_coefficient(y_true, y_pred):
    """Dice coefficient for binary segmentation"""
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss for binary segmentation"""
    return 1.0 - dice_coefficient(y_true, y_pred)

def plot_training_history(history, save_path):
    """Plot training history and save to file"""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot dice coefficient
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'])
    plt.plot(history.history['val_dice_coefficient'])
    plt.title('Dice Coefficient')
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for vessel segmentation")
    parser.add_argument("--data-dir", default="datasets/vessel_patches", 
                       help="Directory containing prepared training data")
    parser.add_argument("--output-dir", default="models", 
                       help="Output directory for trained model")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=30, 
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--patch-size", type=int, default=64, 
                       help="Size of patches (cubic dimension)")
    parser.add_argument("--use-cpu", action="store_true", 
                       help="Force CPU usage (avoid CUDA compatibility issues)")
    
    args = parser.parse_args()
    
    # Apply CPU restriction if requested
    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Forcing CPU usage as requested")
    
    # Check available devices
    print("Available devices:")
    for i, device in enumerate(tf.config.list_physical_devices()):
        print(f"  {i}: {device.name}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    train_dataset, val_dataset, train_steps, val_steps = load_dataset(
        args.data_dir, 
        batch_size=args.batch_size
    )
    
    # Create model
    input_shape = (args.patch_size, args.patch_size, args.patch_size, 1)
    model = create_3d_unet(input_shape=input_shape, filters_base=16)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=dice_loss,
        metrics=[dice_coefficient, 'binary_accuracy']
    )
    
    model.summary()
    
    # Create unique model name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"vessel_segmentation_3d_unet_{timestamp}"
    model_path = os.path.join(args.output_dir, f"{model_name}.h5")
    
    # Create callbacks
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, 
            save_best_only=True,
            monitor='val_dice_coefficient',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs', model_name),
            write_graph=False
        )
    ]
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        epochs=args.epochs,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks_list
    )
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes")
    
    # Save training history plot
    history_plot_path = os.path.join(args.output_dir, f"{model_name}_history.png")
    plot_training_history(history, history_plot_path)
    
    # Save final model if not already saved by callbacks
    model.save(model_path.replace(".h5", "_final.h5"))
    
    print(f"Model saved to: {model_path}")
    print(f"Training plot saved to: {history_plot_path}")
    print("Done!")

if __name__ == "__main__":
    main()
