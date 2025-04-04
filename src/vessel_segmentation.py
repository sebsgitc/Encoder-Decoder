import os
import numpy as np
import tensorflow as tf
import tifffile as tiff
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Concatenate, Input
from tensorflow.keras.models import Model

def create_3d_unet(input_shape=(64, 64, 64, 1)):
    """Simple 3D UNet for vessel segmentation"""
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv3D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = Conv3D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    # Bridge
    conv3 = Conv3D(128, 3, activation='relu', padding='same')(pool2)
    
    # Decoder
    up1 = UpSampling3D(size=(2, 2, 2))(conv3)
    up1 = Concatenate()([up1, conv2])
    conv4 = Conv3D(64, 3, activation='relu', padding='same')(up1)
    
    up2 = UpSampling3D(size=(2, 2, 2))(conv4)
    up2 = Concatenate()([up2, conv1])
    conv5 = Conv3D(32, 3, activation='relu', padding='same')(up2)
    
    outputs = Conv3D(1, 1, activation='sigmoid')(conv5)
    
    return Model(inputs, outputs)

def process_chunk(model, chunk, has_seed):
    """Process a single 64x64x64 chunk"""
    chunk = tf.expand_dims(chunk, -1)  # Add channel dimension
    prediction = model(tf.expand_dims(chunk, 0), training=False)
    return tf.squeeze(prediction)

def segment_vessels(image_path, seed_points, chunk_size=64, overlap=8):
    """Segment vessels using overlapping 3D chunks"""
    # Load image
    print(f"Loading image from {image_path}")
    image = tiff.imread(image_path)
    print(f"Image shape: {image.shape}")
    
    # Normalize image
    image = tf.cast(image, tf.float32) / 255.0
    
    # Create model
    model = create_3d_unet((chunk_size, chunk_size, chunk_size, 1))
    
    # Prepare output array
    final_mask = np.zeros_like(image, dtype=np.uint8)
    
    # Calculate number of chunks
    z_chunks = (image.shape[0] + chunk_size - 1) // chunk_size
    y_chunks = (image.shape[1] + chunk_size - 1) // chunk_size
    x_chunks = (image.shape[2] + chunk_size - 1) // chunk_size
    
    print(f"Processing {z_chunks}x{y_chunks}x{x_chunks} chunks")
    
    # Process chunks
    for z in range(z_chunks):
        for y in range(y_chunks):
            for x in range(x_chunks):
                # Calculate chunk coordinates
                z_start = max(z * chunk_size - overlap, 0)
                z_end = min((z + 1) * chunk_size + overlap, image.shape[0])
                y_start = max(y * chunk_size - overlap, 0)
                y_end = min((y + 1) * chunk_size + overlap, image.shape[1])
                x_start = max(x * chunk_size - overlap, 0)
                x_end = min((x + 1) * chunk_size + overlap, image.shape[2])
                
                # Extract chunk
                chunk = image[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Check if chunk contains seed points
                chunk_seeds = []
                for sx, sy, sz in seed_points:
                    if (z_start <= sz < z_end and 
                        y_start <= sy < y_end and 
                        x_start <= sx < x_end):
                        chunk_seeds.append((sz-z_start, sy-y_start, sx-x_start))
                
                has_seed = len(chunk_seeds) > 0
                
                # Process chunk
                with tf.device('/GPU:0'):
                    chunk_mask = process_chunk(model, chunk, has_seed)
                
                # Remove overlap regions
                if overlap > 0:
                    z_trim = slice(overlap if z > 0 else 0, 
                                 -overlap if z < z_chunks-1 else None)
                    y_trim = slice(overlap if y > 0 else 0, 
                                 -overlap if y < y_chunks-1 else None)
                    x_trim = slice(overlap if x > 0 else 0, 
                                 -overlap if x < x_chunks-1 else None)
                    chunk_mask = chunk_mask[z_trim, y_trim, x_trim]
                
                # Update final mask
                z_out = slice(z * chunk_size, (z + 1) * chunk_size)
                y_out = slice(y * chunk_size, (y + 1) * chunk_size)
                x_out = slice(x * chunk_size, (x + 1) * chunk_size)
                final_mask[z_out, y_out, x_out] = (chunk_mask.numpy() > 0.5) * 255
                
                print(f"Processed chunk ({z},{y},{x}) with {len(chunk_seeds)} seeds")
    
    return final_mask

if __name__ == "__main__":
    # Configuration
    input_path = "3d-stacks/r01_/r01_.8bit.tif"
    output_path = "output/vessel_segmentation.tif"
    
    seed_points = [(478, 323, 32), (372, 648, 45), (920, 600, 72),
                   (420, 457, 24), (369, 326, 74), (753, 417, 124),
                   (755, 607, 174), (887, 507, 224), (305, 195, 274),
                   (574, 476, 324), (380, 625, 374), (313, 660, 424),
                   (100, 512, 610), (512, 20, 730), (512, 200, 820), 
                   (512, 400, 940)]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process image
    mask = segment_vessels(input_path, seed_points)
    
    # Save result
    tiff.imwrite(output_path, mask)
    print(f"Segmentation saved to {output_path}")