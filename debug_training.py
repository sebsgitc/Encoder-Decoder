#!/usr/bin/env python3
"""
Debug script to analyze and fix NaN issues in a TensorFlow model.
This can be run separately to examine a model and diagnose the problem.
"""

import os
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model
import traceback
import matplotlib.pyplot as plt
import pickle
import tifffile as tiff

def forward_pass_safeguard(tensor):
    """Apply to model outputs to prevent NaN propagation"""
    return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)

def check_model_numerical_stability(model_path, test_input_shape=(1, 64, 64, 64, 2)):
    """
    Checks a model for numerical stability issues
    """
    print(f"Loading model from {model_path}")
    try:
        # Custom objects for model loading
        custom_objects = {
            'forward_pass_safeguard': forward_pass_safeguard
        }
        
        # Load the model with our custom objects
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return
    
    # Create random test input
    print(f"Creating test input with shape {test_input_shape}")
    np.random.seed(42)  # For reproducibility
    test_input = np.random.random(test_input_shape).astype(np.float32)
    
    # Check forward pass
    print("Testing forward pass...")
    try:
        output = model.predict(test_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
        print(f"Output stats: min={np.min(output)}, max={np.max(output)}, mean={np.mean(output)}")
        
        # Check for NaN or Inf values
        if np.isnan(output).any() or np.isinf(output).any():
            print(" Output contains NaN or Inf values")
        else:
            print(" Output contains no NaN or Inf values")
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        traceback.print_exc()
    
    # Check each layer individually
    print("\nChecking each layer individually...")
    
    # Create per-layer test models
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        
        # Skip input layers
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        
        try:
            # Create a model that outputs this layer's activations
            layer_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
            
            # Try a forward pass
            layer_output = layer_model.predict(test_input)
            
            # Check layer output
            has_nan = np.isnan(layer_output).any()
            has_inf = np.isinf(layer_output).any()
            
            if has_nan or has_inf:
                print(f" Layer {i}: {layer_name} - Output {'has NaN' if has_nan else 'has Inf'}")
            else:
                # Only print details for the last few layers
                if i >= len(model.layers) - 5:
                    print(f"Layer {i}: {layer_name} - Output shape: {layer_output.shape}")
                    print(f"  Stats: min={np.min(layer_output):.6f}, max={np.max(layer_output):.6f}, "
                          f"mean={np.mean(layer_output):.6f}")
                
        except Exception as e:
            print(f"Error checking layer {i} ({layer_name}): {e}")
    
    # Analyze weight values
    print("\nAnalyzing model weights...")
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            weights = layer.kernel.numpy()
            bias = layer.bias.numpy() if hasattr(layer, 'bias') and layer.bias is not None else None
            
            print(f"Layer {i}: {layer.name}")
            print(f"  Kernel shape: {weights.shape}")
            print(f"  Kernel stats: min={np.min(weights):.6f}, max={np.max(weights):.6f}, "
                  f"mean={np.mean(weights):.6f}, std={np.std(weights):.6f}")
            
            # Check for unusual weight distributions
            weight_max_abs = np.max(np.abs(weights))
            if weight_max_abs > 10.0:
                print(f"   Warning: Unusually large weight values (max abs: {weight_max_abs:.2f})")
                
            # Check for NaN/Inf values
            if np.isnan(weights).any():
                print(f"   Kernel contains NaN values")
            if np.isinf(weights).any():
                print(f"   Kernel contains Inf values")
                
            # Check bias if available
            if bias is not None:
                if np.isnan(bias).any():
                    print(f"   Bias contains NaN values")
                if np.isinf(bias).any():
                    print(f"   Bias contains Inf values")
    
    # Suggest potential fixes
    print("\nRecommendations based on analysis:")
    print("1. If NaN values are present, consider using a simpler loss function")
    print("2. Reduce the learning rate significantly (e.g., to 1e-5)")
    print("3. Add gradient clipping with a low threshold (e.g., clipnorm=0.1)")
    print("4. If certain layers show NaNs, simplify the model architecture")
    print("5. Check inputs for extreme values and add stronger normalization")

def test_model_inference(model_path, test_volume_path, seed_points=None):
    """
    Test a model's inference capabilities on a small patch of a real volume
    """
    try:
        # Custom objects for model loading
        custom_objects = {
            'forward_pass_safeguard': forward_pass_safeguard
        }
        
        # Load the model with our custom objects
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return
    
    # Load test volume
    print(f"Loading test volume from {test_volume_path}")
    try:
        volume = tiff.imread(test_volume_path)
        print(f"Volume loaded. Shape: {volume.shape}, dtype: {volume.dtype}")
    except Exception as e:
        print(f"Error loading volume: {e}")
        return
    
    # Extract a small test patch centered at a random seed point or at the center
    if seed_points:
        # Use the first seed point
        x, y, z = seed_points[0]
    else:
        # Use center of volume
        z, y, x = [s // 2 for s in volume.shape]
    
    # Extract patch
    patch_size = 64
    z_min = max(0, z - patch_size // 2)
    z_max = min(volume.shape[0], z + patch_size // 2)
    y_min = max(0, y - patch_size // 2)
    y_max = min(volume.shape[1], y + patch_size // 2)
    x_min = max(0, x - patch_size // 2)
    x_max = min(volume.shape[2], x + patch_size // 2)
    
    patch = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Pad if needed
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size or patch.shape[2] < patch_size:
        padded_patch = np.zeros((patch_size, patch_size, patch_size), dtype=patch.dtype)
        padded_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        patch = padded_patch
    
    # Create input with a guidance channel
    patch_norm = patch.astype(np.float32) / 255.0
    guidance = np.zeros_like(patch_norm)
    
    # Add a single guidance point in the center
    center = patch_size // 2
    guidance[center, center, center] = 1.0
    
    # Create model input
    model_input = np.stack([patch_norm, guidance], axis=-1)
    model_input = np.expand_dims(model_input, axis=0)  # Add batch dimension
    
    # Run inference
    print("Running inference on test patch...")
    try:
        prediction = model.predict(model_input)
        print(f"Inference successful. Output shape: {prediction.shape}")
        print(f"Output stats: min={np.min(prediction)}, max={np.max(prediction)}, mean={np.mean(prediction)}")
        
        # Check for NaN
        if np.isnan(prediction).any():
            print(" Prediction contains NaN values")
        else:
            print(" Prediction contains no NaN values")
            
        # Visualize middle slice
        plt.figure(figsize=(12, 4))
        
        # Input volume
        plt.subplot(1, 3, 1)
        plt.imshow(model_input[0, patch_size//2, :, :, 0], cmap='gray')
        plt.title("Input Volume")
        plt.axis('off')
        
        # Guidance
        plt.subplot(1, 3, 2)
        plt.imshow(model_input[0, patch_size//2, :, :, 1], cmap='jet')
        plt.title("Guidance")
        plt.axis('off')
        
        # Prediction
        plt.subplot(1, 3, 3)
        plt.imshow(prediction[0, patch_size//2, :, :, 0], cmap='gray')
        plt.title("Prediction")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("test_inference_result.png")
        print("Visualization saved to test_inference_result.png")
            
    except Exception as e:
        print(f"Error during inference: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug TensorFlow model numerical stability")
    parser.add_argument("--model-path", required=True, help="Path to the model file (.h5)")
    parser.add_argument("--test-volume", help="Path to a test volume for inference testing")
    parser.add_argument("--test-inference", action="store_true", help="Test model inference on a real volume")
    
    args = parser.parse_args()
    
    # Set up TensorFlow for better debugging
    tf.config.run_functions_eagerly(True)
    
    # Fix for TensorFlow memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    
    # Check model numerical stability
    check_model_numerical_stability(args.model_path)
    
    # Test inference if requested
    if args.test_inference and args.test_volume:
        test_model_inference(args.model_path, args.test_volume)
