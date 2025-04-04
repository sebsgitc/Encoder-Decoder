#!/usr/bin/env python3
"""
Wrapper script to run segmentation with proper environment settings
to avoid cuDNN version mismatch issues.
"""
import os
import sys
import subprocess

# Disable TensorFlow's built-in cuDNN to avoid version mismatch
os.environ['TF_DISABLE_CUDNN'] = '1'
# Disable cuBLAS and cuFFT to prevent registration errors
os.environ['TF_DISABLE_CUBLAS'] = '1'  
os.environ['TF_DISABLE_CUFFT'] = '1'
print("Set environment variables to avoid library conflicts")

# Keep GPU available but avoid using cuDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF logging noise
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only use first GPU to avoid conflicts

# Print environment settings
print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")
print("Environment variables:")
for key in ['TF_DISABLE_CUDNN', 'TF_DISABLE_CUBLAS', 'TF_DISABLE_CUFFT', 
            'CUDA_VISIBLE_DEVICES', 'TF_CPP_MIN_LOG_LEVEL']:
    print(f"  {key}={os.environ.get(key, 'not set')}")

# Run a simple test to check SimpleITK works correctly
print("\nTesting SimpleITK...")
test_cmd = [
    sys.executable, 
    '-c', 
    'import SimpleITK as sitk; import numpy as np; '
    'img = sitk.GetImageFromArray(np.zeros((10,10,10), dtype=np.uint8)); '
    'print("SimpleITK test successful")'
]

try:
    subprocess.run(test_cmd, env=os.environ, check=True)
except Exception as e:
    print(f"SimpleITK test failed: {e}")
    print("Continuing anyway...")

# Run the actual segmentation script
print("\nRunning segmentation with adjusted environment settings...\n")
try:
    # Use the simple vessel segmentation instead which is more reliable
    script_to_run = 'simple_vessel_segmentation.py'
    print(f"Running segmentation script: {script_to_run}")
    
    result = subprocess.run([sys.executable, script_to_run], 
                           env=os.environ, 
                           check=True)
    print(f"\nSegmentation completed with exit code: {result.returncode}")
    sys.exit(result.returncode)
except subprocess.CalledProcessError as e:
    print(f"\nSegmentation failed with exit code: {e.returncode}")
    sys.exit(e.returncode)
except Exception as e:
    print(f"\nError running segmentation: {str(e)}")
    sys.exit(1)
