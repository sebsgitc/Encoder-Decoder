import os
import sys
import subprocess
import tensorflow as tf

# Print environment information
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Current LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'Not set'))

# Try to find CUDA libraries
try:
    result = subprocess.run("find / -name 'libcudart.so*' 2>/dev/null", shell=True, capture_output=True, text=True)
    print("\nCUDA libraries found:\n", result.stdout)
except:
    print("Error running find command")

# Check if nvidia-smi is available
try:
    result = subprocess.run("which nvidia-smi", shell=True, capture_output=True, text=True)
    print("\nnvidia-smi path:", result.stdout.strip())
    
    if result.stdout.strip():
        # Get GPU information
        nvidia_smi = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        print("\nnvidia-smi output:\n", nvidia_smi.stdout)
except:
    print("Error checking for nvidia-smi")

# Check TensorFlow GPU detection
print("\nTF GPU available:", tf.config.list_physical_devices('GPU'))
print("TF CUDA built with:", tf.sysconfig.get_build_info())