#!/usr/bin/env python3
import os
import sys
import subprocess

# Set environment variables BEFORE anything else
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu'
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Print environment for debugging
print("LD_LIBRARY_PATH set to:", os.environ.get('LD_LIBRARY_PATH'))

# Run the original script with all arguments passed through
cmd = [sys.executable, 'deep_vessel_segmentation.py'] + sys.argv[1:]
print("Running command:", ' '.join(cmd))
sys.exit(subprocess.call(cmd))