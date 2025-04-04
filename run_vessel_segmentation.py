#!/usr/bin/env python3
"""
Safe wrapper script to run vessel segmentation without library conflicts
"""
import os
import sys
import subprocess
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run vessel segmentation safely")
    parser.add_argument("--input", default="3d-stacks/r01_/r01_.8bit.tif", 
                      help="Input file path")
    parser.add_argument("--output", default="output/vessel_segmentation/result.tif", 
                      help="Output file path")
    args = parser.parse_args()
    
    # Disable TensorFlow libraries that cause conflicts
    env = os.environ.copy()
    env['TF_DISABLE_CUDNN'] = '1'
    env['TF_DISABLE_CUBLAS'] = '1'  
    env['TF_DISABLE_CUFFT'] = '1'
    env['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF logging noise
    
    # Only use first GPU
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    print("=== Vessel Segmentation Runner ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("Environment variables set to avoid library conflicts")
    
    # First run a quick test to check if SimpleITK works
    print("\nRunning SimpleITK test...")
    test_cmd = [
        sys.executable, 
        '-c', 
        'import SimpleITK as sitk; import numpy as np; '
        'print("SimpleITK version:", sitk.Version.VersionString()); '
        'img = sitk.Image(10,10,10, sitk.sitkUInt8); '
        'print("SimpleITK test successful - created image with size:", img.GetSize())'
    ]
    
    try:
        subprocess.run(test_cmd, env=env, check=True)
        print("SimpleITK test passed!")
    except subprocess.CalledProcessError as e:
        print(f"SimpleITK test failed with exit code {e.returncode}")
        print("Will attempt segmentation anyway...")
    except Exception as e:
        print(f"SimpleITK test error: {e}")
    
    # Run the segmentation script with our arguments
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segment_vessels.py")
    
    if not os.path.exists(script_path):
        print(f"Error: Segmentation script not found at {script_path}")
        sys.exit(1)
        
    print(f"\nRunning segmentation script: {script_path}")
    cmd = [
        sys.executable,
        script_path,
        "--input", args.input,
        "--output", args.output
    ]
    
    try:
        print(f"Executing: {' '.join(cmd)}")
        process = subprocess.run(cmd, env=env)
        
        if process.returncode == 0:
            print(f"\nSegmentation completed successfully!")
            print(f"Results saved to: {args.output}")
            return 0
        else:
            print(f"\nSegmentation failed with exit code: {process.returncode}")
            return process.returncode
            
    except Exception as e:
        print(f"\nError running segmentation: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
