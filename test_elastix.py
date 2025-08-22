#!/usr/bin/env python3
"""
Test script to check if Elastix is properly configured.
This script attempts to create and use an ElastixRegistration_affine object.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import h5py
import time

# Try to import pirt
try:
    import pirt
    from pirt.reg import ElastixRegistration_affine
    print("Successfully imported pirt modules")
except ImportError as e:
    print(f"Error importing pirt: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Test Elastix configuration')
    parser.add_argument('--elastix_path', type=str, required=True,
                        help='Path to elastix executable directory')
    parser.add_argument('--test_h5_file', type=str, required=True,
                        help='Path to an h5 file to test with')
    return parser.parse_args()

def read_h5_image(file_path, dataset_name='Data', threshold=100):
    """Read an image from an h5 file"""
    try:
        print(f"Opening h5 file: {file_path}")
        with h5py.File(file_path, 'r') as h5_file:
            # List available datasets
            print(f"Available datasets: {list(h5_file.keys())}")
            
            # Check if the requested dataset exists
            if dataset_name not in h5_file:
                print(f"Dataset '{dataset_name}' not found. Available datasets: {list(h5_file.keys())}")
                return None
                
            print(f"Reading dataset: {dataset_name}")
            data = h5_file[dataset_name][:]
            print(f"Data shape: {data.shape}, dtype: {data.dtype}")
            print(f"Data min: {data.min()}, max: {data.max()}")
            
            # Apply threshold
            data[data < threshold] = threshold
            print(f"After thresholding - min: {data.min()}, max: {data.max()}")
            
            return data
    except Exception as e:
        print(f"Error reading h5 file: {e}")
        return None

def main():
    args = parse_args()
    
    # Print environment info
    print("\n=== Environment Information ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Print ELASTIX_PATH if already set
    if 'ELASTIX_PATH' in os.environ:
        print(f"ELASTIX_PATH from environment: {os.environ['ELASTIX_PATH']}")
    else:
        print("ELASTIX_PATH not set in environment")
    
    # Set ELASTIX_PATH from argument
    elastix_path = args.elastix_path
    os.environ['ELASTIX_PATH'] = elastix_path
    print(f"Setting ELASTIX_PATH to: {elastix_path}")
    
    # Check if elastix directory exists
    elastix_dir = Path(elastix_path)
    if not elastix_dir.exists():
        print(f"Error: Elastix directory {elastix_path} does not exist!")
        sys.exit(1)
    
    # Check for elastix executable
    elastix_exe = elastix_dir / "elastix"
    if not elastix_exe.exists():
        # On some systems it might not have an extension
        elastix_exe_alt = elastix_dir / "elastix.exe"
        if not elastix_exe_alt.exists():
            print(f"Error: Elastix executable not found at {elastix_exe} or {elastix_exe_alt}")
            sys.exit(1)
        else:
            elastix_exe = elastix_exe_alt
    
    print(f"Found elastix executable at: {elastix_exe}")
    
    # Print pirt version info
    print("\n=== PIRT Information ===")
    print(f"PIRT version: {pirt.__version__ if hasattr(pirt, '__version__') else 'unknown'}")
    
    # Get absolute path to pyelastix module to help debug
    if hasattr(pirt, '__file__'):
        print(f"PIRT module location: {Path(pirt.__file__).parent}")
    
    # Read test images from h5 file
    print("\n=== Loading Test Images ===")
    h5_file = args.test_h5_file
    
    # Try both Data_4_4_4 and Data datasets
    img_small = read_h5_image(h5_file, 'Data_4_4_4', 100)
    if img_small is None:
        print("Failed to read Data_4_4_4 dataset, trying Data...")
        img_small = read_h5_image(h5_file, 'Data', 100)
    
    if img_small is None:
        print("Failed to read any valid dataset from the h5 file.")
        sys.exit(1)
    
    # Create a second test image by shifting the first one slightly
    print("\n=== Creating Test Images for Registration ===")
    img1 = img_small.copy()
    img2 = np.zeros_like(img1)
    
    # Shift image2 by 10 pixels in each dimension
    shift = 10
    if img1.ndim == 3:
        img2[shift:, shift:, shift:] = img1[:-shift, :-shift, :-shift]
    else:
        print(f"Unexpected image dimensions: {img1.ndim}")
        sys.exit(1)
    
    print(f"Created two test images with shape {img1.shape}")
    
    # Test Elastix registration
    print("\n=== Testing Elastix Registration ===")
    print("Attempting to create ElastixRegistration_affine object...")
    
    try:
        # Debug: Try to access the ElastixRegistration_affine source code
        import inspect
        print(f"ElastixRegistration_affine source location: {inspect.getfile(ElastixRegistration_affine)}")
    except Exception as e:
        print(f"Could not inspect ElastixRegistration_affine: {e}")
        
    try:
        # Check where pyelastix is looking for elastix
        if hasattr(pirt.reg, 'elastix'):
            elastix_module = pirt.reg.elastix
            if hasattr(elastix_module, '_elastix_exe'):
                print(f"pyelastix is looking for elastix at: {elastix_module._elastix_exe}")
            else:
                print("Could not determine where pyelastix is looking for elastix")
        
        print("Creating registration object...")
        start_time = time.time()
        reg_affine = ElastixRegistration_affine(img1, img2)
        
        print("Running registration...")
        result = reg_affine.register(verbose=1)
        
        elapsed_time = time.time() - start_time
        print(f"Registration completed in {elapsed_time:.2f} seconds!")
        
        print(f"Result shape: {result.shape}")
        print("Registration test successful!")
        
        return 0
    
    except Exception as e:
        print(f"Error during elastix registration: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())