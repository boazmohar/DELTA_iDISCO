#!/usr/bin/env python3
"""
Test script to demonstrate affine-like behavior using DiffeomorphicDemonsRegistration
with large grid parameters, as an alternative to ElastixRegistration_affine.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import h5py
import time
import tifffile
from scipy import ndimage

# Try to import pirt
try:
    import pirt
    print("Successfully imported pirt modules")
except ImportError as e:
    print(f"Error importing pirt: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Test affine-like registration with pirt')
    parser.add_argument('--reference_h5_file', type=str, required=True,
                        help='Path to the reference h5 file')
    parser.add_argument('--moving_h5_file', type=str, required=True,
                        help='Path to the moving h5 file to register')
    parser.add_argument('--output_dir', type=str, default='./registration_output',
                        help='Output directory for visualization files')
    parser.add_argument('--dataset', type=str, default='Data_4_4_4',
                        choices=['Data', 'Data_2_2_2', 'Data_4_4_4'],
                        help='Dataset to use from the h5 files')
    parser.add_argument('--filter_sigma', type=float, default=1.0,
                        help='Sigma value for Gaussian filter (0 to disable)')
    parser.add_argument('--median_size', type=int, default=0,
                        help='Size of median filter kernel (0 to disable)')
    parser.add_argument('--save_filtered', action='store_true',
                        help='Save the filtered images as TIFF files')
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

def save_3d_image(img, name, output_dir):
    """Save a 3D image as a TIFF file
    
    Args:
        img: 3D image array
        name: Name for the output file
        output_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to uint16 for TIFF file
    def to_uint16(img):
        # Rescale to full uint16 range
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            scaled = (img - img_min) / (img_max - img_min) * 65535
        else:
            scaled = img
        return scaled.astype(np.uint16)
    
    # Save as TIFF file using tifffile
    file_path = os.path.join(output_dir, f"{name}.tif")
    tifffile.imwrite(file_path, to_uint16(img))
    
    print(f"Saved 3D image: {file_path}")
    return file_path


def perform_affine_like_registration(img1, img2):
    """
    Perform registration with parameters that mimic affine behavior
    using DiffeomorphicDemonsRegistration with large grid settings.
    
    Args:
        img1: Reference image
        img2: Moving image
        
    Returns:
        transformed: Transformed image
    """
    print("Setting up registration with affine-like parameters...")
    
    # Create registration object
    reg = pirt.DiffeomorphicDemonsRegistration(img1, img2)
    
    # Set parameters to mimic affine behavior
    # Using large grid sampling for global transformation (affine-like)
    reg.params.grid_sampling_factor = 8.0  # Very large grid
    reg.params.scale_sampling = 20         # Moderate iterations
    reg.params.speed_factor = 3.0          # Higher speed
    reg.params.final_scale = 1.0           # Standard
    reg.params.scale_levels = 4            # Reasonable level count
    reg.params.final_grid_sampling = 8.0   # Keep grid large throughout
    reg.params.noise_factor = 1.0          # Standard
    
    print("Starting registration...")
    start_time = time.time()
    reg.register(verbose=1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Registration completed in {elapsed_time:.2f} seconds!")
    
    # Get deformation field and apply it
    deform = reg.get_deform(1)  # Deform for the moving image (img2)
    transformed = deform.apply_deformation(img2)
    
    return transformed, deform, elapsed_time

def apply_filters(img, sigma=0, median_size=0):
    """Apply filtering to reduce noise and artifacts
    
    Args:
        img: Input 3D image array
        sigma: Sigma for Gaussian filter (0 to disable)
        median_size: Size of median filter kernel (0 to disable)
        
    Returns:
        Filtered image
    """
    filtered = img.copy()
    
    # Apply Gaussian filter if sigma > 0
    if sigma > 0:
        print(f"Applying Gaussian filter with sigma={sigma}...")
        filtered = ndimage.gaussian_filter(filtered, sigma=sigma)
    
    # Apply median filter if median_size > 0
    if median_size > 0:
        print(f"Applying median filter with size={median_size}...")
        filtered = ndimage.median_filter(filtered, size=median_size)
    
    return filtered

def main():
    args = parse_args()
    
    # Print environment info
    print("\n=== Environment Information ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Filter settings: Gaussian sigma={args.filter_sigma}, Median size={args.median_size}")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Read reference and moving images from h5 files
    print("\n=== Loading Images ===")
    print(f"Reference file: {args.reference_h5_file}")
    print(f"Moving file: {args.moving_h5_file}")
    
    # Load the reference image
    img1_orig = read_h5_image(args.reference_h5_file, args.dataset, 100)
    if img1_orig is None:
        print(f"Failed to read {args.dataset} from reference file.")
        sys.exit(1)
    
    # Load the moving image
    img2_orig = read_h5_image(args.moving_h5_file, args.dataset, 100)
    if img2_orig is None:
        print(f"Failed to read {args.dataset} from moving file.")
        sys.exit(1)
    
    print(f"Loaded reference image with shape {img1_orig.shape}")
    print(f"Loaded moving image with shape {img2_orig.shape}")
    
    # Check if images have the same shape
    if img1_orig.shape != img2_orig.shape:
        print("Warning: Images have different shapes. Resizing moving image to match reference...")
        from skimage import transform
        img2_orig = transform.resize(img2_orig, img1_orig.shape, anti_aliasing=True, preserve_range=True)
        print(f"Resized moving image to shape {img2_orig.shape}")
    
    # Save original 3D images
    print("\n=== Saving Original 3D Images ===")
    save_3d_image(img1_orig, "reference_original", output_dir)
    save_3d_image(img2_orig, "moving_original", output_dir)
    
    # Apply filtering to reduce artifacts
    print("\n=== Applying Filters to Reduce Artifacts ===")
    img1 = apply_filters(img1_orig, sigma=args.filter_sigma, median_size=args.median_size)
    img2 = apply_filters(img2_orig, sigma=args.filter_sigma, median_size=args.median_size)
    
    # Save filtered images if requested
    if args.save_filtered:
        print("Saving filtered images...")
        os.makedirs(os.path.join(output_dir, "filtered"), exist_ok=True)
        
        # Convert to uint16 for saving
        def to_uint16(img):
            # Rescale to full uint16 range
            img_min, img_max = np.min(img), np.max(img)
            if img_max > img_min:
                scaled = (img - img_min) / (img_max - img_min) * 65535
            else:
                scaled = img
            return scaled.astype(np.uint16)
        
        # Save the 3D filtered images
        tifffile.imwrite(os.path.join(output_dir, "filtered", "reference_filtered.tif"), to_uint16(img1))
        tifffile.imwrite(os.path.join(output_dir, "filtered", "moving_filtered.tif"), to_uint16(img2))
        
        # Save filtered 3D images
        save_3d_image(img1, "reference_filtered", output_dir)
        save_3d_image(img2, "moving_filtered", output_dir)
    
    # Test registration
    print("\n=== Testing Affine-like Registration ===")
    try:
        transformed, deform, elapsed_time = perform_affine_like_registration(img1, img2)
        print(f"Registration successful. Transformed image shape: {transformed.shape}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        
        # Save registered 3D image
        print("\n=== Saving Registered 3D Image ===")
        save_3d_image(transformed, "registered", output_dir)
        
        # Calculate similarity before and after registration
        mse_before = np.mean((img1 - img2) ** 2)
        mse_after = np.mean((img1 - transformed) ** 2)
        
        improvement = (1 - mse_after / mse_before) * 100
        print(f"Mean squared error before registration: {mse_before:.4f}")
        print(f"Mean squared error after registration: {mse_after:.4f}")
        print(f"Improvement: {improvement:.2f}%")
        
        print("\nRegistration test completed successfully!")
        print(f"Output files are in: {output_dir}")
        return 0
    
    except Exception as e:
        print(f"Error during registration: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())