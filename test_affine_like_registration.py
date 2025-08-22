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

def create_max_projections(img, prefix, output_dir):
    import tifffile
    """Create and save maximum intensity projections of a 3D image
    
    Args:
        img: 3D image array
        prefix: Prefix for the output filenames
        output_dir: Directory to save the projections
        
    Returns:
        List of paths to the saved projections
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create maximum intensity projections
    max_xy = np.max(img, axis=0)  # Max along Z-axis
    max_xz = np.max(img, axis=1)  # Max along Y-axis
    max_yz = np.max(img, axis=2)  # Max along X-axis
    
    # Convert to uint16 for TIFF files
    def to_uint16(img):
        # Rescale to full uint16 range
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            scaled = (img - img_min) / (img_max - img_min) * 65535
        else:
            scaled = img
        return scaled.astype(np.uint16)
    
    # Convert projections to uint16
    max_xy_uint16 = to_uint16(max_xy)
    max_xz_uint16 = to_uint16(max_xz)
    max_yz_uint16 = to_uint16(max_yz)
    
    # Save as TIFF files using tifffile
    xy_path = os.path.join(output_dir, f"{prefix}_xy_projection.tif")
    xz_path = os.path.join(output_dir, f"{prefix}_xz_projection.tif")
    yz_path = os.path.join(output_dir, f"{prefix}_yz_projection.tif")
    
    tifffile.imwrite(xy_path, max_xy_uint16)
    tifffile.imwrite(xz_path, max_xz_uint16)
    tifffile.imwrite(yz_path, max_yz_uint16)
    
    print(f"Saved TIFF projections for {prefix}:")
    print(f"  XY: {xy_path}")
    print(f"  XZ: {xz_path}")
    print(f"  YZ: {yz_path}")
    
    # Also create visualization PNGs for quick viewing
    import matplotlib.pyplot as plt
    
    # XY projection (axial)
    plt.figure(figsize=(10, 10))
    plt.imshow(max_xy, cmap='gray')
    plt.title(f"{prefix} - XY Projection (Axial)")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, f"{prefix}_xy_projection_viz.png"), dpi=150)
    plt.close()
    
    # XZ projection (coronal)
    plt.figure(figsize=(10, 5))
    plt.imshow(max_xz, cmap='gray')
    plt.title(f"{prefix} - XZ Projection (Coronal)")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, f"{prefix}_xz_projection_viz.png"), dpi=150)
    plt.close()
    
    # YZ projection (sagittal)
    plt.figure(figsize=(5, 10))
    plt.imshow(max_yz, cmap='gray')
    plt.title(f"{prefix} - YZ Projection (Sagittal)")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, f"{prefix}_yz_projection_viz.png"), dpi=150)
    plt.close()
    
    return [xy_path, xz_path, yz_path]

def create_comparison_figure(ref_img, mov_img, reg_img, output_dir):
    """Create a comparison figure showing all three images
    
    Args:
        ref_img: Reference image
        mov_img: Moving image before registration
        reg_img: Moving image after registration
        output_dir: Directory to save the figure
        
    Returns:
        Path to the saved figure
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create maximum intensity projections for all three images
    max_xy_ref = np.max(ref_img, axis=0)  # Reference XY
    max_xy_mov = np.max(mov_img, axis=0)  # Moving XY
    max_xy_reg = np.max(reg_img, axis=0)  # Registered XY
    
    max_xz_ref = np.max(ref_img, axis=1)  # Reference XZ
    max_xz_mov = np.max(mov_img, axis=1)  # Moving XZ
    max_xz_reg = np.max(reg_img, axis=1)  # Registered XZ
    
    # Convert to uint16 for TIFF files
    def to_uint16(img):
        # Rescale to full uint16 range
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            scaled = (img - img_min) / (img_max - img_min) * 65535
        else:
            scaled = img
        return scaled.astype(np.uint16)
    
    # Save each projection as a separate TIFF file
    tifffile.imwrite(os.path.join(output_dir, "comparison_xy_reference.tif"), to_uint16(max_xy_ref))
    tifffile.imwrite(os.path.join(output_dir, "comparison_xy_moving.tif"), to_uint16(max_xy_mov))
    tifffile.imwrite(os.path.join(output_dir, "comparison_xy_registered.tif"), to_uint16(max_xy_reg))
    
    tifffile.imwrite(os.path.join(output_dir, "comparison_xz_reference.tif"), to_uint16(max_xz_ref))
    tifffile.imwrite(os.path.join(output_dir, "comparison_xz_moving.tif"), to_uint16(max_xz_mov))
    tifffile.imwrite(os.path.join(output_dir, "comparison_xz_registered.tif"), to_uint16(max_xz_reg))
    
    # Also save the original 3D registered image
    tifffile.imwrite(os.path.join(output_dir, "registered_3d.tif"), to_uint16(reg_img))
    
    # Normalize for better visualization in PNG
    def normalize(img):
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img
    
    # Normalize all projections for PNG visualization
    projections = [max_xy_ref, max_xy_mov, max_xy_reg, max_xz_ref, max_xz_mov, max_xz_reg]
    normalized = [normalize(proj) for proj in projections]
    
    max_xy_ref, max_xy_mov, max_xy_reg, max_xz_ref, max_xz_mov, max_xz_reg = normalized
    
    # Create the comparison figure for visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # XY projections
    axes[0, 0].imshow(max_xy_ref, cmap='gray')
    axes[0, 0].set_title('Reference - XY')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(max_xy_mov, cmap='gray')
    axes[0, 1].set_title('Moving - XY')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(max_xy_reg, cmap='gray')
    axes[0, 2].set_title('Registered - XY')
    axes[0, 2].axis('off')
    
    # XZ projections
    axes[1, 0].imshow(max_xz_ref, cmap='gray')
    axes[1, 0].set_title('Reference - XZ')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(max_xz_mov, cmap='gray')
    axes[1, 1].set_title('Moving - XZ')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(max_xz_reg, cmap='gray')
    axes[1, 2].set_title('Registered - XZ')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "registration_comparison.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    
    print("Saved comparison files:")
    print("  PNG visualization: registration_comparison.png")
    print("  TIFF projections: comparison_*.tif")
    print("  3D registered image: registered_3d.tif")
    
    return fig_path

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

def main():
    args = parse_args()
    
    # Print environment info
    print("\n=== Environment Information ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Read reference and moving images from h5 files
    print("\n=== Loading Images ===")
    print(f"Reference file: {args.reference_h5_file}")
    print(f"Moving file: {args.moving_h5_file}")
    
    # Load the reference image
    img1 = read_h5_image(args.reference_h5_file, args.dataset, 100)
    if img1 is None:
        print(f"Failed to read {args.dataset} from reference file.")
        sys.exit(1)
    
    # Load the moving image
    img2 = read_h5_image(args.moving_h5_file, args.dataset, 100)
    if img2 is None:
        print(f"Failed to read {args.dataset} from moving file.")
        sys.exit(1)
    
    print(f"Loaded reference image with shape {img1.shape}")
    print(f"Loaded moving image with shape {img2.shape}")
    
    # Check if images have the same shape
    if img1.shape != img2.shape:
        print("Warning: Images have different shapes. Resizing moving image to match reference...")
        from skimage import transform
        img2 = transform.resize(img2, img1.shape, anti_aliasing=True, preserve_range=True)
        print(f"Resized moving image to shape {img2.shape}")
    
    # Create max projections of original images
    print("\n=== Creating Max Projections of Original Images ===")
    create_max_projections(img1, "reference", output_dir)
    create_max_projections(img2, "moving", output_dir)
    
    # Test registration
    print("\n=== Testing Affine-like Registration ===")
    try:
        transformed, deform, elapsed_time = perform_affine_like_registration(img1, img2)
        print(f"Registration successful. Transformed image shape: {transformed.shape}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        
        # Create max projections of registered image
        print("\n=== Creating Max Projections of Registered Image ===")
        create_max_projections(transformed, "registered", output_dir)
        
        # Create comparison figure
        print("\n=== Creating Comparison Figure ===")
        comparison_path = create_comparison_figure(img1, img2, transformed, output_dir)
        print(f"Saved comparison figure to: {comparison_path}")
        
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