#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Group Brain Registration Script

This script performs 3D group registration for brain images using pirt.DiffeomorphicDemonsRegistration.
It can process multiple 3D brain images (ch0.tif) from different animals and generate a common template.

Usage:
    python register_brains_3d.py --data_dir /path/to/data --output_dir /path/to/output [options]

Author: Claude
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pirt
import tifffile
from pathlib import Path
from skimage import transform, exposure
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3D Group Brain Registration using pirt')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory containing animal folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory for registration results')
    parser.add_argument('--downscale', type=float, default=0.1,
                        help='Factor to downscale images for faster processing (default: 0.1)')
    parser.add_argument('--slice_count', type=int, default=50,
                        help='Number of slices to use from 3D volume (default: 50, use 0 for all slices)')
    parser.add_argument('--grid_sampling_factor', type=int, default=1,
                        help='Grid sampling of the grid at the final level (default: 1)')
    parser.add_argument('--scale_sampling', type=int, default=20,
                        help='Amount of iterations for each level (default: 20)')
    parser.add_argument('--speed_factor', type=int, default=2,
                        help='Relative force of the transform (default: 2)')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Generate and save visualization of registration results')
    
    return parser.parse_args()

def find_brain_images(data_dir):
    """Find all ch0.tif files in the data directory
    
    Args:
        data_dir (Path): Path to the data directory
        
    Returns:
        dict: Dictionary mapping animal IDs to file paths
    """
    brain_images = {}
    
    # List all animal directories
    for animal_dir in data_dir.glob('ANM*'):
        if animal_dir.is_dir():
            # Check if itk/ch0.tif exists
            ch0_file = animal_dir / 'itk' / 'ch0.tif'
            if ch0_file.exists():
                animal_id = animal_dir.name
                brain_images[animal_id] = ch0_file
    
    return brain_images

def load_and_preprocess_image(file_path, downscale_factor=0.1, slice_count=None):
    """Load and preprocess a 3D brain image
    
    Args:
        file_path (Path): Path to the image file
        downscale_factor (float): Factor to downscale the image
        slice_count (int): Number of slices to use from 3D volume (None for all slices)
        
    Returns:
        np.ndarray: Preprocessed image (3D volume)
    """
    # Load image
    print(f"Loading {file_path}...")
    img = tifffile.imread(file_path)
    
    # Get image info
    print(f"  Original shape: {img.shape}, dtype: {img.dtype}")
    
    # Ensure we have a 3D image
    if len(img.shape) != 3:
        raise ValueError(f"Expected a 3D image, got shape {img.shape}")
    
    # Select a subset of slices if requested (for faster processing)
    if slice_count is not None and slice_count > 0 and slice_count < img.shape[0]:
        # Take evenly spaced slices from the volume
        indices = np.linspace(0, img.shape[0]-1, slice_count).astype(int)
        img = img[indices]
        print(f"  Selected {slice_count} slices, new shape: {img.shape}")
    
    # Rescale to 0-1 float
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    
    # Downscale if needed (this is crucial for 3D registration to be computationally feasible)
    if downscale_factor < 1.0:
        # Calculate new shape, preserving 3D structure
        new_shape = (img.shape[0], 
                     int(img.shape[1] * downscale_factor), 
                     int(img.shape[2] * downscale_factor))
        img = transform.resize(img, new_shape, anti_aliasing=True, preserve_range=True)
        print(f"  Downscaled to: {img.shape}")
    
    return img

def register_brain_images(images, settings):
    """Register 3D brain images using pirt's DiffeomorphicDemonsRegistration
    
    Args:
        images (dict): Dictionary mapping animal IDs to preprocessed 3D images
        settings (dict): Registration settings
        
    Returns:
        tuple: (registration object, deformation fields, transformed images)
    """
    # Convert dictionary to list in a consistent order
    animal_ids = list(images.keys())
    image_list = [images[animal_id] for animal_id in animal_ids]
    
    # Create registration object - pirt can handle 3D volumes directly
    print(f"Registering {len(image_list)} 3D images with shapes: {[img.shape for img in image_list]}")
    reg = pirt.DiffeomorphicDemonsRegistration(*image_list)
    
    # Set registration parameters
    reg.params.grid_sampling_factor = settings['grid_sampling_factor']
    reg.params.scale_sampling = settings['scale_sampling']
    reg.params.speed_factor = settings['speed_factor']
    
    # Perform registration
    print("Starting registration - this may take a while for 3D volumes...")
    reg.register(verbose=1)
    print("Registration completed!")
    
    # Get deformation fields and transformed images
    deforms = []
    transformed_images = {}
    
    for i, animal_id in enumerate(animal_ids):
        print(f"Processing deformation for {animal_id}...")
        # Get deformation field
        deform = reg.get_deform(i)
        deforms.append(deform)
        
        # Transform image
        transformed = deform.apply_deformation(image_list[i])
        transformed_images[animal_id] = transformed
    
    return reg, deforms, transformed_images

def calculate_average_template(transformed_images):
    """Calculate average brain template from registered 3D images
    
    Args:
        transformed_images (dict): Dictionary mapping animal IDs to transformed 3D images
        
    Returns:
        np.ndarray: Average brain template (3D volume)
    """
    # Stack all transformed images
    image_stack = [np.asarray(img) for img in transformed_images.values()]

    # Calculate average
    template = np.mean(np.stack(image_stack), axis=0)
    
    return template

def save_registration_results(transformed_images, template, deforms, animal_ids, output_dir):
    """Save registration results
    
    Args:
        transformed_images (dict): Dictionary mapping animal IDs to transformed 3D images
        template (np.ndarray): Average brain template (3D volume)
        deforms (list): List of deformation field objects
        animal_ids (list): List of animal IDs
        output_dir (Path): Path to the output directory
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save template
    template_path = output_dir / 'template.tif'
    tifffile.imwrite(template_path, template)
    print(f"Saved template to {template_path}")
    
    # Save transformed images
    for animal_id, img in transformed_images.items():
        img_path = output_dir / f"{animal_id}_registered.tif"
        tifffile.imwrite(img_path, img)
        print(f"Saved {animal_id} to {img_path}")
    
    # Save deformation fields
    deform_dir = output_dir / 'deformation_fields'
    deform_dir.mkdir(exist_ok=True)
    
    for i, animal_id in enumerate(animal_ids):
        # For 3D images, save deformation fields for each dimension
        for dim in range(deforms[i].ndim):
            field = deforms[i].get_field(dim)
            np.save(deform_dir / f"{animal_id}_deform_dim{dim}.npy", field)
        print(f"Saved deformation fields for {animal_id}")

def create_visualizations(original_images, transformed_images, template, output_dir):
    """Create and save visualizations of registration results
    
    Args:
        original_images (dict): Dictionary mapping animal IDs to original 3D images
        transformed_images (dict): Dictionary mapping animal IDs to transformed 3D images
        template (np.ndarray): Average brain template (3D volume)
        output_dir (Path): Path to the output directory
    """
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Select slices to visualize
    first_img = list(original_images.values())[0]
    slice_indices = [0, first_img.shape[0]//4, first_img.shape[0]//2, 
                      3*first_img.shape[0]//4, first_img.shape[0]-1]
    
    # Save visualizations for selected slices
    for slice_idx in slice_indices:
        if slice_idx >= first_img.shape[0]:
            continue
            
        # Create figure for this slice
        num_images = len(original_images)
        animal_ids = list(original_images.keys())
        
        # Comparison figure (original vs registered)
        plt.figure(figsize=(15, 5 * num_images))
        
        for i, animal_id in enumerate(animal_ids):
            # Original image
            plt.subplot(num_images, 3, i*3+1)
            plt.imshow(original_images[animal_id][slice_idx], cmap='gray')
            plt.title(f"{animal_id} - Original")
            plt.axis('off')
            
            # Transformed image
            plt.subplot(num_images, 3, i*3+2)
            plt.imshow(transformed_images[animal_id][slice_idx], cmap='gray')
            plt.title(f"{animal_id} - Registered")
            plt.axis('off')
            
            # Difference image
            plt.subplot(num_images, 3, i*3+3)
            diff = np.abs(original_images[animal_id][slice_idx] - transformed_images[animal_id][slice_idx])
            plt.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            plt.title(f"{animal_id} - Difference")
            plt.axis('off')
            plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(viz_dir / f"comparison_slice_{slice_idx}.png", dpi=150)
        plt.close()
        
        # Template for this slice
        plt.figure(figsize=(10, 8))
        plt.imshow(template[slice_idx], cmap='gray')
        plt.title(f"Average Brain Template - Slice {slice_idx}")
        plt.colorbar()
        plt.savefig(viz_dir / f"template_slice_{slice_idx}.png", dpi=150)
        plt.close()
    
    # Create a montage of the template
    montage_slices = np.linspace(0, template.shape[0]-1, min(9, template.shape[0])).astype(int)
    plt.figure(figsize=(15, 15))
    for i, slice_idx in enumerate(montage_slices):
        plt.subplot(3, 3, i+1)
        plt.imshow(template[slice_idx], cmap='gray')
        plt.title(f"Template Slice {slice_idx}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(viz_dir / "template_montage.png", dpi=150)
    plt.close()
    
    print(f"Saved visualizations to {viz_dir}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Convert paths to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist!")
        return 1
    
    # Create registration settings
    settings = {
        'grid_sampling_factor': args.grid_sampling_factor,
        'scale_sampling': args.scale_sampling,
        'speed_factor': args.speed_factor,
    }
    
    # Find all brain images
    brain_images = find_brain_images(data_dir)
    if not brain_images:
        print("Error: No brain images found!")
        return 1
    
    print(f"Found {len(brain_images)} brain images:")
    for animal_id, file_path in brain_images.items():
        print(f"  {animal_id}: {file_path}")
    
    # Handle slice count
    slice_count = args.slice_count if args.slice_count > 0 else None
    
    # Load and preprocess images
    loaded_images = {}
    for animal_id, file_path in tqdm(brain_images.items(), desc="Loading images"):
        loaded_images[animal_id] = load_and_preprocess_image(
            file_path, 
            args.downscale, 
            slice_count
        )
    
    # Register brain images
    reg, deforms, transformed_images = register_brain_images(loaded_images, settings)
    
    # Calculate average template
    template = calculate_average_template(transformed_images)
    
    # Save registration results
    save_registration_results(
        transformed_images, 
        template, 
        deforms, 
        list(brain_images.keys()), 
        output_dir
    )
    
    # Create visualizations if requested
    if args.save_visualizations:
        create_visualizations(loaded_images, transformed_images, template, output_dir)
    
    print("3D registration completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())