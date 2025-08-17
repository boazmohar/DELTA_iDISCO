#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Group Brain Registration Script

This script performs group registration for brain images using pirt.DiffeomorphicDemonsRegistration.
It can process multiple brain images (ch0.tif) from different animals and generate a common template.

Usage:
    python register_brains.py --data_dir /path/to/data --output_dir /path/to/output [options]

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Group Brain Registration using pirt')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory containing animal folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory for registration results')
    parser.add_argument('--downscale', type=float, default=0.25,
                        help='Factor to downscale images for faster processing (default: 0.25)')
    parser.add_argument('--grid_sampling_factor', type=int, default=1,
                        help='Grid sampling of the grid at the final level (default: 1)')
    parser.add_argument('--scale_sampling', type=int, default=20,
                        help='Amount of iterations for each level (default: 20)')
    parser.add_argument('--speed_factor', type=int, default=2,
                        help='Relative force of the transform (default: 2)')
    parser.add_argument('--slice_idx', type=int, default=None,
                        help='Index of slice to use for 3D images (default: middle slice)')
    parser.add_argument('--use_max_projection', action='store_true',
                        help='Use maximum projection instead of a single slice for 3D images')
    parser.add_argument('--save_deformation', action='store_true',
                        help='Save deformation fields')
    parser.add_argument('--visualize', action='store_true',
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

def load_and_preprocess_image(file_path, downscale_factor=0.25, slice_idx=None, use_max_projection=False):
    """Load and preprocess a brain image
    
    Args:
        file_path (Path): Path to the image file
        downscale_factor (float): Factor to downscale the image
        slice_idx (int): Index of slice to use for 3D images (if None, use middle slice)
        use_max_projection (bool): Use maximum projection instead of a single slice
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Load image
    print(f"Loading {file_path}...")
    img = tifffile.imread(file_path)
    
    # Get image info
    print(f"  Original shape: {img.shape}, dtype: {img.dtype}")
    
    # For 3D images, take a slice or use maximum projection
    if len(img.shape) == 3:
        if use_max_projection:
            img = np.max(img, axis=0)
            print(f"  Using maximum projection, new shape: {img.shape}")
        else:
            # Use specified slice or middle slice
            if slice_idx is None:
                slice_idx = img.shape[0] // 2
            img = img[slice_idx]
            print(f"  Using slice {slice_idx}, new shape: {img.shape}")
    
    # Rescale to 0-1 float
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    
    # Downscale if needed
    if downscale_factor < 1.0:
        new_shape = (int(img.shape[0] * downscale_factor), int(img.shape[1] * downscale_factor))
        img = transform.resize(img, new_shape, anti_aliasing=True, preserve_range=True)
        print(f"  Downscaled to: {img.shape}")
    
    # Enhance contrast
    img = exposure.equalize_adapthist(img)
    
    return img

def register_brain_images(images, settings):
    """Register brain images using pirt's DiffeomorphicDemonsRegistration
    
    Args:
        images (dict): Dictionary mapping animal IDs to preprocessed images
        settings (dict): Registration settings
        
    Returns:
        tuple: (registration object, deformation fields, transformed images)
    """
    # Convert dictionary to list in a consistent order
    animal_ids = list(images.keys())
    image_list = [images[animal_id] for animal_id in animal_ids]
    
    # Create registration object
    print(f"Registering {len(image_list)} images...")
    reg = pirt.DiffeomorphicDemonsRegistration(*image_list)
    
    # Set registration parameters
    reg.params.grid_sampling_factor = settings['grid_sampling_factor']
    reg.params.scale_sampling = settings['scale_sampling']
    reg.params.speed_factor = settings['speed_factor']
    
    # Perform registration
    reg.register(verbose=1)
    
    # Get deformation fields and transformed images
    deforms = []
    transformed_images = {}
    
    for i, animal_id in enumerate(animal_ids):
        # Get deformation field
        deform = reg.get_deform(i)
        deforms.append(deform)
        
        # Transform image
        transformed = deform.apply_deformation(image_list[i])
        transformed_images[animal_id] = transformed
    
    return reg, deforms, transformed_images

def calculate_average_template(transformed_images):
    """Calculate average brain template from registered images
    
    Args:
        transformed_images (dict): Dictionary mapping animal IDs to transformed images
        
    Returns:
        np.ndarray: Average brain template
    """
    # Stack all transformed images
    image_stack = np.stack(list(transformed_images.values()))
    
    # Calculate average
    template = np.mean(image_stack, axis=0)
    
    return template

def save_registration_results(transformed_images, template, deforms, animal_ids, output_dir, save_deformation=False):
    """Save registration results
    
    Args:
        transformed_images (dict): Dictionary mapping animal IDs to transformed images
        template (np.ndarray): Average brain template
        deforms (list): List of deformation field objects
        animal_ids (list): List of animal IDs
        output_dir (Path): Path to the output directory
        save_deformation (bool): Whether to save deformation fields
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
    
    # Save deformation fields if requested
    if save_deformation:
        deform_dir = output_dir / 'deformation_fields'
        deform_dir.mkdir(exist_ok=True)
        
        for i, animal_id in enumerate(animal_ids):
            # Save deformation field as numpy array
            deform_x = deforms[i].get_field(0)
            deform_y = deforms[i].get_field(1)
            
            np.save(deform_dir / f"{animal_id}_deform_x.npy", deform_x)
            np.save(deform_dir / f"{animal_id}_deform_y.npy", deform_y)
            
            print(f"Saved deformation field for {animal_id}")

def create_visualization(original_images, transformed_images, template, output_dir):
    """Create and save visualization of registration results
    
    Args:
        original_images (dict): Dictionary mapping animal IDs to original images
        transformed_images (dict): Dictionary mapping animal IDs to transformed images
        template (np.ndarray): Average brain template
        output_dir (Path): Path to the output directory
    """
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Create figure for all original images
    num_images = len(original_images)
    animal_ids = list(original_images.keys())
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    # Original images
    plt.figure(figsize=(15, 5 * rows))
    for i, animal_id in enumerate(animal_ids):
        plt.subplot(rows, cols, i+1)
        plt.imshow(original_images[animal_id], cmap='gray')
        plt.title(f"{animal_id} - Original")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(viz_dir / 'original_images.png', dpi=150)
    plt.close()
    
    # Registered images
    plt.figure(figsize=(15, 5 * rows))
    for i, animal_id in enumerate(animal_ids):
        plt.subplot(rows, cols, i+1)
        plt.imshow(transformed_images[animal_id], cmap='gray')
        plt.title(f"{animal_id} - Registered")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(viz_dir / 'registered_images.png', dpi=150)
    plt.close()
    
    # Difference images
    plt.figure(figsize=(15, 5 * rows))
    for i, animal_id in enumerate(animal_ids):
        plt.subplot(rows, cols, i+1)
        diff = np.abs(original_images[animal_id] - transformed_images[animal_id])
        plt.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        plt.title(f"{animal_id} - Difference")
        plt.axis('off')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(viz_dir / 'difference_images.png', dpi=150)
    plt.close()
    
    # Template
    plt.figure(figsize=(10, 8))
    plt.imshow(template, cmap='gray')
    plt.title("Average Brain Template")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(viz_dir / 'template.png', dpi=150)
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
    
    # Load and preprocess images
    loaded_images = {}
    for animal_id, file_path in tqdm(brain_images.items(), desc="Loading images"):
        loaded_images[animal_id] = load_and_preprocess_image(
            file_path, 
            args.downscale, 
            args.slice_idx, 
            args.use_max_projection
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
        output_dir, 
        args.save_deformation
    )
    
    # Create visualization if requested
    if args.visualize:
        create_visualization(loaded_images, transformed_images, template, output_dir)
    
    print("Registration completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())