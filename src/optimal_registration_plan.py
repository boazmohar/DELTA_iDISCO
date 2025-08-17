#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimal Registration Plan for Brain Images

This script implements a workflow to:
1. Identify the best parameter set from a 3D registration parameter scan
2. Apply the optimal registration transforms to full-size channel images
3. Create a template from the optimally registered images
4. Register the template to the CCF atlas
5. Extract region statistics using the CCF alignment

Uses existing functionality from the src package where possible.

Author: Claude
"""

import os
import sys
import json
import numpy as np
import itk
import tifffile
from datetime import datetime
from pathlib import Path
import pandas as pd
import argparse
from scipy import ndimage
import multiprocessing
from functools import partial
import time

# Add the src directory to the Python path
sys.path.append(os.path.abspath('..'))
from src.DELAT_utils import setup_logging, match_h5_files_by_channels, read_h5_image
from src.registration import register_and_transform
from src.stats import compute_region_stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Apply optimal registration from parameter scan')
    
    parser.add_argument('--param_scan_dir', type=str, required=True,
                        help='Path to the parameter scan results directory')
    parser.add_argument('--param_scan_results', type=str, required=True,
                        help='Path to the parameter scan results JSON file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the directory containing original animal data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to store optimized registration results')
    parser.add_argument('--ccf_atlas', type=str, required=True,
                        help='Path to the CCF atlas reference image')
    parser.add_argument('--annotation_path', type=str, required=True,
                        help='Path to the CCF annotation volume')
    parser.add_argument('--param_files_dir', type=str, default='/nrs/spruston/Boaz/I2/itk',
                        help='Directory containing the ITK parameter files')
    
    return parser.parse_args()


def get_best_parameters(param_scan_results_path):
    """
    Extract the best parameter set from parameter scan results
    
    Args:
        param_scan_results_path (str): Path to the parameter scan results JSON
        
    Returns:
        tuple: Best parameter set and downscale factor
    """
    logger = setup_logging(os.path.dirname(param_scan_results_path), 'optimal_registration')
    logger.info(f"Loading parameter scan results from {param_scan_results_path}")
    
    with open(param_scan_results_path, 'r') as f:
        scan_results = json.load(f)
    
    # Find best parameter set based on mean cross-correlation
    best_result = max(scan_results['results'], 
                     key=lambda x: x['metrics']['mean_cc'] if 'metrics' in x else -1)
    
    # Extract downscale factor used in the scan
    downscale_factor = scan_results.get('downscale', 0.1)
    
    logger.info(f"Best parameters found: {best_result['parameters']}")
    logger.info(f"Metrics: mean_cc={best_result['metrics']['mean_cc']:.4f}, mean_mse={best_result['metrics']['mean_mse']:.4f}")
    logger.info(f"Downscale factor used in scan: {downscale_factor}")
    
    return best_result, downscale_factor, logger


def load_channel_tif_images(data_dir, animal_ids, logger):
    """
    Load all channel TIF files for the specified animals
    
    Args:
        data_dir (str): Base directory containing animal data
        animal_ids (list): List of animal IDs to process
        logger: Logger object
        
    Returns:
        dict: Dictionary mapping animal IDs to channel images
    """
    channel_images = {}
    for animal_id in animal_ids:
        animal_data = {}
        animal_dir = os.path.join(data_dir, animal_id, 'itk')
        
        if not os.path.exists(animal_dir):
            logger.warning(f"Directory not found for {animal_id}: {animal_dir}")
            continue
            
        for channel in ['ch0', 'ch1', 'ch2']:
            tif_path = os.path.join(animal_dir, f'{channel}.tif')
            if os.path.exists(tif_path):
                animal_data[channel] = tifffile.imread(tif_path)
                logger.info(f"Loaded {tif_path}, shape: {animal_data[channel].shape}")
            else:
                logger.warning(f"File not found: {tif_path}")
        
        if animal_data:
            channel_images[animal_id] = animal_data
    
    return channel_images


def resize_field_for_shape(field_data, target_shape, field_path, output_path, logger):
    """
    Resize a deformation field to match the target shape and save the result
    
    Args:
        field_data (ndarray): Deformation field data
        target_shape (tuple): Target shape to resize to (Allen CCF size)
        field_path (str): Original field path for logging
        output_path (str): Path to save the resized field
        logger: Logger object
        
    Returns:
        str: Path to the resized field
    """
    # Calculate zoom factors for each dimension
    zoom_factors = [target_shape[i] / field_data.shape[i] for i in range(len(field_data.shape))]
    
    # Use scipy's zoom function for numpy arrays
    resized_field = ndimage.zoom(field_data, zoom_factors, order=3)
    
    # Save the resized field
    np.save(output_path, resized_field)
    logger.info(f"Resized field {field_path} from {field_data.shape} to {resized_field.shape} and saved to {output_path}")
    
    return output_path


def resize_fields_parallel(deform_fields_dir, animal_id, ccf_shape, output_dir, logger):
    """
    Resize all deformation fields for an animal in parallel to match Allen CCF shape
    
    Args:
        deform_fields_dir (str): Directory containing original deformation fields
        animal_id (str): Animal ID
        ccf_shape (tuple): Allen CCF shape for the resized fields
        output_dir (str): Directory to save resized fields
        logger: Logger object
        
    Returns:
        list: Paths to the resized fields
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all deformation field paths
    deform_paths = []
    field_data = []
    output_paths = []
    
    # First, load all the field data and prepare output paths
    for dim in range(3):  # 3D images have 3 dimensions
        field_path = os.path.join(deform_fields_dir, f"{animal_id}_deform_dim{dim}.npy")
        if not os.path.exists(field_path):
            logger.error(f"Deformation field not found: {field_path}")
            return None
        
        # Generate output path with size info
        size_str = 'x'.join([str(s) for s in ccf_shape])
        output_path = os.path.join(output_dir, f"{animal_id}_deform_dim{dim}_size{size_str}.npy")
        
        # Check if resized field already exists (caching)
        if os.path.exists(output_path):
            logger.info(f"Using cached resized field: {output_path}")
            output_paths.append(output_path)
            continue
        
        # Load field data
        data = np.load(field_path)
        field_data.append(data)
        deform_paths.append(field_path)
        output_paths.append(output_path)
    
    # If we have any fields to resize
    if field_data:
        # Create partial function with fixed arguments
        resize_func = partial(resize_field_for_shape, target_shape=ccf_shape, logger=logger)
        
        # Set up arguments for each field
        args = [
            (field, path, out_path) 
            for field, path, out_path in zip(field_data, deform_paths, output_paths[-len(field_data):])  # Only process fields that need resizing
        ]
        
        # Create process pool and run in parallel
        with multiprocessing.Pool(processes=min(len(args), os.cpu_count())) as pool:
            # Use starmap to pass multiple arguments
            pool.starmap(resize_func, args)
    
    return output_paths


def apply_transform_to_channel_images(channel_images, deform_fields_dir, animal_id, logger):
    """
    Apply deformation fields to all channel images using pirt
    
    Args:
        channel_images (dict): Dictionary of channel images for an animal
        deform_fields_dir (str): Directory containing deformation fields
        animal_id (str): Animal ID being processed
        logger: Logger object
        
    Returns:
        dict: Dictionary of transformed channel images
    """
    # Import pirt here to avoid import errors if not used
    try:
        import pirt
        from pirt.deform import DeformationFieldBackward
    except ImportError:
        logger.error("Error: pirt library not found. Please install it first.")
        return None
    
    # The target shape should be the Allen CCF shape - (800, 1320, 658)
    # This is the standard size we're aiming for, but verify from the first channel image
    sample_image = next(iter(channel_images.values()))
    ccf_shape = sample_image.shape
    logger.info(f"Using Allen CCF shape: {ccf_shape} for deformation field resizing")
    
    # Create a subdirectory for resized fields
    resized_fields_dir = os.path.join(os.path.dirname(deform_fields_dir), 'resized_fields')
    
    # Resize fields in parallel and get paths to resized fields
    start_time = time.time()
    resized_field_paths = resize_fields_parallel(
        deform_fields_dir, animal_id, ccf_shape, resized_fields_dir, logger)
    
    if not resized_field_paths:
        logger.error(f"Failed to resize fields for {animal_id} to CCF shape {ccf_shape}")
        return None
    
    # Load resized fields
    resized_fields = [np.load(path) for path in resized_field_paths]
    logger.info(f"Resized deformation fields in {time.time() - start_time:.2f} seconds")
    
    # Create deformation object with resized fields
    deform = DeformationFieldBackward(resized_fields)
    
    # Apply transformations to all channels using the same deformation object
    transformed_channels = {}
    for channel, image in channel_images.items():
        logger.info(f"Applying deformation to {channel} image")
        transformed_channels[channel] = deform.apply_deformation(image)
        logger.info(f"Transformed {channel} to shape {transformed_channels[channel].shape}")
    
    return transformed_channels


def create_ch0_template(transformed_channel_images, logger):
    """
    Create average template for ch0 from transformed images
    
    Args:
        transformed_channel_images (dict): Dictionary mapping animals to transformed channels
        logger: Logger object
        
    Returns:
        ndarray: The ch0 template
    """
    # We only need ch0 for registration to CCF
    channel = 'ch0'
    channel_images = []
    
    for animal_id, channels in transformed_channel_images.items():
        if channel in channels:
            channel_images.append(channels[channel])
    
    if not channel_images:
        logger.error(f"No {channel} images found to create template")
        return None
        
    # Stack and average
    image_stack = np.stack(channel_images)
    ch0_template = np.mean(image_stack, axis=0)
    logger.info(f"Created template for {channel}, shape: {ch0_template.shape}")
    
    return ch0_template


def register_template_to_ccf(template, ccf_atlas, param_files, output_dir, logger):
    """
    Register the ch0 template to the CCF atlas using existing registration code
    
    Args:
        template (ndarray): The ch0 template image
        ccf_atlas (str): Path to the CCF atlas reference image
        param_files (list): List of parameter files for registration
        output_dir (str): Directory to save registration results
        logger: Logger object
        
    Returns:
        tuple: (output_dir path, files dictionary)
    """
    # Create output directory for CCF registration
    ccf_reg_dir = os.path.join(output_dir, 'ccf_registration')
    os.makedirs(ccf_reg_dir, exist_ok=True)
    
    # Save template as TIFF for registration
    template_path = os.path.join(ccf_reg_dir, 'ch0_template.tif')
    tifffile.imwrite(template_path, template)
    
    # Load CCF atlas as fixed image
    logger.info(f"Loading CCF atlas from {ccf_atlas}")
    fx = itk.imread(ccf_atlas, pixel_type=itk.US)
    
    # Create files dictionary in the format expected by register_and_transform
    files = {
        'ch0': template_path,
    }
    
    # Use the existing registration function
    logger.info("Registering template to CCF atlas")
    register_and_transform(fx, files, ccf_reg_dir, param_files, logger)
    
    logger.info(f"Template registered to CCF atlas. Results saved to {ccf_reg_dir}")
    
    return ccf_reg_dir, files


def extract_region_statistics(ccf_aligned_dir, animal_ids, annotation_path, output_dir, logger):
    """
    Extract region statistics from CCF-aligned images using existing stats code
    
    Args:
        ccf_aligned_dir (str): Directory containing CCF-aligned images
        animal_ids (list): List of animal IDs
        annotation_path (str): Path to the CCF annotation volume
        output_dir (str): Directory to save statistics
        logger: Logger object
        
    Returns:
        dict: Dictionary of region statistics DataFrames
    """
    # Load annotation volume
    logger.info(f"Loading annotation volume from {annotation_path}")
    itk_annotation = itk.imread(annotation_path, itk.ULL)
    annotation_np = itk.array_view_from_image(itk_annotation)
    
    # Define statistical functions
    funcs = [np.mean, np.median, np.std, np.max]
    
    # Create directory for statistics
    stats_dir = os.path.join(output_dir, 'region_stats')
    os.makedirs(stats_dir, exist_ok=True)
    
    # Process each animal
    all_stats = {}
    for animal_id in animal_ids:
        animal_dir = os.path.join(ccf_aligned_dir, animal_id)
        if not os.path.exists(animal_dir):
            logger.warning(f"CCF-aligned directory not found for {animal_id}")
            continue
            
        logger.info(f"Extracting region statistics for {animal_id}")
        
        # Prepare channel files dict in the format expected by compute_region_stats
        channel_files = {}
        for channel in ['ch0', 'ch1', 'ch2']:
            channel_path = os.path.join(animal_dir, f"{channel}_ccf_aligned.tif")
            if os.path.exists(channel_path):
                channel_files[channel] = channel_path
        
        if not channel_files:
            logger.warning(f"No channel files found for {animal_id}")
            continue
            
        # Compute region statistics
        num_cores = os.cpu_count()
        animal_stats_dir = os.path.join(stats_dir, animal_id)
        os.makedirs(animal_stats_dir, exist_ok=True)
        
        df_stats = compute_region_stats(channel_files, animal_stats_dir, annotation_np, funcs, num_cores)
        
        # Save to CSV
        stats_path = os.path.join(stats_dir, f"{animal_id}_region_stats.csv")
        df_stats.to_csv(stats_path, index=False)
        
        all_stats[animal_id] = df_stats
        logger.info(f"Region statistics saved to {stats_path}")
    
    # Create combined statistics
    if all_stats:
        combined_stats = pd.concat(all_stats.values(), keys=all_stats.keys())
        combined_path = os.path.join(stats_dir, "all_animals_region_stats.csv")
        combined_stats.to_csv(combined_path)
        logger.info(f"Combined statistics saved to {combined_path}")
    
    return all_stats


def main():
    """Main function to execute the optimal registration workflow"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Get best parameters from scan results
    best_param_set, downscale_factor, logger = get_best_parameters(args.param_scan_results)
    
    # Construct path to best parameter set's deformation fields
    # Build the parameter string according to the format in register_brains_3d_parallel.py
    params = best_param_set['parameters']
    
    # Check if all the new parameters exist in the results
    if all(key in params for key in ['final_scale', 'scale_levels', 'final_grid_sampling', 'noise_factor']):
        # New format with all parameters
        param_str = f"grid{params['grid_sampling_factor']}_scale{params['scale_sampling']}_speed{params['speed_factor']}_finalscale{params['final_scale']}_levels{params['scale_levels']}_finalgrid{params['final_grid_sampling']}_noise{params['noise_factor']}"
    else:
        # Legacy format with just the basic parameters
        param_str = f"grid{params['grid_sampling_factor']}_scale{params['scale_sampling']}_speed{params['speed_factor']}"
    
    logger.info(f"Looking for parameter directory: {param_str}")
    param_dir = os.path.join(args.param_scan_dir, param_str)
    deform_fields_dir = os.path.join(param_dir, 'deformation_fields')
    
    # Get animal IDs from deformation field directory
    animal_ids = []
    if os.path.exists(deform_fields_dir):
        file_pattern = "_deform_dim0.npy"
        for filename in os.listdir(deform_fields_dir):
            if filename.endswith(file_pattern):
                animal_id = filename.replace(file_pattern, "")
                animal_ids.append(animal_id)
    
    if not animal_ids:
        logger.error("No animal deformation fields found")
        return 1
    
    logger.info(f"Found {len(animal_ids)} animals with deformation fields: {animal_ids}")
    
    # 2. Load all channel TIF images
    channel_images = load_channel_tif_images(args.data_dir, animal_ids, logger)
    
    # 3. Apply transforms to all channel images
    transformed_channel_images = {}
    transformed_dir = os.path.join(args.output_dir, 'transformed_images')
    os.makedirs(transformed_dir, exist_ok=True)
    
    for animal_id, images in channel_images.items():
        logger.info(f"Processing animal: {animal_id}")
        transformed_channels = apply_transform_to_channel_images(
            images, deform_fields_dir, animal_id, logger)
        
        if transformed_channels:
            transformed_channel_images[animal_id] = transformed_channels
            
            # Save transformed channel images
            animal_dir = os.path.join(transformed_dir, animal_id)
            os.makedirs(animal_dir, exist_ok=True)
            
            for channel, transformed_image in transformed_channels.items():
                output_path = os.path.join(animal_dir, f"{channel}_transformed.tif")
                
                # Save with metadata
                tifffile.imwrite(output_path, transformed_image, metadata={
                    'downscale_factor': downscale_factor,
                    'original_shape': str(images[channel].shape),
                    'transform_params': str(best_param_set['parameters']),
                    'processing_date': datetime.now().isoformat()
                })
                
                logger.info(f"Saved transformed {channel} to {output_path}")
    
    # 4. Create ch0 template
    ch0_template = create_ch0_template(transformed_channel_images, logger)
    if ch0_template is None:
        logger.error("Error: Failed to create ch0 template")
        return 1
        
    # Save the ch0 template
    templates_dir = os.path.join(args.output_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    template_path = os.path.join(templates_dir, "ch0_template.tif")
    tifffile.imwrite(template_path, ch0_template)
    logger.info(f"Saved ch0 template to {template_path}")
    
    # 5. Register ch0 template to CCF atlas
    
    # Load ITK parameter files
    param_files = [
        os.path.join(args.param_files_dir, 'Order1_Par0000affine.txt'),
        os.path.join(args.param_files_dir, 'Order3_Par0000bspline.txt'),
        os.path.join(args.param_files_dir, 'Order4_Par0000bspline.txt'),
        os.path.join(args.param_files_dir, 'Order5_Par0000bspline.txt')
    ]
    
    ccf_reg_dir, template_files = register_template_to_ccf(
        ch0_template, args.ccf_atlas, param_files, args.output_dir, logger)
    
    # We only need to register the ch0 template to CCF
    # No need to register other channel templates
    
    # 7. Apply same transform to all animal channel images
    ccf_aligned_dir = os.path.join(args.output_dir, 'ccf_aligned_images')
    os.makedirs(ccf_aligned_dir, exist_ok=True)
    
    # Create transformix parameter object for reuse
    transformix_params = itk.ParameterObject.New()
    for i in range(4):  # There are typically 4 transform parameters files
        param_path = os.path.join(ccf_reg_dir, f'TransformParameters.{i}.txt')
        if os.path.exists(param_path):
            transformix_params.AddParameterFile(param_path)
            logger.info(f"Added transform parameter file: {param_path}")
    
    # Apply transform to each animal's channel images
    for animal_id, channels in transformed_channel_images.items():
        animal_dir = os.path.join(ccf_aligned_dir, animal_id)
        os.makedirs(animal_dir, exist_ok=True)
        
        for channel, image in channels.items():
            logger.info(f"Applying CCF transform to {animal_id} {channel}")
            
            # Save image to temporary file
            temp_path = os.path.join(ccf_reg_dir, f"temp_{animal_id}_{channel}.tif")
            tifffile.imwrite(temp_path, image)
            
            # Read image with ITK
            itk_image = itk.imread(temp_path, pixel_type=itk.F)
            
            # Apply transform
            transformix_filter = itk.TransformixFilter.New(
                Input=itk_image, 
                TransformParameterObject=transformix_params
            )
            transformix_filter.SetComputeSpatialJacobian(False)
            transformix_filter.SetComputeDeterminantOfSpatialJacobian(False)
            transformix_filter.SetComputeDeformationField(False)
            transformix_filter.Update()
            
            # Get and save result
            result = transformix_filter.GetOutput()
            output_path = os.path.join(animal_dir, f"{channel}_ccf_aligned.tif")
            itk.imwrite(result, output_path)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            logger.info(f"Saved CCF-aligned {channel} to {output_path}")
    
    # 8. Extract region statistics
    all_stats = extract_region_statistics(
        ccf_aligned_dir, animal_ids, args.annotation_path, args.output_dir, logger)
    
    logger.info(f"Optimal registration process completed. Results saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    main()