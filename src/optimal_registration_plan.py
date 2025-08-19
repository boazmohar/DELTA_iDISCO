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
from skimage import transform
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
    parser.add_argument('--param_files_dir', type=str, default='/nearline/spruston/Boaz/DELTA/I2/itk',
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
    downscale_factor = scan_results.get('downscale', 0.2)
    
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
        # Create partial function with fixed arguments - don't include target_shape here 
        # since it will be passed as an argument
        resize_func = partial(resize_field_for_shape, logger=logger)
        
        # Set up arguments for each field, including target_shape
        args = [
            (field, ccf_shape, path, out_path) 
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
            # Convert to numpy array to avoid pirt sampling issues
            image = channels[channel]
            if not isinstance(image, np.ndarray):
                try:
                    image = np.asarray(image)
                    logger.info(f"Converted {animal_id} {channel} image to numpy array, shape: {image.shape}")
                except Exception as e:
                    logger.error(f"Error converting {animal_id} {channel} image to numpy array: {str(e)}")
                    continue
            channel_images.append(image)
    
    if not channel_images:
        logger.error(f"No {channel} images found to create template")
        return None
    
    # Log details about each image before stacking
    for i, img in enumerate(channel_images):
        logger.info(f"Image {i} type: {type(img)}, shape: {img.shape}, dtype: {img.dtype}")
    
    try:
        # Stack and average
        image_stack = np.stack(channel_images)
        ch0_template = np.mean(image_stack, axis=0)
        logger.info(f"Created template for {channel}, shape: {ch0_template.shape}")
        return ch0_template
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        
        # Try alternative approach
        logger.info("Trying alternative approach with array conversion")
        try:
            # Try to convert all images to arrays with the same shape
            first_shape = channel_images[0].shape
            converted_images = []
            
            for img in channel_images:
                if img.shape != first_shape:
                    logger.warning(f"Image shape mismatch: expected {first_shape}, got {img.shape}")
                    # Resize to match the first image
                    img = transform.resize(img, first_shape, anti_aliasing=True, preserve_range=True)
                converted_images.append(np.array(img, dtype=np.float32))
            
            # Stack and average
            image_stack = np.stack(converted_images)
            ch0_template = np.mean(image_stack, axis=0)
            logger.info(f"Created template (alternative method) for {channel}, shape: {ch0_template.shape}")
            return ch0_template
        except Exception as e2:
            logger.error(f"Alternative approach also failed: {str(e2)}")
            return None


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
    
    # Instead of using register_and_transform which expects HDF5 files,
    # we'll implement a similar workflow directly for TIFF files
    logger.info("Registering template to CCF atlas")
    
    # Load the template as an ITK image
    try:
        # Load the template with ITK instead of using read_h5_image
        logger.info(f"Loading template from {template_path}")
        mv = itk.imread(template_path, pixel_type=itk.F)
        
        # Create a new ParameterObject for registration
        parameter_object = itk.ParameterObject.New()
        for p in param_files:
            if os.path.exists(p):
                parameter_object.AddParameterFile(p)
                logger.info(f"Added parameter file: {p}")
            else:
                logger.warning(f"Parameter file not found: {p}")
        
        # Perform registration
        logger.info("Starting elastix registration")
        res, params = itk.elastix_registration_method(
            fx, mv, parameter_object, 
            log_to_file=True, 
            output_directory=ccf_reg_dir
        )
        logger.info(f"Registration completed, result saved to {ccf_reg_dir}")
        
        # Also save the transformed template
        output_image_path = os.path.join(ccf_reg_dir, 'ch0_template_ccf_aligned.tif')
        itk.imwrite(res, output_image_path)
        logger.info(f"Transformed template saved to {output_image_path}")
        
        # Create dictionary of output files
        files = {
            'ch0': template_path,
            'ch0_ccf_aligned': output_image_path
        }
        
        return ccf_reg_dir, files
    except Exception as e:
        logger.error(f"Error in CCF registration: {str(e)}")
        # Return what we have so far
        return ccf_reg_dir, {'ch0': template_path}


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
        
        # Create a directory for this animal's stats
        animal_stats_dir = os.path.join(stats_dir, animal_id)
        os.makedirs(animal_stats_dir, exist_ok=True)
        
        # First copy or link the aligned files to the animal_stats_dir with simple names
        # that will work with the compute_region_stats function
        try:
            # Prepare channel files dict with simplified names
            channel_files = {}
            for channel in ['ch0', 'ch1', 'ch2']:
                source_path = os.path.join(animal_dir, f"{channel}_ccf_aligned.tif")
                if os.path.exists(source_path):
                    # Create a local copy or link in animal_stats_dir
                    local_path = os.path.join(animal_stats_dir, f"{channel}.tif")
                    if not os.path.exists(local_path):
                        try:
                            # Try to create a symbolic link first
                            os.symlink(source_path, local_path)
                            logger.info(f"Created symlink from {source_path} to {local_path}")
                        except Exception as e:
                            logger.warning(f"Error creating symlink: {str(e)}, will try copying")
                            # If symlink fails, copy the file
                            import shutil
                            shutil.copy2(source_path, local_path)
                            logger.info(f"Copied file from {source_path} to {local_path}")
                    
                    # Add to channel_files dict with just the channel name (compute_region_stats will add .tif)
                    channel_files[channel] = channel
                    logger.info(f"Added channel {channel} for {animal_id}")
            
            if not channel_files:
                logger.warning(f"No channel files found for {animal_id}")
                continue
                
            # Compute region statistics
            num_cores = os.cpu_count()
            logger.info(f"Starting compute_region_stats for {animal_id} with {num_cores} cores")
            
            try:
                df_stats = compute_region_stats(channel_files, animal_stats_dir, annotation_np, funcs, num_cores)
                
                # Save to CSV
                stats_path = os.path.join(stats_dir, f"{animal_id}_region_stats.csv")
                df_stats.to_csv(stats_path, index=False)
                
                all_stats[animal_id] = df_stats
                logger.info(f"Region statistics saved to {stats_path}")
            except Exception as e:
                logger.error(f"Error computing region statistics for {animal_id}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error preparing files for {animal_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Create combined statistics if we have any results
    if all_stats:
        try:
            combined_stats = pd.concat(all_stats.values(), keys=all_stats.keys())
            combined_path = os.path.join(stats_dir, "all_animals_region_stats.csv")
            combined_stats.to_csv(combined_path)
            logger.info(f"Combined statistics saved to {combined_path}")
        except Exception as e:
            logger.error(f"Error creating combined statistics: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return all_stats


def check_stage_completion(output_dir, stage_name, logger):
    """
    Check if a processing stage has been completed
    
    Args:
        output_dir (str): Base output directory
        stage_name (str): Name of the stage to check
        logger: Logger object
        
    Returns:
        tuple: (is_completed, data)
            is_completed (bool): Whether the stage is completed
            data: Stage-specific data if available
    """
    if stage_name == "transformed_images":
        # Check if transformed images directory exists and has content
        transformed_dir = os.path.join(output_dir, 'transformed_images')
        if os.path.exists(transformed_dir):
            # Check if there are animal subdirectories with transformed files
            animal_dirs = [d for d in os.listdir(transformed_dir) if os.path.isdir(os.path.join(transformed_dir, d))]
            if animal_dirs:
                logger.info(f"Found {len(animal_dirs)} animals with transformed images")
                
                # Load transformed images
                transformed_channel_images = {}
                for animal_id in animal_dirs:
                    animal_dir = os.path.join(transformed_dir, animal_id)
                    animal_channels = {}
                    for channel in ['ch0', 'ch1', 'ch2']:
                        channel_path = os.path.join(animal_dir, f"{channel}_transformed.tif")
                        if os.path.exists(channel_path):
                            try:
                                animal_channels[channel] = tifffile.imread(channel_path)
                                logger.info(f"Loaded transformed {channel} for {animal_id}")
                            except Exception as e:
                                logger.warning(f"Error loading transformed {channel} for {animal_id}: {str(e)}")
                    
                    if animal_channels:
                        transformed_channel_images[animal_id] = animal_channels
                
                if transformed_channel_images:
                    return True, transformed_channel_images
        
        return False, None
    
    elif stage_name == "ch0_template":
        # Check if ch0 template exists
        templates_dir = os.path.join(output_dir, 'templates')
        template_path = os.path.join(templates_dir, "ch0_template.tif")
        if os.path.exists(template_path):
            try:
                ch0_template = tifffile.imread(template_path)
                logger.info(f"Found existing ch0 template: {template_path}")
                return True, ch0_template
            except Exception as e:
                logger.warning(f"Error loading ch0 template: {str(e)}")
        
        return False, None
    
    elif stage_name == "ccf_registration":
        # Check if CCF registration directory exists and has transform parameters
        ccf_reg_dir = os.path.join(output_dir, 'ccf_registration')
        if os.path.exists(ccf_reg_dir):
            # Check if there are transform parameters files
            param_files = [f for f in os.listdir(ccf_reg_dir) if f.startswith("TransformParameters")]
            if param_files:
                logger.info(f"Found {len(param_files)} transform parameter files in {ccf_reg_dir}")
                
                # Create transformix parameter object
                transformix_params = itk.ParameterObject.New()
                has_params = False
                for i in range(4):  # There are typically 4 transform parameters files
                    param_path = os.path.join(ccf_reg_dir, f'TransformParameters.{i}.txt')
                    if os.path.exists(param_path):
                        transformix_params.AddParameterFile(param_path)
                        logger.info(f"Added transform parameter file: {param_path}")
                        has_params = True
                
                if has_params:
                    return True, (ccf_reg_dir, transformix_params)
        
        return False, None
    
    elif stage_name == "ccf_aligned_images":
        # Check if CCF aligned images directory exists and has content
        ccf_aligned_dir = os.path.join(output_dir, 'ccf_aligned_images')
        if os.path.exists(ccf_aligned_dir):
            # Check if there are animal subdirectories with aligned files
            animal_dirs = [d for d in os.listdir(ccf_aligned_dir) if os.path.isdir(os.path.join(ccf_aligned_dir, d))]
            if animal_dirs:
                logger.info(f"Found {len(animal_dirs)} animals with CCF-aligned images")
                
                # Check which animals have been fully processed
                completed_animals = []
                for animal_id in animal_dirs:
                    animal_dir = os.path.join(ccf_aligned_dir, animal_id)
                    # Check if all channels are present
                    all_channels_present = True
                    for channel in ['ch0', 'ch1', 'ch2']:
                        channel_path = os.path.join(animal_dir, f"{channel}_ccf_aligned.tif")
                        if not os.path.exists(channel_path):
                            all_channels_present = False
                            break
                    
                    if all_channels_present:
                        completed_animals.append(animal_id)
                
                if completed_animals:
                    logger.info(f"Found {len(completed_animals)} animals with complete CCF alignment")
                    return True, completed_animals
        
        return False, None
    
    elif stage_name == "region_stats":
        # Check if region stats directory exists and has content
        stats_dir = os.path.join(output_dir, 'region_stats')
        if os.path.exists(stats_dir):
            # Check if there are individual animal stats files
            stats_files = [f for f in os.listdir(stats_dir) if f.endswith('_region_stats.csv')]
            if stats_files:
                # Extract animal IDs from filenames
                completed_animals = [f.split('_region_stats.csv')[0] for f in stats_files]
                logger.info(f"Found region statistics for {len(completed_animals)} animals")
                
                # Check if combined stats file exists
                combined_path = os.path.join(stats_dir, "all_animals_region_stats.csv")
                has_combined = os.path.exists(combined_path)
                
                return True, (completed_animals, has_combined)
        
        return False, None
    
    # Unknown stage
    logger.warning(f"Unknown stage: {stage_name}")
    return False, None

def main():
    """Main function to execute the optimal registration workflow"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up a status file to track progress
    status_file = os.path.join(args.output_dir, 'pipeline_status.json')
    
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
    
    # Initialize variables for pipeline stages
    transformed_channel_images = None
    ch0_template = None
    ccf_reg_data = None
    
    # 2-3. Check if transformed images already exist
    transformed_complete, transformed_data = check_stage_completion(args.output_dir, "transformed_images", logger)
    if transformed_complete:
        logger.info("Resuming from existing transformed images")
        transformed_channel_images = transformed_data
    else:
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
    
    if not transformed_channel_images:
        logger.error("No transformed channel images available")
        return 1
    
    # 4. Check if ch0 template already exists
    template_complete, template_data = check_stage_completion(args.output_dir, "ch0_template", logger)
    if template_complete:
        logger.info("Resuming from existing ch0 template")
        ch0_template = template_data
    else:
        # Create ch0 template
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
    
    # 5. Check if CCF registration already exists
    ccf_reg_complete, ccf_reg_data = check_stage_completion(args.output_dir, "ccf_registration", logger)
    if ccf_reg_complete:
        logger.info("Resuming from existing CCF registration")
        ccf_reg_dir, transformix_params = ccf_reg_data
    else:
        # Register ch0 template to CCF atlas
        # Load ITK parameter files
        param_files = [
            os.path.join(args.param_files_dir, 'Order1_Par0000affine.txt'),
            os.path.join(args.param_files_dir, 'Order3_Par0000bspline.txt'),
            os.path.join(args.param_files_dir, 'Order4_Par0000bspline.txt'),
            os.path.join(args.param_files_dir, 'Order5_Par0000bspline.txt')
        ]
        
        ccf_reg_dir, template_files = register_template_to_ccf(
            ch0_template, args.ccf_atlas, param_files, args.output_dir, logger)
        
        # Create transformix parameter object for reuse
        transformix_params = itk.ParameterObject.New()
        has_params = False
        for i in range(4):  # There are typically 4 transform parameters files
            param_path = os.path.join(ccf_reg_dir, f'TransformParameters.{i}.txt')
            if os.path.exists(param_path):
                transformix_params.AddParameterFile(param_path)
                logger.info(f"Added transform parameter file: {param_path}")
                has_params = True
        
        if not has_params:
            logger.error("No transform parameter files found! Cannot apply CCF alignment to animal images.")
            return 1
    
    # 6. Check if CCF-aligned images already exist
    ccf_aligned_complete, ccf_aligned_data = check_stage_completion(args.output_dir, "ccf_aligned_images", logger)
    completed_animals = ccf_aligned_data if ccf_aligned_complete else []
    
    # 7. Apply CCF transform to remaining animal channel images
    ccf_aligned_dir = os.path.join(args.output_dir, 'ccf_aligned_images')
    os.makedirs(ccf_aligned_dir, exist_ok=True)
    
    # Process animals that haven't been fully CCF-aligned yet
    animals_to_process = [animal_id for animal_id in animal_ids if animal_id not in completed_animals]
    
    if animals_to_process:
        logger.info(f"Applying CCF alignment to {len(animals_to_process)} animals")
        
        for animal_id in animals_to_process:
            if animal_id not in transformed_channel_images:
                logger.warning(f"No transformed images found for {animal_id}, skipping CCF alignment")
                continue
                
            channels = transformed_channel_images[animal_id]
            animal_dir = os.path.join(ccf_aligned_dir, animal_id)
            os.makedirs(animal_dir, exist_ok=True)
            
            for channel, image in channels.items():
                # Check if this channel is already aligned
                output_path = os.path.join(animal_dir, f"{channel}_ccf_aligned.tif")
                if os.path.exists(output_path):
                    logger.info(f"CCF-aligned {channel} already exists for {animal_id}, skipping")
                    continue
                    
                try:
                    logger.info(f"Applying CCF transform to {animal_id} {channel}")
                    
                    # Make sure image is a numpy array
                    if not isinstance(image, np.ndarray):
                        logger.info(f"Converting {animal_id} {channel} to numpy array")
                        image = np.asarray(image)
                    
                    # Save image to temporary file
                    temp_path = os.path.join(ccf_reg_dir, f"temp_{animal_id}_{channel}.tif")
                    tifffile.imwrite(temp_path, image)
                    logger.info(f"Saved temporary file: {temp_path}")
                    
                    # Read image with ITK
                    itk_image = itk.imread(temp_path, pixel_type=itk.F)
                    logger.info(f"Loaded ITK image from {temp_path}")
                    
                    # Apply transform
                    transformix_filter = itk.TransformixFilter.New(
                        Input=itk_image, 
                        TransformParameterObject=transformix_params
                    )
                    transformix_filter.SetComputeSpatialJacobian(False)
                    transformix_filter.SetComputeDeterminantOfSpatialJacobian(False)
                    transformix_filter.SetComputeDeformationField(False)
                    logger.info(f"Starting transformix for {animal_id} {channel}")
                    transformix_filter.Update()
                    logger.info(f"Transformix completed for {animal_id} {channel}")
                    
                    # Get and save result
                    result = transformix_filter.GetOutput()
                    itk.imwrite(result, output_path)
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.warning(f"Error removing temporary file {temp_path}: {str(e)}")
                    
                    logger.info(f"Saved CCF-aligned {channel} to {output_path}")
                except Exception as e:
                    logger.error(f"Error applying CCF transform to {animal_id} {channel}: {str(e)}")
                    continue
    else:
        logger.info("All animals already have CCF-aligned images")
    
    # 8. Check which animals have region statistics
    stats_complete, stats_data = check_stage_completion(args.output_dir, "region_stats", logger)
    completed_stats = stats_data[0] if stats_complete else []
    
    # Filter animals for stats computation
    animals_for_stats = [animal_id for animal_id in animal_ids if animal_id not in completed_stats]
    
    if animals_for_stats:
        logger.info(f"Computing region statistics for {len(animals_for_stats)} animals")
        # Extract region statistics for remaining animals
        all_stats = extract_region_statistics(
            ccf_aligned_dir, animals_for_stats, args.annotation_path, args.output_dir, logger)
    else:
        logger.info("All animals already have region statistics")
    
    # Save pipeline status
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'param_str': param_str,
            'downscale_factor': downscale_factor,
            'stages': {
                'transformed_images': transformed_complete,
                'ch0_template': template_complete,
                'ccf_registration': ccf_reg_complete,
                'ccf_aligned_images': ccf_aligned_complete,
                'region_stats': stats_complete
            },
            'animals': {
                'total': len(animal_ids),
                'ccf_aligned': len(completed_animals),
                'stats_computed': len(completed_stats)
            }
        }
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info(f"Pipeline status saved to {status_file}")
    except Exception as e:
        logger.error(f"Error saving pipeline status: {str(e)}")
    
    logger.info(f"Optimal registration process completed. Results saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    main()