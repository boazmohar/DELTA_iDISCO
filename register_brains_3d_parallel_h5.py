#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Group Brain Registration Script with Parameter Scanning for h5 files

This script performs 3D group registration for brain images from h5 files using 
pirt.DiffeomorphicDemonsRegistration with parameter scanning capabilities running in parallel.

It can process multiple 3D brain images from h5 files from different animals, test various registration
parameters in parallel, and generate common templates for each parameter set.

Usage:
    python register_brains_3d_parallel_h5.py --data_dir /path/to/data --output_dir /path/to/output [options]

Author: Claude, modified for h5 file support
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pirt
import tifffile
import h5py
from pathlib import Path
from skimage import transform, exposure
import warnings
import multiprocessing
import itertools
import time
import json
import logging
from datetime import datetime
from src.DELAT_utils import read_h5_image

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

warnings.filterwarnings('ignore')

# Set up logging
def setup_debug_logging(output_dir):
    """Set up detailed debug logging"""
    log_file = os.path.join(output_dir, 'debug_log.txt')
    
    # Create a logger
    logger = logging.getLogger('registration_debug')
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters and add to handlers
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3D Group Brain Registration for h5 files with Parameter Scanning')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory containing animal folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory for registration results')
    parser.add_argument('--h5_resolution', type=int, default=1, choices=[1, 2, 4],
                        help='Resolution level to use from h5 file (1=Data, 2=Data_2_2_2, 4=Data_4_4_4)')
    parser.add_argument('--threshold', type=int, default=100, 
                        help='Threshold value for h5 image reading (default: 100)')
   
    # Parameter ranges for scanning
    parser.add_argument('--grid_sampling_factors', type=float, nargs='+', default=[0.5, 0.7, 1.0],
                        help='Grid sampling factors to test (default: 0.5 0.7 1.0)')
    parser.add_argument('--scale_samplings', type=int, nargs='+', default=[20, 30, 40],
                        help='Scale sampling iterations to test (default: 20 30 40)')
    parser.add_argument('--speed_factors', type=float, nargs='+', default=[1.0, 2.0, 3.0, 4.0],
                        help='Speed factors to test (default: 1.0 2.0 3.0 4.0)')
    parser.add_argument('--final_scales', type=float, nargs='+', default=[1.0],
                        help='Final scales to test (default: 1.0, min: 0.5)')
    parser.add_argument('--scale_levels', type=int, nargs='+', default=[4, 6],
                        help='Scale levels to test (default: 4 6)')
    parser.add_argument('--final_grid_samplings', type=float, nargs='+', default=[1.0, 2.0],
                        help='Final grid sampling to test (default: 1.0 2.0)')
    parser.add_argument('--noise_factors', type=float, nargs='+', default=[1.0],
                        help='Noise factors to test (default: 1.0, recommended)')
    
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Generate and save visualization of registration results')
    parser.add_argument('--max_processes', type=int, default=0,
                        help='Maximum number of parallel processes (default: number of CPU cores)')
    parser.add_argument('--param_subset', type=int, default=0,
                        help='Only run a subset of parameter combinations (0 = all, otherwise number to run)')
    
    return parser.parse_args()

def find_brain_h5_files(data_dir, h5_pattern="uni_tp-0_ch-0_*.lux.h5"):
    """Find all h5 files in the data directory matching the pattern
    
    Args:
        data_dir (Path): Path to the data directory
        h5_pattern (str): Pattern to match h5 files
        
    Returns:
        dict: Dictionary mapping animal IDs to file paths
    """
    brain_images = {}
    
    # List all animal directories
    for animal_dir in data_dir.glob('ANM*'):
        if animal_dir.is_dir():
            # Look for h5 files matching the pattern
            h5_files = list(animal_dir.glob(h5_pattern))
            if h5_files:
                animal_id = animal_dir.name
                brain_images[animal_id] = h5_files[0]  # Take the first matching file
    
    return brain_images

def load_and_preprocess_h5_image(file_path, threshold=100, h5_resolution=1):
    """Load and preprocess a 3D brain image from h5 file
    
    Args:
        file_path (Path): Path to the h5 file
        dataset_name (str): Name of the dataset in h5 file
        threshold (int): Threshold value for image reading
        downscale_factor (float): Factor to downscale the image
        
    Returns:
        np.ndarray: Preprocessed image (3D volume)
    """
    # Determine dataset name based on resolution
    if h5_resolution == 1:
        dataset_name = 'Data'
    elif h5_resolution == 2:
        dataset_name = 'Data_2_2_2'
    elif h5_resolution == 4:
        dataset_name = 'Data_4_4_4'
    else:
        raise ValueError(f"Invalid h5_resolution: {h5_resolution}. Must be 1, 2, or 4.")
    
    # Load image using read_h5_image
    print(f"Loading {file_path} dataset {dataset_name}...")
    img = read_h5_image(str(file_path), dataset_name, threshold)
    
    # Get image info
    print(f"  Original shape: {img.shape}, dtype: {img.dtype}")
    
    # Ensure we have a 3D image
    if len(img.shape) != 3:
        raise ValueError(f"Expected a 3D image, got shape {img.shape}")

    # Rescale to 0-1 float
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    
    # No need for additional downscaling as we're using the pre-downsampled datasets from the h5 file
    
    return img

def register_brain_images(images, settings, logger):
    """Register 3D brain images using pirt's DiffeomorphicDemonsRegistration
    
    Args:
        images (dict): Dictionary mapping animal IDs to preprocessed 3D images
        settings (dict): Registration settings
        logger: Logger for debugging information
        
    Returns:
        tuple: (registration object, deformation fields, transformed images)
    """
    # Convert dictionary to list in a consistent order
    animal_ids = list(images.keys())
    image_list = [images[animal_id] for animal_id in animal_ids]
    
    # Log detailed information about images
    logger.debug(f"Number of images: {len(image_list)}")
    for i, animal_id in enumerate(animal_ids):
        img = image_list[i]
        logger.debug(f"Image {i} (Animal {animal_id}):\n  Shape: {img.shape}\n  Type: {img.dtype}")
        logger.debug(f"  Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.4f}, Std: {img.std():.4f}")
        # Check for NaN or Inf values
        if np.isnan(img).any() or np.isinf(img).any():
            logger.warning(f"Image {i} (Animal {animal_id}) contains NaN or Inf values!")
        
    # Log all shape information
    shapes = [img.shape for img in image_list]
    logger.debug(f"All image shapes: {shapes}")
    
    # Check if all images have the same shape
    if len(set(shapes)) > 1:
        logger.warning(f"Not all images have the same shape: {shapes}")
        logger.warning("Continuing with registration, but this may affect results. Consider using the same h5_resolution for all images.")
    
    # Create registration object - pirt can handle 3D volumes directly
    print(f"Registering {len(image_list)} 3D images with shapes: {shapes}")
    logger.info(f"Registering {len(image_list)} 3D images with shapes: {shapes}")
    
    # Log registration parameters
    param_info = (f"Parameters:\n"
          f"  grid_sampling_factor={settings['grid_sampling_factor']}\n"
          f"  scale_sampling={settings['scale_sampling']}\n"
          f"  speed_factor={settings['speed_factor']}\n"
          f"  final_scale={settings['final_scale']}\n"
          f"  scale_levels={settings['scale_levels']}\n"
          f"  final_grid_sampling={settings['final_grid_sampling']}\n"
          f"  noise_factor={settings['noise_factor']}")
    
    print(f"Parameters: grid_sampling_factor={settings['grid_sampling_factor']}, "
          f"scale_sampling={settings['scale_sampling']}, speed_factor={settings['speed_factor']}, "
          f"final_scale={settings['final_scale']}, scale_levels={settings['scale_levels']}, "
          f"final_grid_sampling={settings['final_grid_sampling']}, noise_factor={settings['noise_factor']}")
    logger.info(param_info)
    
    try:
        # Log info about pirt version if available
        logger.debug(f"PIRT version: {pirt.__version__ if hasattr(pirt, '__version__') else 'unknown'}")
        
        # Create registration object
        logger.debug("Creating DiffeomorphicDemonsRegistration object")
        reg = pirt.DiffeomorphicDemonsRegistration(*image_list)
        
        # Set registration parameters
        logger.debug("Setting registration parameters")
        reg.params.grid_sampling_factor = settings['grid_sampling_factor']
        reg.params.scale_sampling = settings['scale_sampling']
        reg.params.speed_factor = settings['speed_factor']
        reg.params.final_scale = settings['final_scale']
        reg.params.scale_levels = settings['scale_levels']
        reg.params.final_grid_sampling = settings['final_grid_sampling']
        reg.params.noise_factor = settings['noise_factor']
        
        # Log additional information about registration object if available
        logger.debug(f"Registration params: {reg.params.__dict__ if hasattr(reg.params, '__dict__') else 'Not available'}")
        
        # Perform registration
        print("Starting registration - this may take a while for 3D volumes...")
        logger.info("Starting registration")
        start_time = time.time()
        reg.register(verbose=1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Registration completed in {elapsed_time:.2f} seconds!")
        logger.info(f"Registration completed in {elapsed_time:.2f} seconds!")
    
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        logger.exception("Registration exception details:")
        raise e
    
    # Get deformation fields and transformed images
    deforms = []
    transformed_images = {}
    
    for i, animal_id in enumerate(animal_ids):
        print(f"Processing deformation for {animal_id}...")
        logger.info(f"Processing deformation for {animal_id}...")
        
        try:
            # Get deformation field
            logger.debug(f"Getting deformation field for animal {animal_id} (index {i})")
            deform = reg.get_deform(i)
            deforms.append(deform)
            
            # Transform image
            logger.debug(f"Applying deformation to image {i}")
            transformed = deform.apply_deformation(image_list[i])
            # Convert to numpy array to avoid sampling issues
            transformed_array = np.asarray(transformed)
            logger.debug(f"Transformed image shape: {transformed_array.shape}, min: {transformed_array.min()}, max: {transformed_array.max()}")
            transformed_images[animal_id] = transformed_array
        except Exception as e:
            logger.error(f"Error processing deformation for {animal_id}: {str(e)}")
            logger.exception("Deformation exception details:")
            raise
    
    return reg, deforms, transformed_images, elapsed_time

def calculate_average_template(transformed_images):
    """Calculate average brain template from registered 3D images
    
    Args:
        transformed_images (dict): Dictionary mapping animal IDs to transformed 3D images
        
    Returns:
        np.ndarray: Average brain template (3D volume)
    """
    # Extract the numpy array data from each transformed image if needed
    image_arrays = [np.asarray(img) for img in transformed_images.values()]
    
    # Stack the arrays
    image_stack = np.stack(image_arrays)
    
    # Calculate average
    template = np.mean(image_stack, axis=0)
    
    return template

def calculate_registration_quality(transformed_images):
    """Calculate registration quality metrics
    
    Args:
        transformed_images (dict): Dictionary mapping animal IDs to transformed 3D images
        
    Returns:
        dict: Dictionary of quality metrics
    """
    # Extract the numpy array data from each transformed image if needed
    image_arrays = [np.asarray(img) for img in transformed_images.values()]
    
    # Stack the arrays
    image_stack = np.stack(image_arrays)
    
    # Calculate average
    template = np.mean(image_stack, axis=0)
    
    # Calculate metrics
    metrics = {}
    
    # Mean squared difference between each image and the template
    mse_values = []
    for img in image_arrays:
        mse = float(np.mean((img - template) ** 2))
        mse_values.append(mse)
    
    metrics['mean_mse'] = float(np.mean(mse_values))
    metrics['std_mse'] = float(np.std(mse_values))
    
    # Calculate cross-correlation between each pair of registered images
    cc_values = []
    n_images = len(image_arrays)
    
    for i in range(n_images):
        for j in range(i+1, n_images):
            # Flatten images
            img1_flat = image_arrays[i].flatten()
            img2_flat = image_arrays[j].flatten()
            
            # Calculate correlation coefficient
            cc = float(np.corrcoef(img1_flat, img2_flat)[0, 1])
            cc_values.append(cc)
    
    metrics['mean_cc'] = float(np.mean(cc_values))
    metrics['std_cc'] = float(np.std(cc_values))
    
    return metrics

def save_registration_results(transformed_images, template, deforms, animal_ids, output_dir, 
                             settings, metrics, elapsed_time, save_fields=True):
    """Save registration results
    
    Args:
        transformed_images (dict): Dictionary mapping animal IDs to transformed 3D images
        template (np.ndarray): Average brain template (3D volume)
        deforms (list): List of deformation field objects
        animal_ids (list): List of animal IDs
        output_dir (Path): Path to the output directory
        settings (dict): Registration settings
        metrics (dict): Registration quality metrics
        elapsed_time (float): Registration time in seconds
        save_fields (bool): Whether to save deformation fields
    """
    # Create parameter-specific output directory
    param_str = f"grid{settings['grid_sampling_factor']}_scale{settings['scale_sampling']}_speed{settings['speed_factor']}_finalscale{settings['final_scale']}_levels{settings['scale_levels']}_finalgrid{settings['final_grid_sampling']}_noise{settings['noise_factor']}"
    param_dir = output_dir / param_str
    param_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters and metrics
    results = {
        'parameters': settings,
        'quality_metrics': metrics,
        'elapsed_time': elapsed_time,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(param_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    # Save template
    template_path = param_dir / 'template.tif'
    tifffile.imwrite(template_path, template)
    print(f"Saved template to {template_path}")
    
    # Save transformed images
    for animal_id, img in transformed_images.items():
        img_path = param_dir / f"{animal_id}_registered.tif"
        tifffile.imwrite(img_path, img)
        print(f"Saved {animal_id} to {img_path}")
    
    # Save deformation fields if requested
    if save_fields:
        deform_dir = param_dir / 'deformation_fields'
        deform_dir.mkdir(exist_ok=True)
        
        for i, animal_id in enumerate(animal_ids):
            # For 3D images, save deformation fields for each dimension
            for dim in range(deforms[i].ndim):
                field = deforms[i].get_field(dim)
                np.save(deform_dir / f"{animal_id}_deform_dim{dim}.npy", field)
            print(f"Saved deformation fields for {animal_id}")
    
    return param_dir

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

def process_parameter_set(loaded_images, param_set, output_dir, save_visualizations=False, animal_ids=None):
    """Process a single parameter set
    
    Args:
        loaded_images (dict): Dictionary of loaded brain images
        param_set (dict): Registration parameters to use
        output_dir (Path): Path to the output directory
        save_visualizations (bool): Whether to generate visualizations
        animal_ids (list): List of animal IDs
        
    Returns:
        dict: Results including metrics and paths
    """
    process_id = os.getpid()
    
    # Create parameter-specific output directory for this run
    param_str = f"grid{param_set['grid_sampling_factor']}_scale{param_set['scale_sampling']}_speed{param_set['speed_factor']}_finalscale{param_set['final_scale']}_levels{param_set['scale_levels']}_finalgrid{param_set['final_grid_sampling']}_noise{param_set['noise_factor']}"
    param_dir = os.path.join(output_dir, param_str)
    os.makedirs(param_dir, exist_ok=True)
    
    # Set up logging for this process
    debug_log_file = os.path.join(param_dir, f"debug_process_{process_id}.log")
    logger = logging.getLogger(f"registration_debug_{process_id}")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler(debug_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Log starting information
    logger.info(f"Process {process_id} starting with parameters: {param_set}")
    print(f"Process {process_id} starting with parameters: {param_set}")
    logger.debug(f"Output directory: {param_dir}")
    
    try:
        # Log detailed information about the input images
        logger.debug(f"Number of images in loaded_images: {len(loaded_images)}")
        logger.debug(f"Animal IDs: {list(loaded_images.keys())}")
        
        # Register brain images
        logger.info("Starting brain registration")
        reg, deforms, transformed_images, elapsed_time = register_brain_images(loaded_images, param_set, logger)
        
        # Calculate average template
        logger.info("Calculating average template")
        template = calculate_average_template(transformed_images)
        logger.debug(f"Template shape: {template.shape}, min: {template.min()}, max: {template.max()}")
        
        # Calculate quality metrics
        logger.info("Calculating quality metrics")
        metrics = calculate_registration_quality(transformed_images)
        logger.debug(f"Metrics: {metrics}")
        
        # Save registration results
        logger.info("Saving registration results")
        param_dir_path = save_registration_results(
            transformed_images, 
            template, 
            deforms, 
            animal_ids if animal_ids else list(loaded_images.keys()), 
            output_dir,
            param_set,
            metrics,
            elapsed_time
        )
        
        # Create visualizations if requested
        if save_visualizations:
            logger.info("Creating visualizations")
            create_visualizations(loaded_images, transformed_images, template, param_dir_path)
        
        # Prepare serializable result
        logger.info("Preparing result dictionary")
        result = {
            'parameters': {
                'grid_sampling_factor': float(param_set['grid_sampling_factor']),
                'scale_sampling': int(param_set['scale_sampling']),
                'speed_factor': float(param_set['speed_factor']),
                'final_scale': float(param_set['final_scale']),
                'scale_levels': int(param_set['scale_levels']),
                'final_grid_sampling': float(param_set['final_grid_sampling']),
                'noise_factor': float(param_set['noise_factor'])
            },
            'metrics': {
                'mean_mse': float(metrics['mean_mse']),
                'std_mse': float(metrics['std_mse']),
                'mean_cc': float(metrics['mean_cc']),
                'std_cc': float(metrics['std_cc'])
            },
            'elapsed_time': float(elapsed_time),
            'output_dir': str(param_dir_path),
            'debug_log': debug_log_file
        }
        
        logger.info("Processing completed successfully")
        return result
    
    except Exception as e:
        error_message = f"Error in process {process_id} with parameters {param_set}: {str(e)}"
        print(error_message)
        logger.error(error_message)
        logger.exception("Detailed traceback:")
        
        # Try to log more information about the images if possible
        try:
            for animal_id, img in loaded_images.items():
                logger.debug(f"Image {animal_id} stats: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}, has_nan={np.isnan(img).any()}")
        except Exception as img_error:
            logger.error(f"Error while logging image info: {str(img_error)}")
        
        return {
            'parameters': param_set,
            'error': str(e),
            'debug_log': debug_log_file
        }

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Convert paths to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    main_logger = setup_debug_logging(output_dir)
    main_logger.info(f"Starting brain registration with data_dir={data_dir}, output_dir={output_dir}")
    
    # Check if data directory exists
    if not data_dir.exists():
        error_msg = f"Error: Data directory '{data_dir}' does not exist!"
        print(error_msg)
        main_logger.error(error_msg)
        return 1
    
    # Set up parameter scanning
    grid_sampling_factors = args.grid_sampling_factors
    scale_samplings = args.scale_samplings
    speed_factors = args.speed_factors
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        grid_sampling_factors, 
        scale_samplings, 
        speed_factors,
        args.final_scales,
        args.scale_levels,
        args.final_grid_samplings,
        args.noise_factors
    ))
    
    # If param_subset is specified, take only that many combinations
    if args.param_subset > 0 and args.param_subset < len(param_combinations):
        param_combinations = param_combinations[:args.param_subset]
    
    print(f"Will test {len(param_combinations)} parameter combinations:")
    for i, (gsf, ss, sf, fs, sl, fgs, nf) in enumerate(param_combinations):
        print(f"  {i+1}. grid_sampling_factor={gsf}, scale_sampling={ss}, speed_factor={sf}, "
              f"final_scale={fs}, scale_levels={sl}, final_grid_sampling={fgs}, noise_factor={nf}")
    
    # Find all brain h5 files
    brain_images = find_brain_h5_files(data_dir)
    if not brain_images:
        print("Error: No brain h5 files found!")
        return 1
    
    print(f"Found {len(brain_images)} brain images:")
    for animal_id, file_path in brain_images.items():
        print(f"  {animal_id}: {file_path}")
    
   
    # Load and preprocess images
    loaded_images = {}
    for animal_id, file_path in tqdm(brain_images.items(), desc="Loading images"):
        loaded_images[animal_id] = load_and_preprocess_h5_image(
            file_path,
            args.threshold,
            args.h5_resolution
        )
    
    # Set up parameter sets for multiprocessing
    param_sets = []
    for grid_sampling_factor, scale_sampling, speed_factor, final_scale, scale_levels, final_grid_sampling, noise_factor in param_combinations:
        param_sets.append({
            'grid_sampling_factor': grid_sampling_factor,
            'scale_sampling': scale_sampling,
            'speed_factor': speed_factor,
            'final_scale': final_scale,
            'scale_levels': scale_levels,
            'final_grid_sampling': final_grid_sampling,
            'noise_factor': noise_factor
        })
    
    # Create the output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create process pool
    if args.max_processes <= 0:
        max_processes = multiprocessing.cpu_count()
    else:
        max_processes = min(args.max_processes, multiprocessing.cpu_count())
    
    print(f"Using {max_processes} processes for parameter scanning")
    main_logger.info(f"Using {max_processes} processes for parameter scanning")
    
    # Execute parameter scanning
    overall_start_time = time.time()
    
    # Create a copy of loaded_images that can be shared between processes
    # This is important to avoid unnecessary memory usage
    shared_loaded_images = loaded_images
    animal_ids = list(brain_images.keys())
    
    # Log information about shared data
    main_logger.debug(f"Shared loaded_images contains {len(shared_loaded_images)} images")
    main_logger.debug(f"Animal IDs: {animal_ids}")
    
    # Log parameter sets
    main_logger.debug(f"Parameter sets to test: {param_sets}")
    
    with multiprocessing.Pool(processes=max_processes) as pool:
        main_logger.info(f"Created process pool with {max_processes} processes")
        results = []
        for i, param_set in enumerate(param_sets):
            main_logger.debug(f"Submitting parameter set {i+1}/{len(param_sets)}: {param_set}")
            result = pool.apply_async(
                process_parameter_set, 
                args=(shared_loaded_images, param_set, output_dir, args.save_visualizations, animal_ids)
            )
            results.append(result)
        
        # Wait for all processes to complete
        main_logger.info(f"Waiting for {len(results)} processes to complete...")
        all_results = []
        for i, result in enumerate(results):
            try:
                main_logger.debug(f"Getting result from process {i+1}/{len(results)}")
                all_results.append(result.get())
            except Exception as e:
                error_msg = f"Error getting result from process {i+1}: {str(e)}"
                print(error_msg)
                main_logger.error(error_msg)
                main_logger.exception("Exception details:")
                all_results.append({'error': str(e)})
    
    overall_elapsed_time = time.time() - overall_start_time
    
    # Save overall results
    main_logger.info("Saving overall results")
    overall_results = {
        'parameter_sets': int(len(param_combinations)),
        'elapsed_time': float(overall_elapsed_time),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'h5_resolution': args.h5_resolution,  # Resolution level used from h5 file
        'threshold': args.threshold,  # Add threshold value
        'results': all_results
    }
    
    # Count successful registrations vs errors
    error_count = sum(1 for r in all_results if 'error' in r)
    success_count = len(all_results) - error_count
    main_logger.info(f"Completed {len(all_results)} parameter sets: {success_count} successful, {error_count} with errors")
    
    # Log errors if any
    if error_count > 0:
        main_logger.warning("Errors encountered during parameter scan:")
        for i, result in enumerate(all_results):
            if 'error' in result:
                main_logger.warning(f"  Parameter set {i+1}: {result.get('parameters', 'unknown')} - Error: {result['error']}")
    
    results_file = os.path.join(output_dir, 'parameter_scan_results.json')
    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=4, cls=NumpyEncoder)
    main_logger.info(f"Saved results to {results_file}")
    
    # Sort results by quality metrics
    main_logger.info("Sorting results by quality metrics")
    sorted_results = sorted(
        [r for r in all_results if 'metrics' in r], 
        key=lambda x: x['metrics']['mean_cc'],
        reverse=True
    )
    
    # Print summary
    print("\nParameter Scan Complete!")
    print(f"Total time: {overall_elapsed_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    main_logger.info(f"Parameter scan completed in {overall_elapsed_time:.2f} seconds")
    
    # Log and print top results
    if sorted_results:
        print("\nTop 3 parameter sets by cross-correlation:")
        main_logger.info("Top parameter sets by cross-correlation:")
        
        for i, result in enumerate(sorted_results[:3]):
            params = result['parameters']
            result_str = (f"{i+1}. grid={params['grid_sampling_factor']}, scale={params['scale_sampling']}, "
                  f"speed={params['speed_factor']}, final_scale={params['final_scale']}, "
                  f"levels={params['scale_levels']}, final_grid={params['final_grid_sampling']}, "
                  f"noise={params['noise_factor']}: mean_cc={result['metrics']['mean_cc']:.4f}, "
                  f"mean_mse={result['metrics']['mean_mse']:.4f}, time={result['elapsed_time']:.2f}s")
            print(result_str)
            main_logger.info(result_str)
    else:
        main_logger.warning("No successful parameter sets to rank")
    
    print("\n3D registration parameter scan completed successfully!")
    main_logger.info("3D registration parameter scan completed successfully!")
    return 0

if __name__ == "__main__":
    # Ensure multiprocessing works correctly
    multiprocessing.set_start_method('spawn', force=True)
    sys.exit(main())