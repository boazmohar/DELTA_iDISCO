#!/usr/bin/env python3
"""
Groupwise registration script using NiftyReg through nipype workflows.
This approach creates an average brain from multiple inputs and registers
all brains to this average.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import nibabel as nib
import h5py
import tifffile
import multiprocessing
import glob
import re
from niflow.nipype1.workflows.smri.niftyreg import create_groupwise_average
from nipype import config, logging
from concurrent.futures import ProcessPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description='Perform groupwise registration using NiftyReg')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input brain h5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for registered images and transformations')
    parser.add_argument('--dataset', type=str, default='Data_4_4_4',
                        choices=['Data', 'Data_2_2_2', 'Data_4_4_4'],
                        help='Dataset to use from the h5 files')
    parser.add_argument('--file_pattern', type=str, default='*.h5',
                        help='Pattern to match h5 files')
    parser.add_argument('--threshold', type=int, default=100,
                        help='Threshold value for image preprocessing')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations for groupwise averaging')
    parser.add_argument('--n_procs', type=int, default=multiprocessing.cpu_count(),
                        help='Number of parallel processes to use')
    parser.add_argument('--plugin', type=str, default='MultiProc',
                        choices=['Linear', 'MultiProc', 'PBS'],
                        help='Nipype plugin to use for parallel execution')
    parser.add_argument('--omp_num_threads', type=int, default=8,
                        help='Number of OpenMP threads per job (set OMP_NUM_THREADS)')
    parser.add_argument('--skip_conversion', action='store_true',
                        help='Skip h5 to NIfTI conversion if NIfTI files already exist')
    parser.add_argument('--nifti_dir', type=str, default=None,
                        help='Directory with existing NIfTI files (if skipping conversion)')
    return parser.parse_args()

def read_h5_image(file_path, dataset_name='Data', threshold=100):
    """Read an image from an h5 file"""
    try:
        print(f"Opening h5 file: {file_path}")
        with h5py.File(file_path, 'r') as h5_file:
            # Check if the requested dataset exists
            if dataset_name not in h5_file:
                print(f"Dataset '{dataset_name}' not found. Available datasets: {list(h5_file.keys())}")
                return None
                
            print(f"Reading dataset: {dataset_name}")
            data = h5_file[dataset_name][:]
            print(f"Data shape: {data.shape}, dtype: {data.dtype}")
            
            # Apply threshold
            data[data < threshold] = threshold
            print(f"After thresholding - min: {data.min()}, max: {data.max()}")
            
            return data
    except Exception as e:
        print(f"Error reading h5 file: {e}")
        return None

def extract_animal_id(file_path):
    """Extract animal ID from a file path"""
    animal_id = "unknown"
    path_parts = str(file_path).split('/')
    for part in path_parts:
        if part.startswith("ANM"):
            animal_id = part
            break
    return animal_id

def h5_to_nifti_filename(file_path, animal_id=None):
    """Generate a consistent NIfTI filename from an h5 file path"""
    if animal_id is None:
        animal_id = extract_animal_id(file_path)
    basename = os.path.basename(file_path)
    return f"{animal_id}_{basename.replace('.h5', '.nii.gz')}"

def convert_h5_to_nii_worker(args):
    """Worker function for parallel h5 to NIfTI conversion"""
    file_path, output_dir, dataset_name, threshold = args
    return convert_h5_to_nii(file_path, output_dir, dataset_name, threshold)

def convert_h5_to_nii(file_path, output_dir, dataset_name, threshold):
    """Convert h5 image to NIfTI format for NiftyReg"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the h5 image
    data = read_h5_image(file_path, dataset_name, threshold)
    if data is None:
        return None
        
    # Create NIfTI image (using identity affine since we don't have spatial information)
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(data, affine)
    
    # Generate output path with animal number
    animal_id = extract_animal_id(file_path)
    file_name = h5_to_nifti_filename(file_path, animal_id)
    output_path = os.path.join(output_dir, file_name)
    
    # Save NIfTI file
    nib.save(nii_img, output_path)
    print(f"Saved NIfTI file: {output_path}")
    
    return output_path

def save_results_as_tiff(nii_path, output_dir, name=None, animal_id=None):
    """Save NIfTI result as TIFF file"""
    # Load NIfTI file
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output name if not provided
    if name is None:
        basename = os.path.basename(nii_path).replace('.nii.gz', '')
        if animal_id:
            name = f"{animal_id}_{basename}"
        else:
            name = basename
    
    # Convert to uint16 for TIFF file
    img_min, img_max = np.min(data), np.max(data)
    if img_max > img_min:
        scaled = (data - img_min) / (img_max - img_min) * 65535
    else:
        scaled = data
    uint16_data = scaled.astype(np.uint16)
    
    # Save as TIFF file
    output_path = os.path.join(output_dir, f"{name}.tif")
    tifffile.imwrite(output_path, uint16_data)
    print(f"Saved TIFF file: {output_path}")
    
    return output_path

def run_groupwise_registration(input_files, output_dir, iterations=3, n_procs=None, plugin='MultiProc', omp_num_threads=8):
    """
    Run groupwise registration using NiftyReg through nipype
    
    Args:
        input_files: List of NIfTI files to register
        output_dir: Output directory for results
        iterations: Number of iterations for groupwise averaging
    
    Returns:
        Dictionary with results including average image and transform files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define options with just the OMP core value
    linear_options = {
        'omp_core_val': omp_num_threads
    }
    
    nonlinear_options = {
        'omp_core_val': omp_num_threads
    }
    
    # Create groupwise registration workflow with thread options
    print(f"Setting up groupwise registration workflow with {omp_num_threads} OpenMP cores per job...")
    workflow = create_groupwise_average(
        name='groupwise_reg',
        linear_options_hash=linear_options,
        nonlinear_options_hash=nonlinear_options
    )
    
    # Set input files
    workflow.inputs.inputspec.in_files = input_files
    
    # Initial reference image (use first image as initial reference)
    workflow.inputs.inputspec.ref_file = input_files[0]
    
    # Set output directory
    workflow.base_dir = output_dir
    
    # Configure Nipype for parallel processing
    if n_procs is None:
        n_procs = multiprocessing.cpu_count()
    
    # Configure workflow to use parallel processing
    if plugin == 'MultiProc':
        print(f"Running with MultiProc plugin using {n_procs} processes, {omp_num_threads} threads per process")
        
        # Set OMP_NUM_THREADS in the environment (this will be inherited by child processes)
        set_omp_threads(omp_num_threads)
        
        # Configure nipype
        nipype_config = dict(execution={'stop_on_first_crash': False,
                                     'hash_method': 'timestamp',
                                     'crashfile_format': 'txt',
                                     'parameterize_dirs': False,
                                     'remove_unnecessary_outputs': False,
                                     'use_relative_paths': True,
                                     'plugin': plugin,
                                     'plugin_args': {'n_procs': n_procs}})
        config.update_config(nipype_config)
        logging.update_logging(config)
    
    # Run the workflow
    print(f"Running groupwise registration with {len(input_files)} input files...")
    result = workflow.run(plugin=plugin, plugin_args={'n_procs': n_procs})
    
    # Get outputs
    outputs = {}
    # Extract average image and transformation files from result
    # Note: Exact output path structure depends on the workflow implementation
    average_image = os.path.join(output_dir, 'groupwise_reg', 'average_image', 'result_avg_img.nii.gz')
    if os.path.exists(average_image):
        outputs['average_image'] = average_image
    
    # Find transformation files
    transform_dir = os.path.join(output_dir, 'groupwise_reg', 'transformations')
    if os.path.exists(transform_dir):
        transform_files = [os.path.join(transform_dir, f) for f in os.listdir(transform_dir) if f.endswith('.txt')]
        outputs['transform_files'] = transform_files
    
    # Find transformed images
    transformed_dir = os.path.join(output_dir, 'groupwise_reg', 'transformed_images')
    if os.path.exists(transformed_dir):
        transformed_files = [os.path.join(transformed_dir, f) for f in os.listdir(transformed_dir) if f.endswith('.nii.gz')]
        outputs['transformed_images'] = transformed_files
    
    return outputs

def set_omp_threads(omp_num_threads):
    """Set OMP threads environment variables globally"""
    # Set the OMP_NUM_THREADS environment variable globally
    os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(omp_num_threads)
    # Also set NSLOTS which some software uses
    os.environ["NSLOTS"] = str(omp_num_threads)
    print(f"Setting OMP_NUM_THREADS={omp_num_threads} globally")

def main():
    args = parse_args()
    
    # Set the OMP_NUM_THREADS environment variable globally
    set_omp_threads(args.omp_num_threads)
    print(f"Thread control: each registration process will use {args.omp_num_threads} cores")
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up NIfTI directory
    if args.nifti_dir:
        # Use provided NIfTI directory
        nifti_dir = args.nifti_dir
        if not os.path.isdir(nifti_dir):
            print(f"Error: Provided NIfTI directory does not exist: {nifti_dir}")
            return 1
    else:
        # Create a temporary directory for NIfTI files
        nifti_dir = os.path.join(args.output_dir, 'nifti_files')
        os.makedirs(nifti_dir, exist_ok=True)
    
    # List of final NIfTI files to use for registration
    nifti_files = []
    
    if args.skip_conversion and args.nifti_dir:
        # Skip conversion and use existing NIfTI files
        print(f"Skipping conversion and using existing NIfTI files in {nifti_dir}")
        # Find all NIfTI files in the specified directory
        nifti_pattern = os.path.join(nifti_dir, "*.nii.gz")
        nifti_files = sorted(glob.glob(nifti_pattern))
        
        # Filter to only include files with ANM pattern if needed
        anm_pattern = re.compile(r'ANM\d+')  # Pattern for animal IDs
        nifti_files = [f for f in nifti_files if anm_pattern.search(os.path.basename(f))]
        
        if len(nifti_files) == 0:
            print(f"Error: No NIfTI files found in {nifti_dir}")
            print("Will proceed with conversion from h5 files")
            args.skip_conversion = False
        else:
            print(f"Found {len(nifti_files)} existing NIfTI files for processing")
    
    if not args.skip_conversion or len(nifti_files) == 0:
        # Find all h5 files in the input directory
        input_files = []
        for root, dirs, _ in os.walk(args.input_dir):
            for d in dirs:
                if d.startswith('ANM'):
                    # Look for h5 files in animal directories
                    animal_path = os.path.join(root, d)
                    pattern = args.file_pattern.replace('*/', '')
                    matched_files = list(Path(animal_path).glob(pattern))
                    input_files.extend(matched_files)
                    if matched_files:
                        print(f"Found {len(matched_files)} files in {animal_path}")

        if len(input_files) == 0:
            print(f"Error: No h5 files found in {args.input_dir} with pattern {args.file_pattern}")
            return 1
        
        print(f"Found {len(input_files)} h5 files for processing")
        
        # Sort files by animal ID for better logging
        input_files.sort(key=lambda x: str(x))
        
        # Check if any NIfTI files already exist before conversion
        new_conversion_needed = []
        existing_nifti_files = []
        
        for h5_file in input_files:
            # Generate the expected NIfTI filename
            animal_id = extract_animal_id(h5_file)
            nii_filename = h5_to_nifti_filename(h5_file, animal_id)
            nii_path = os.path.join(nifti_dir, nii_filename)
            
            # Check if the NIfTI file already exists
            if os.path.exists(nii_path):
                print(f"NIfTI file already exists: {nii_path}")
                existing_nifti_files.append(nii_path)
            else:
                # Need to convert this file
                new_conversion_needed.append((str(h5_file), nifti_dir, args.dataset, args.threshold))
        
        # Only convert files that don't already have NIfTI versions
        if new_conversion_needed:
            print(f"Converting {len(new_conversion_needed)} h5 files to NIfTI format using {args.n_procs} processes...")
            # Use ProcessPoolExecutor for parallel conversion
            with ProcessPoolExecutor(max_workers=args.n_procs) as executor:
                results = list(executor.map(convert_h5_to_nii_worker, new_conversion_needed))
            
            # Filter out None results and combine with existing files
            converted_files = [result for result in results if result is not None]
            nifti_files = existing_nifti_files + converted_files
        else:
            nifti_files = existing_nifti_files
            print("All required NIfTI files already exist, skipping conversion")
        
    # Make sure we have NIfTI files to work with
    if len(nifti_files) == 0:
        print("Error: No valid NIfTI files available for registration")
        return 1
    
    if len(nifti_files) == 0:
        print("Error: Failed to convert any h5 files to NIfTI format")
        return 1
    
    print(f"Converted {len(nifti_files)} h5 files to NIfTI format")
    
    # Set thread control for NiftyReg
    set_omp_threads(args.omp_num_threads)
    
    # Run groupwise registration
    results = run_groupwise_registration(nifti_files, args.output_dir, args.iterations, 
                                       args.n_procs, args.plugin, args.omp_num_threads)
    
    # Save results as TIFF files
    if 'average_image' in results:
        save_results_as_tiff(results['average_image'], args.output_dir, 'average_brain')
    
    # Save individual registered images as TIFF if available
    if 'transformed_images' in results:
        for img_path in results['transformed_images']:
            # Extract animal ID from the filename
            basename = os.path.basename(img_path)
            animal_id = None
            for part in basename.split('_'):
                if part.startswith("ANM"):
                    animal_id = part
                    break
            save_results_as_tiff(img_path, args.output_dir, None, animal_id)
    
    print(f"Groupwise registration completed. Results saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())