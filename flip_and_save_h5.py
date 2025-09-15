#!/usr/bin/env python3
"""
Script to flip an h5 image along a specified axis and save it back to the original file.
This is used to correct orientation differences between samples.
"""

import argparse
import os
import numpy as np
import h5py
import time
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Flip an h5 image along a specified axis')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input h5 file')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save the output h5 file (defaults to input_file_flipped.h5)')
    parser.add_argument('--axis', type=int, default=2, 
                        help='Axis to flip along (0=Z, 1=Y, 2=X, default is X axis which is usually the shortest)')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['Data', 'Data_2_2_2', 'Data_4_4_4'],
                        help='Datasets to flip within the h5 file')
    parser.add_argument('--backup', action='store_true',
                        help='Create a backup of the original file')
    parser.add_argument('--in_place', action='store_true',
                        help='Modify the file in place (overwrites original file)')
    return parser.parse_args()

def flip_h5_file(input_file, output_file, flip_axis, datasets, create_backup=False):
    """
    Flip datasets in an h5 file along a specified axis
    
    Args:
        input_file: Path to the input h5 file
        output_file: Path to save the output h5 file
        flip_axis: Axis to flip along (0=Z, 1=Y, 2=X)
        datasets: List of datasets to flip
        create_backup: Whether to create a backup of the original file
    """
    # Create backup if requested
    if create_backup:
        backup_file = f"{input_file}.backup"
        print(f"Creating backup at: {backup_file}")
        shutil.copy2(input_file, backup_file)
    
    # Open the input file
    print(f"Opening h5 file: {input_file}")
    with h5py.File(input_file, 'r') as f_in:
        # List all datasets
        available_datasets = list(f_in.keys())
        print(f"Available datasets: {available_datasets}")
        
        # Process each requested dataset
        datasets_to_process = [d for d in datasets if d in available_datasets]
        if not datasets_to_process:
            raise ValueError(f"None of the requested datasets {datasets} found in file. Available datasets: {available_datasets}")
        
        print(f"Processing datasets: {datasets_to_process}")
        
        # For each dataset, read data, flip it, and prepare for writing
        flipped_data = {}
        for dataset in datasets_to_process:
            # Read the dataset
            print(f"Reading dataset: {dataset}")
            data = f_in[dataset][:]
            print(f"  Shape: {data.shape}, dtype: {data.dtype}")
            print(f"  Min: {data.min()}, Max: {data.max()}")
            
            # Flip the data along the specified axis
            print(f"Flipping along axis {flip_axis}...")
            flipped = np.flip(data, axis=flip_axis)
            print(f"  Flipped shape: {flipped.shape}")
            
            # Store for writing
            flipped_data[dataset] = flipped
            
            # Also copy all other datasets and attributes
            for key in available_datasets:
                if key not in datasets_to_process:
                    # Check if dataset is scalar or array
                    dset = f_in[key]
                    if dset.shape == () or dset.ndim == 0:  # Scalar dataset
                        # Read scalar values without slicing
                        flipped_data[key] = dset[()]
                    else:  # Regular dataset
                        flipped_data[key] = dset[:]
    
    # If output file is not specified, modify the input file
    if output_file is None:
        output_file = f"{os.path.splitext(input_file)[0]}_flipped.h5"
    
    # Write the flipped data to the output file
    print(f"Writing flipped data to: {output_file}")
    with h5py.File(output_file, 'w') as f_out:
        # Write each dataset
        for dataset, data in flipped_data.items():
            print(f"Writing dataset: {dataset}")
            # Only use compression for array datasets (not for scalar)
            if hasattr(data, 'shape') and data.shape != ():
                f_out.create_dataset(dataset, data=data, compression="gzip", compression_opts=9)
            else:
                # For scalar datasets, don't use compression
                f_out.create_dataset(dataset, data=data)
    
    print(f"Successfully saved flipped data to: {output_file}")
    return output_file

def main():
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        return 1
    
    # Determine output file
    output_file = args.output_file
    if args.in_place:
        # If in-place modification requested, use the input file as output
        output_file = args.input_file
        # Always create a backup in this case unless explicitly disabled
        create_backup = True
    else:
        # Otherwise, use the specified output file or a default name
        if output_file is None:
            output_file = f"{os.path.splitext(args.input_file)[0]}_flipped.h5"
        create_backup = args.backup
    
    try:
        # Process the file
        result_file = flip_h5_file(
            args.input_file, 
            output_file, 
            args.axis, 
            args.datasets,
            create_backup
        )
        print(f"Operation completed successfully.")
        print(f"Flipped file saved to: {result_file}")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())