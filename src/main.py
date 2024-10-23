import os
import argparse
import logging
from DELAT_utils import read_h5_image, match_h5_files_by_channels, setup_logging
from registration import register_and_transform
from stats import compute_region_stats
from skimage import io
import numpy as np
import itk

# Mapping of string names to actual NumPy functions
FUNCTION_MAP = {
    'mean': np.mean,
    'median': np.median,
    'std': np.std,
    'min': np.min,
    'max': np.max
}

def process_animal(animal, files, base_dir, fx, param_files, annotation_np, funcs, logger):
    """
    Process a single animal by:
    - Registering and transforming the images
    - Computing regional statistics
    """
    output_dir = os.path.join(base_dir, animal, 'itk')  # Ensure logs and outputs go to /itk
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Register and Transform
    logger.info(f"Starting registration and transformation for {animal}.")
    register_and_transform(fx, files, output_dir, param_files, logger)
    logger.info(f"Finished registration and transformation for {animal}.")

    # Step 2: Compute Region Statistics
    logger.info(f"Starting computation of region statistics for {animal}.")

    # Compute region statistics in parallel using shared memory and dynamic number of cores
    num_cores = os.cpu_count()  # Automatically detect the number of cores
    compute_region_stats(files, output_dir, annotation_np, funcs, num_cores)
    logger.info(f"Processing logs and outputs are saved in {output_dir}")


if __name__ == "__main__":
    # Argument parser to get input from the command line
    parser = argparse.ArgumentParser(description="Process an animal for registration and analysis.")
    
    parser.add_argument('--animal', type=str, required=True, help="Animal name (e.g., ANM550749_left_JF552)")
    parser.add_argument('--base_dir', type=str, default='/nrs/spruston/Boaz/I2/2024-09-19_iDISCO_CalibrationBrains',
                        help="Base directory where animal data is stored")
    parser.add_argument('--fx', type=str, default='/nrs/spruston/Boaz/I2/atlas10_hemi.tif', 
                        help="Path to the atlas fixed image (fx) in TIFF format")
    parser.add_argument('--param_files_dir', type=str, default='/nrs/spruston/Boaz/I2/itk', 
                        help="Directory containing the parameter files for registration")
    parser.add_argument('--annotation_np', type=str, default='/nrs/spruston/Boaz/I2/annotatin10_hemi.tif', 
                        help="Path to the annotation volume file in TIFF format")
    
    # New argument: List of functions to compute
    parser.add_argument('--functions', type=str, nargs='+', default=['mean', 'median', 'std'],
                        help="List of statistical functions to compute. Options: mean, median, std, min, max")

    args = parser.parse_args()

    # Setup logging specific to this animal, logging to both file and console
    logger = setup_logging(args.base_dir, args.animal)

    # Load the fixed image (fx)
    logger.info(f"Loading fixed image (fx) from {args.fx}.")
    fx = itk.imread(args.fx, pixel_type=itk.US)
    
    # Load the parameter files
    param_files = [
        os.path.join(args.param_files_dir, 'Order1_Par0000affine.txt'),
        os.path.join(args.param_files_dir, 'Order3_Par0000bspline.txt'),
        os.path.join(args.param_files_dir, 'Order4_Par0000bspline.txt'),
        os.path.join(args.param_files_dir, 'Order5_Par0000bspline.txt')
    ]
    logger.info(f"Loaded parameter files from {args.param_files_dir}.")

    # Load the annotation volume
    logger.info(f"Loading annotation volume from {args.annotation_np}.")
    itk_annotation = itk.imread(args.annotation_np, itk.ULL)
    annotation_np = itk.array_view_from_image(itk_annotation)
    
    # Match H5 files by channels for all animals
    logger.info(f"Matching H5 files in {args.base_dir}.")
    animals_files = match_h5_files_by_channels(args.base_dir)

    # Get the files for the selected animal
    animal = args.animal
    files = animals_files.get(animal)
    
    if not files:
        logger.error(f"No files found for animal {animal}.")
        print(f"No files found for animal {animal}")
    else:
        logger.info(f"Files found for {animal}. Starting processing.")

        # Convert the list of function names into actual NumPy functions
        funcs = [FUNCTION_MAP[func] for func in args.functions if func in FUNCTION_MAP]

        if not funcs:
            logger.error("No valid functions specified. Exiting.")
            exit(1)

        # Process the animal
        process_animal(animal, files, args.base_dir, fx, param_files, annotation_np, funcs, logger)

    logger.info(f"Processing completed for {animal}.")
