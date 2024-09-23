import os
import argparse
import logging
from utils import read_h5_image, match_h5_files_by_channels, setup_logging
from registration import register_and_transform
from stats import compute_region_stats
from skimage import io
import numpy as np
import itk

def process_animal(animal, files, base_dir, fx, param_files, annotation_np):
    """
    Process a single animal by:
    - Registering and transforming the images
    - Computing regional statistics
    """
    output_dir = os.path.join(base_dir, animal, 'itk')  # Ensure logs and outputs go to /itk
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Register and Transform
    logging.info("Starting registration and transformation.")
    register_and_transform(fx, files, output_dir, param_files)
    logging.info("Finished registration and transformation.")

    # Step 2: Compute Region Statistics
    logging.info("Starting computation of region statistics.")
    compute_region_stats(files, output_dir, annotation_np)
    logging.info("Finished computation of region statistics.")
    
    # Log the location of the logs and output files
    log_file = os.path.join(output_dir, f'registration_log_{animal}.txt')
    logging.info(f"Processing logs and outputs are saved in: {log_file}")


if __name__ == "__main__":
    # Argument parser to get input from the command line
    parser = argparse.ArgumentParser(description="Process an animal for registration and analysis.")
    
    parser.add_argument('--animal', type=str, required=True, default='ANM550749_left_JF552',
                        help="Animal name (e.g., ANM550749_left_JF552)")
    parser.add_argument('--base_dir', type=str, default='/nrs/spruston/Boaz/I2/2024-09-19_iDISCO_CalibrationBrains',
                        help="Base directory where animal data is stored")
    parser.add_argument('--fx', type=str, default='/nrs/spruston/Boaz/I2/atlas10_hemi.tif', 
                        help="Path to the atlas fixed image (fx) in TIFF format")
    parser.add_argument('--param_files_dir', type=str, default='/nrs/spruston/Boaz/I2/itk', 
                        help="Directory containing the parameter files for registration")
    parser.add_argument('--annotation_np', type=str, default='/nrs/spruston/Boaz/I2/annotatin10_hemi.tif', 
                        help="Path to the annotation volume file in TIFF format")
    
    args = parser.parse_args()

    # Setup logging specific to this animal
    setup_logging(args.base_dir, args.animal)

    # Load the fixed image (fx)
    logging.info(f"Loading fixed image (fx) from {args.fx}.")
    fx = itk.imread(args.fx, pixel_type=itk.US)
    
    # Load the parameter files
    param_files = [
        os.path.join(args.param_files_dir, 'Order1_Par0000affine.txt'),
        os.path.join(args.param_files_dir, 'Order3_Par0000bspline.txt'),
        os.path.join(args.param_files_dir, 'Order4_Par0000bspline.txt'),
        os.path.join(args.param_files_dir, 'Order5_Par0000bspline.txt')
    ]
    logging.info(f"Loaded parameter files from {args.param_files_dir}.")

    # Load the annotation volume
    logging.info(f"Loading annotation volume from {args.annotation_np}.")
    annotation_np = np.int64(io.imread(args.annotation_np))
    
    # Match H5 files by channels for all animals
    logging.info(f"Matching H5 files in {args.base_dir}.")
    animals_files = match_h5_files_by_channels(args.base_dir)

    # Get the files for the selected animal
    animal = args.animal
    files = animals_files.get(animal)
    
    if not files:
        logging.error("No files found for this animal.")
        print(f"No files found for animal {animal}")
    else:
        logging.info("Files found. Starting processing.")
        # Process the animal
        process_animal(animal, files, args.base_dir, fx, param_files, annotation_np)

    logging.info("Processing completed.")
