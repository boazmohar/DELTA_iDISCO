import os
import argparse
from .utils import match_h5_files_by_channels
from .registration_and_transformation import register_and_transform
from .stats import compute_region_stats
from skimage import io
import numpy as np

def process_animal(animal, base_dir, fx, param_files, annotation_np):
    animals = match_h5_files_by_channels(base_dir)
    files = animals[animal]
    output_dir = os.path.join(base_dir, animal, 'itk')
    
    # Step 1: Register and Transform
    register_and_transform(fx, files, output_dir, param_files)
    
    # Step 2: Compute Region Statistics
    compute_region_stats(files, output_dir, annotation_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an animal for registration and analysis.")
    parser.add_argument('--animal', type=str, required=True, help="Animal name (e.g., ANM549057_left_JF522)")
    args = parser.parse_args()

    base_dir = '/nrs/spruston/Boaz/I2/2024-09-19_iDISCO_CalibrationBrains'
    fx = itk.imread('/nrs/spruston/Boaz/I2/atlas10_hemi.tif', pixel_type=itk.US)
    param_files = [
        '/nrs/spruston/Boaz/I2/itk/Order1_Par0000affine.txt',
        '/nrs/spruston/Boaz/I2/itk/Order3_Par0000bspline.txt',
        '/nrs/spruston/Boaz/I2/itk/Order4_Par0000bspline.txt',
        '/nrs/spruston/Boaz/I2/itk/Order5_Par0000bspline.txt'
    ]
    annotation_np = np.int64(io.imread('/nrs/spruston/Boaz/I2/annotatin10_hemi.tif'))

    process_animal(args.animal, base_dir, fx, param_files, annotation_np)
