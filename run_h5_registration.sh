#!/bin/bash

# Example script to run the h5-based 3D registration with pre-downsampled datasets

# Usage: bash run_h5_registration.sh

# Set the directories
DATA_DIR="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3"
OUTPUT_DIR="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/registration_results_h5"

# Resolution options:
# 1 = Data (full resolution)
# 2 = Data_2_2_2 (2x downsampled in each dimension)
# 4 = Data_4_4_4 (4x downsampled in each dimension)
H5_RESOLUTION=2

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the registration
python register_brains_3d_parallel_h5.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --h5_resolution $H5_RESOLUTION \
  --threshold 100 \
  --grid_sampling_factors 0.75 1 \
  --scale_samplings 45 65 100 \
  --speed_factors 6 10 20 \
  --final_scales 1.0 \
  --scale_levels 4 6 8 \
  --final_grid_samplings 1.0 2.0 \
  --save_visualizations

echo "Registration complete. Results saved to $OUTPUT_DIR"