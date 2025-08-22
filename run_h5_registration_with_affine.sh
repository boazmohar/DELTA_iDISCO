#!/bin/bash

# Set the ELASTIX_PATH environment variable to point to your Elastix installation
export ELASTIX_PATH="/groups/spruston/home/moharb/elastix/bin"

# Verify elastix path
if [ ! -d "$ELASTIX_PATH" ]; then
    echo "Warning: Elastix directory $ELASTIX_PATH does not exist!"
    echo "Please create it or modify the path above."
    exit 1
fi

echo "Using Elastix from: $ELASTIX_PATH"
echo "Checking for elastix binary..."
if [ -f "$ELASTIX_PATH/elastix" ]; then
    echo "Found elastix binary ✓"
else
    echo "Warning: elastix binary not found at $ELASTIX_PATH/elastix!"
    echo "Make sure elastix is properly installed."
    exit 1
fi

# Example script to run the h5-based 3D registration with affine initialization
# This version uses pre-downsampled datasets and performs affine registration as an initialization step

# Set the directories
DATA_DIR="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3"
OUTPUT_DIR="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/registration_results_h5_affine"

# Resolution options:
# 1 = Data (full resolution)
# 2 = Data_2_2_2 (2x downsampled in each dimension)
# 4 = Data_4_4_4 (4x downsampled in each dimension)
H5_RESOLUTION=4  # Using 4x downsampled for faster processing

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the registration
python register_brains_3d_parallel_h5.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --h5_resolution $H5_RESOLUTION \
  --threshold 100 \
  --grid_sampling_factors 1 \
  --scale_samplings 10 \
  --speed_factors 1 2 \
  --final_scales 1.0 \
  --scale_levels 4 \
  --final_grid_samplings 10 \
  --save_visualizations \
  --max_processes 44 \
  --elastix_path "$ELASTIX_PATH"

echo "Registration complete. Results saved to $OUTPUT_DIR"