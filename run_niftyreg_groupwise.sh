#!/bin/bash

# Define input and output directories
INPUT_DIR="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3"
OUTPUT_DIR="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/niftyreg_groupwise_output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Activate the NiftyReg conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate niftyreg-env

# Set OpenMP thread count explicitly
export OMP_NUM_THREADS=8
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8

# Run the groupwise registration script
echo "Starting NiftyReg groupwise registration..."
python niftyreg_groupwise_registration.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset "Data_2_2_2" \
  --file_pattern "*/uni_tp-0_ch-0_st-0-x00-y00_obj-right_cam-long_etc.lux.h5" \
  --threshold 100 \
  --iterations 3 \
  --n_procs 8 \
  --plugin MultiProc \
  --omp_num_threads 8 \
  --skip_conversion \
  --nifti_dir "$OUTPUT_DIR/nifti_files"

# Show the exit status
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Groupwise registration completed successfully!"
else
    echo "Groupwise registration failed with exit code $STATUS"
fi

# Deactivate the conda environment
conda deactivate