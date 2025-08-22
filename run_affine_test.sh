#!/bin/bash

# Define h5 files to use
REF_H5_FILE="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/ANM555976/uni_tp-0_ch-0_st-0-x00-y00_obj-right_cam-long_etc.lux.h5"

# Find a second animal file to use as the moving image
MOV_H5_FILE="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/ANM555974/uni_tp-0_ch-0_st-0-x00-y00_obj-right_cam-long_etc.lux.h5"

# Check if files exist
if [ ! -f "$REF_H5_FILE" ]; then
    echo "Error: Reference h5 file does not exist: $REF_H5_FILE"
    exit 1
fi

if [ ! -f "$MOV_H5_FILE" ]; then
    echo "Error: Moving h5 file does not exist: $MOV_H5_FILE"
    exit 1
fi

echo "Using reference h5 file: $REF_H5_FILE"
echo "Using moving h5 file: $MOV_H5_FILE"

# Define output directory
OUTPUT_DIR="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/affine_test_output"
mkdir -p "$OUTPUT_DIR"

# Run the test
echo "Running affine-like registration test..."
python test_affine_like_registration.py \
  --reference_h5_file "$REF_H5_FILE" \
  --moving_h5_file "$MOV_H5_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --dataset "Data_4_4_4"

# Show the exit status
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed with exit code $STATUS"
fi