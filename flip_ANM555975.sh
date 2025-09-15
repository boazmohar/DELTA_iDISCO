#!/bin/bash

# Script to flip ANM555975 h5 file along the shortest axis
# This script demonstrates how to use flip_and_save_h5.py to correct orientation

# Define the path to the h5 file
H5_FILE="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/ANM555975/uni_tp-0_ch-0_st-0-x00-y00_obj-right_cam-long_etc.lux.h5"

# Check if the file exists
if [ ! -f "$H5_FILE" ]; then
    echo "Error: H5 file does not exist: $H5_FILE"
    exit 1
fi

echo "Found h5 file: $H5_FILE"

# Create a backup directory
BACKUP_DIR="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/ANM555975/backups"
mkdir -p "$BACKUP_DIR"

# Run the flip script with axis=2 (X axis, typically the shortest)
# This will create a backup and modify the file in place
echo "Running flip_and_save_h5.py to flip the h5 file along the shortest axis (X axis)..."

python flip_and_save_h5.py \
    --input_file "$H5_FILE" \
    --axis 2 \
    --datasets "Data" "Data_2_2_2" "Data_4_4_4" \
    --in_place \
    --backup

# Check the exit status
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Successfully flipped h5 file!"
    echo "A backup was created at: ${H5_FILE}.backup"
else
    echo "Failed to flip h5 file with exit code $STATUS"
fi