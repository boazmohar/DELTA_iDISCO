#!/bin/bash

# submit_jobs.sh
# Script to submit registration and analysis jobs for each animal using bsub

# Default paths for the base directory, atlas (fx), parameter files, and annotation volume
BASE_DIR='/nrs/spruston/Boaz/I2/2024-09-19_iDISCO_CalibrationBrains'
FX='/nrs/spruston/Boaz/I2/atlas10_hemi.tif'
PARAM_FILES_DIR='/nrs/spruston/Boaz/I2/itk'
ANNOTATION_NP='/nrs/spruston/Boaz/I2/annotatin10_hemi.tif'

# Path to the main Python script
PYTHON_SCRIPT='/path/to/DELTA_iDISCO/src/main.py'

# List of animals to process (replace with your actual animal IDs)
animals=(
    "ANM550749_left_JF552"
    "ANM550751_left_JF673"
    "ANM551089_left_JF673"
    # Add more animal IDs here
)

# Create logs directory if it doesn't exist
mkdir -p ../logs

# Loop over each animal and submit a job
for animal in "${animals[@]}"; do
    # Construct the job name (optional)
    job_name="job_${animal}"

    # Submit the job using bsub
    bsub -J "$job_name" -n 4 -W 4:00  \
         -o ../logs/output_${animal}.log \
         -e ../logs/error_${animal}.log \
         "python $PYTHON_SCRIPT --animal $animal --base_dir $BASE_DIR --fx $FX --param_files_dir $PARAM_FILES_DIR --annotation_np $ANNOTATION_NP"
done
