python compare_registration_results.py \
    --results_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan \
    --data_dir /nearline/spruston/Boaz/I2/20250111_IDISCO_MouseCity3/ \
    --create_baseline \
    --downscale 0.2
 
 python register_brains_3d_parallel.py --data_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/ \
  --output_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan \
   --grid_sampling_factors 1 --scale_samplings 45 55 65 75 100 --speed_factors 4 6 8 10 20

python register_brains_3d_parallel.py --data_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/ \
  --output_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan \
   --grid_sampling_factors 1 --scale_samplings 200 100 --speed_factors 25 35 --downscale 0.2

python optimal_registration_plan.py \
    --param_scan_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan \
    --param_scan_results /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan/parameter_scan_results.json \
    --data_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3 \
    --output_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/optimal_registration_results \
    --ccf_atlas /nearline/spruston/Boaz/DELTA/I2/atlas10_hemi.tif \
    --annotation_path /nearline/spruston/Boaz/DELTA/I2/annotation_10_hemi.nii \
    --param_files_dir /nearline/spruston/Boaz/DELTA/I2/itk

python register_brains_3d_parallel.py \
    --data_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/ \
    --output_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan \
    --grid_sampling_factors 0.8 1 1.2\
    --scale_samplings  20 40 100 \
    --speed_factors 2 6 12 24 \
    --final_scales 1.0 \
    --scale_levels 2 4 8\
    --final_grid_samplings 5 10 20 \
    --downscale 0.2 \
    --save_visualizations \
    --max_processes 44 

python compare_registration_results.py \
    --results_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan \
    --data_dir /nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/ \
    --create_baseline \
    --downscale 0.2

DATA_DIR=
OUTPUT_DIR=

# Resolution options:
# 1 = Data (full resolution)
# 2 = Data_2_2_2 (2x downsampled in each dimension)
# 4 = Data_4_4_4 (4x downsampled in each dimension)
H5_RESOLUTION=2

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the registration
python register_brains_3d_parallel_h5.py \
  --data_dir "/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3" \
  --output_dir "/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/registration_results_h5" \
  --h5_resolution 4 \
  --threshold 100 \
  --grid_sampling_factors 0.75 1 \
  --scale_samplings 45 65 100 \
  --speed_factors 6 10 20 \
  --final_scales 1.0 \
  --scale_levels 4 6 8 \
  --final_grid_samplings 1.0 2.0 \
  --save_visualizations

echo "Registration complete. Results saved to $OUTPUT_DIR"