python compare_registration_results.py \
    --results_dir /nrs/spruston/Boaz/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan \
    --data_dir /nrs/spruston/Boaz/I2/20250111_IDISCO_MouseCity3/ \
    --create_baseline \
    --downscale 0.2

 python register_brains_3d_parallel.py --data_dir /nrs/spruston/Boaz/I2/20250111_IDISCO_MouseCity3/ \
  --output_dir /nrs/spruston/Boaz/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan \
   --grid_sampling_factors 1 --scale_samplings 45 55 65 75 100 --speed_factors 4 6 8 10 20

    python register_brains_3d_parallel.py --data_dir /nrs/spruston/Boaz/I2/20250111_IDISCO_MouseCity3/ \
  --output_dir /nrs/spruston/Boaz/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan \
   --grid_sampling_factors 1 --scale_samplings 200 100 --speed_factors 25 35 --downscale 0.2

   python optimal_registration_plan.py \
    --param_scan_dir /nrs/spruston/Boaz/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan \
    --param_scan_results /nrs/spruston/Boaz/I2/20250111_IDISCO_MouseCity3/registration_results_param_scan/parameter_scan_results.json \
    --data_dir /nrs/spruston/Boaz/I2/20250111_IDISCO_MouseCity3 \
    --output_dir /nrs/spruston/Boaz/I2/20250111_IDISCO_MouseCity3/optimal_registration_results \
    --ccf_atlas /nrs/spruston/Boaz/I2/atlas10_hemi.tif \
    --annotation_path /nrs/spruston/Boaz/I2/annotation_10_hemi.nii \
    --param_files_dir /nrs/spruston/Boaz/I2/itk