# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

### Conda Environment
```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate delta-idisco
```

### Pip Installation
```bash
# Install dependencies via pip
pip install -r requirements.txt
```

## Key Directories and Files

- `/src`: Core library code for registration, analysis, and stats
- `/scripts`: Utility scripts for batch processing
- `/paramaters`: Configuration files for registration algorithms
- `/logs`: Output and error logs from processing

## Required External Files

The following files are required from external sources:
- Atlas image: `/nearline/spruston/Boaz/DELTA/I2/atlas10_hemi.tif`
- Annotation volume: `/nearline/spruston/Boaz/DELTA/I2/annotation_10_hemi.nii`
- Parameter files directory: `/nearline/spruston/Boaz/DELTA/I2/itk`

## Common Commands

### Processing a Single Animal
```bash
python src/main.py --animal ANM550749_left_JF552 \
  --base_dir /nearline/spruston/Boaz/DELTA/I2/2024-09-19_iDISCO_CalibrationBrains \
  --fx /nearline/spruston/Boaz/DELTA/I2/atlas10_hemi.tif \
  --param_files_dir /nearline/spruston/Boaz/DELTA/I2/itk \
  --annotation_np /nearline/spruston/Boaz/DELTA/I2/annotation_10_hemi.nii
```

### Batch Processing with LSF
```bash
# Submit batch jobs for multiple animals
bash scripts/submit_jobs.sh
```

### Running 3D Brain Registration with Parameter Scanning
```bash
python register_brains_3d_parallel.py \
  --data_dir /path/to/data_directory \
  --output_dir /path/to/output_directory \
  --grid_sampling_factors 0.75 1 \
  --scale_samplings 45 65 100 200 \
  --speed_factors 6 10 20 35 \
  --final_scales 1.0 \
  --scale_levels 4 6 8 10 \
  --final_grid_samplings 1.0 2.0 \
  --downscale 0.2 \
  --save_visualizations
```

### Running Optimal Registration Plan
```bash
python optimal_registration_plan.py \
  --param_scan_dir /path/to/param_scan_dir \
  --param_scan_results /path/to/param_scan_results.json \
  --data_dir /path/to/data_dir \
  --output_dir /path/to/output_dir \
  --ccf_atlas /path/to/atlas10_hemi.tif \
  --annotation_path /path/to/annotation_10_hemi.nii \
  --param_files_dir /path/to/itk
```

## Architecture Overview

DELTA_iDISCO is a pipeline for processing and analyzing imaging data from iDISCO experiments, specifically designed for registering brain images to the Allen Brain Atlas and extracting regional statistics.

Key components:
1. **Registration**: Aligns brain images to a common atlas using ITK Elastix
   - Uses a multi-step registration approach with affine and B-spline transformations
   - `register_brains_3d_parallel.py` for parallelized parameter scanning
   - `registration.py` for core registration functionality

2. **Analysis**: Extracts statistics from registered images
   - `stats.py` for regional statistics computation
   - `analysis.py` for higher-level data analysis

3. **Main Workflow**:
   - Register and transform brain images to align with atlas
   - Compute region-based statistics using annotation volumes
   - Generate visualizations and quality metrics

4. **Parameter Optimization**:
   - Parameter scanning through parallel processing
   - Quality metrics to evaluate registration performance

The pipeline assumes a specific directory structure from MuVi light-sheet microscopy system output, with each animal having channel files (ch0.tif, ch1.tif, etc.) in an 'itk' subdirectory.