#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Registration Results Comparison Script

This script analyzes and compares results from different parameter sets in the 3D brain
registration process. It generates comparison visualizations and creates a baseline template
from unregistered brains to evaluate registration quality.

Usage:
    python compare_registration_results.py --results_dir /path/to/registration_results --data_dir /path/to/data

Author: Claude
"""

# import os  # Not needed
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import json
from pathlib import Path
from skimage import transform
# Exposure module removed as it's not available
import pandas as pd
import seaborn as sns
from tqdm import tqdm
# import time  # Not needed
# Import only what we need
from collections import defaultdict
from numpy_json_encoder import NumpyEncoder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Registration Results Comparison')
    
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to the directory containing registration results')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the directory containing original brain images')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Path to the output directory for comparison results (default: results_dir/comparison)')
    parser.add_argument('--downscale', type=float, default=0.1,
                       help='Factor to downscale images for faster processing (default: 0.1)')
    parser.add_argument('--create_baseline', action='store_true',
                       help='Create a baseline template from unregistered brains')
    
    return parser.parse_args()

def find_brain_images(data_dir):
    """Find all ch0.tif files in the data directory
    
    Args:
        data_dir (Path): Path to the data directory
        
    Returns:
        dict: Dictionary mapping animal IDs to file paths
    """
    brain_images = {}
    
    # List all animal directories
    for animal_dir in data_dir.glob('ANM*'):
        if animal_dir.is_dir():
            # Check if itk/ch0.tif exists
            ch0_file = animal_dir / 'itk' / 'ch0.tif'
            if ch0_file.exists():
                animal_id = animal_dir.name
                brain_images[animal_id] = ch0_file
    
    return brain_images

def load_and_preprocess_image(file_path, downscale_factor=0.1):
    """Load and preprocess a 3D brain image
    
    Args:
        file_path (Path): Path to the image file
        downscale_factor (float): Factor to downscale the image
        
    Returns:
        np.ndarray: Preprocessed image (3D volume)
    """
    # Load image
    print(f"Loading {file_path}...")
    img = tifffile.imread(file_path)
    
    # Get image info
    print(f"  Original shape: {img.shape}, dtype: {img.dtype}")
    
    # Ensure we have a 3D image
    if len(img.shape) != 3:
        raise ValueError(f"Expected a 3D image, got shape {img.shape}")

    # Rescale to 0-1 float
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    
    # Downscale if needed (this is crucial for 3D registration to be computationally feasible)
    if downscale_factor < 1.0:
        # Calculate new shape, preserving 3D structure
        new_shape = (img.shape[0], 
                     int(img.shape[1] * downscale_factor), 
                     int(img.shape[2] * downscale_factor))
        img = transform.resize(img, new_shape, anti_aliasing=True, preserve_range=True)
        print(f"  Downscaled to: {img.shape}")
    
    return img

def create_baseline_template(brain_images, downscale_factor=0.1, output_dir=None):
    """Create a baseline template from unregistered brain images
    
    Args:
        brain_images (dict): Dictionary mapping animal IDs to file paths
        downscale_factor (float): Factor to downscale images
        output_dir (Path): Output directory for saving results
        
    Returns:
        tuple: (baseline template, unregistered images, metrics)
    """
    print("Creating baseline template from unregistered brains...")
    
    # Load and preprocess images
    unregistered_images = {}
    for animal_id, file_path in tqdm(brain_images.items(), desc="Loading images"):
        unregistered_images[animal_id] = load_and_preprocess_image(file_path, downscale_factor)
    
    # Calculate average template
    print("Calculating average template...")
    image_arrays = list(unregistered_images.values())
    image_stack = np.stack(image_arrays)
    baseline_template = np.mean(image_stack, axis=0)
    
    # Calculate metrics
    metrics = calculate_baseline_metrics(unregistered_images, baseline_template)
    
    # Save results if output_dir is provided
    if output_dir:
        baseline_dir = output_dir / 'baseline'
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        # Save template
        tifffile.imwrite(baseline_dir / 'baseline_template.tif', baseline_template)
        
        # Save metrics
        with open(baseline_dir / 'baseline_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
        
        # Create visualizations
        create_baseline_visualizations(unregistered_images, baseline_template, baseline_dir)
    
    return baseline_template, unregistered_images, metrics

def calculate_baseline_metrics(unregistered_images, baseline_template):
    """Calculate metrics for baseline template
    
    Args:
        unregistered_images (dict): Dictionary of unregistered images
        baseline_template (np.ndarray): Baseline template
        
    Returns:
        dict: Dictionary of metrics
    """
    # Extract the numpy array data
    image_arrays = list(unregistered_images.values())
    
    # Calculate metrics
    metrics = {}
    
    # Mean squared difference between each image and the template
    mse_values = []
    for img in image_arrays:
        mse = float(np.mean((img - baseline_template) ** 2))
        mse_values.append(mse)
    
    metrics['mean_mse'] = float(np.mean(mse_values))
    metrics['std_mse'] = float(np.std(mse_values))
    
    # Calculate cross-correlation between each pair of images
    cc_values = []
    n_images = len(image_arrays)
    
    for i in range(n_images):
        for j in range(i+1, n_images):
            # Flatten images
            img1_flat = image_arrays[i].flatten()
            img2_flat = image_arrays[j].flatten()
            
            # Calculate correlation coefficient
            cc = float(np.corrcoef(img1_flat, img2_flat)[0, 1])
            cc_values.append(cc)
    
    metrics['mean_cc'] = float(np.mean(cc_values))
    metrics['std_cc'] = float(np.std(cc_values))
    
    # Record all individual MSE and CC values for detailed analysis
    metrics['individual_mse'] = [float(mse) for mse in mse_values]
    metrics['individual_cc'] = [float(cc) for cc in cc_values]
    
    return metrics

def create_baseline_visualizations(unregistered_images, baseline_template, output_dir):
    """Create visualizations for baseline template
    
    Args:
        unregistered_images (dict): Dictionary of unregistered images
        baseline_template (np.ndarray): Baseline template
        output_dir (Path): Output directory
    """
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Select slices to visualize
    first_img = list(unregistered_images.values())[0]
    slice_indices = [0, first_img.shape[0]//4, first_img.shape[0]//2, 
                     3*first_img.shape[0]//4, first_img.shape[0]-1]
    
    # Save template slices
    for slice_idx in slice_indices:
        if slice_idx >= baseline_template.shape[0]:
            continue
            
        # Create figure for this slice
        plt.figure(figsize=(10, 8))
        plt.imshow(baseline_template[slice_idx], cmap='gray')
        plt.title(f"Baseline Template - Slice {slice_idx}")
        plt.colorbar()
        plt.savefig(viz_dir / f"baseline_template_slice_{slice_idx}.png", dpi=150)
        plt.close()
    
    # Create a montage of the template
    montage_slices = np.linspace(0, baseline_template.shape[0]-1, min(9, baseline_template.shape[0])).astype(int)
    plt.figure(figsize=(15, 15))
    for i, slice_idx in enumerate(montage_slices):
        plt.subplot(3, 3, i+1)
        plt.imshow(baseline_template[slice_idx], cmap='gray')
        plt.title(f"Baseline Slice {slice_idx}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(viz_dir / "baseline_template_montage.png", dpi=150)
    plt.close()
    
    # Visualize differences between each image and the template
    middle_slice = baseline_template.shape[0] // 2
    plt.figure(figsize=(15, 5 * len(unregistered_images)))
    
    for i, (animal_id, img) in enumerate(unregistered_images.items()):
        # Original image
        plt.subplot(len(unregistered_images), 3, i*3+1)
        plt.imshow(img[middle_slice], cmap='gray')
        plt.title(f"{animal_id} - Unregistered")
        plt.axis('off')
        
        # Template
        plt.subplot(len(unregistered_images), 3, i*3+2)
        plt.imshow(baseline_template[middle_slice], cmap='gray')
        plt.title(f"Baseline Template")
        plt.axis('off')
        
        # Difference
        plt.subplot(len(unregistered_images), 3, i*3+3)
        diff = np.abs(img[middle_slice] - baseline_template[middle_slice])
        plt.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        plt.title(f"Difference")
        plt.axis('off')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(viz_dir / f"baseline_differences_slice_{middle_slice}.png", dpi=150)
    plt.close()

def collect_registration_results(results_dir):
    """Collect registration results from all parameter sets
    
    Args:
        results_dir (Path): Path to the results directory
        
    Returns:
        dict: Dictionary of results
    """
    print(f"Collecting registration results from {results_dir}...")
    
    # If overall results not available, collect individual results
    results = {'results': []}
    
    # Find all parameter directories
    param_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('grid')]
    print(f"Found {len(param_dirs)} parameter directories")
    
    for param_dir in tqdm(param_dirs, desc="Loading parameter results"):
        result_path = param_dir / 'results.json'
        if result_path.exists():
            with open(result_path, 'r') as f:
                try:
                    param_result = json.load(f)
                    # Extract parameter information from directory name
                    dir_name = param_dir.name
                    parts = dir_name.split('_')
                    
                    # Extract grid, scale, and speed values from directory name
                    grid_factor = parts[0].replace('grid', '') if len(parts) > 0 else ''
                    scale_factor = parts[1].replace('scale', '') if len(parts) > 1 else ''
                    speed_factor = parts[2].replace('speed', '') if len(parts) > 2 else ''
                    
                    # Add parameters to result if not already present
                    if 'parameters' not in param_result:
                        param_result['parameters'] = {
                            'grid_sampling_factor': int(grid_factor) if grid_factor.isdigit() else grid_factor,
                            'scale_sampling': int(scale_factor) if scale_factor.isdigit() else scale_factor,
                            'speed_factor': int(speed_factor) if speed_factor.isdigit() else speed_factor
                        }
                    
                    # Add output directory to result
                    param_result['output_dir'] = str(param_dir)
                    results['results'].append(param_result)
                except json.JSONDecodeError:
                    print(f"Error loading results from {param_dir}, skipping")
                except Exception as e:
                    print(f"Error processing {param_dir}: {str(e)}, skipping")
    
    print(f"Collected results from {len(results['results'])} parameter sets")
    return results

def analyze_results(results, baseline_metrics=None):
    """Analyze registration results and prepare comparison data
    
    Args:
        results (dict): Dictionary of registration results
        baseline_metrics (dict): Baseline metrics for comparison
        
    Returns:
        tuple: (DataFrame of metrics, top parameter sets)
    """
    print("Analyzing registration results...")
    
    # Extract metrics from results
    data = []
    
    for result in results['results']:
        if 'parameters' in result:
            params = result['parameters']
            # Look for metrics in either 'quality_metrics' or directly in result
            metrics = result.get('quality_metrics', result)
            
            # Skip if we can't find necessary metrics
            if not all(key in metrics for key in ['mean_cc', 'mean_mse']):
                print(f"Skipping result from {result.get('output_dir', 'unknown')} - missing metrics")
                continue
            
            row = {
                'grid_sampling_factor': params.get('grid_sampling_factor', 'unknown'),
                'scale_sampling': params.get('scale_sampling', 'unknown'),
                'speed_factor': params.get('speed_factor', 'unknown'),
                'mean_mse': metrics.get('mean_mse', 0),
                'std_mse': metrics.get('std_mse', 0),
                'mean_cc': metrics.get('mean_cc', 0),
                'std_cc': metrics.get('std_cc', 0),
                'elapsed_time': result.get('elapsed_time', 0),
                'output_dir': result.get('output_dir', '')
            }
            data.append(row)
    
    # Check if we have any valid results
    if not data:
        print("No valid results found in the data!")
        if baseline_metrics:
            print("Only baseline metrics available")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add baseline metrics if available
    if baseline_metrics:
        baseline_row = {
            'grid_sampling_factor': 'baseline',
            'scale_sampling': 'baseline',
            'speed_factor': 'baseline',
            'mean_mse': baseline_metrics['mean_mse'],
            'std_mse': baseline_metrics['std_mse'],
            'mean_cc': baseline_metrics['mean_cc'],
            'std_cc': baseline_metrics['std_cc'],
            'elapsed_time': 0,
            'output_dir': 'baseline'
        }
        df = pd.concat([df, pd.DataFrame([baseline_row])], ignore_index=True)
    
    # Sort by mean_cc (higher is better)
    df_sorted = df.sort_values('mean_cc', ascending=False)
    
    # Get top parameter sets (at most 5, but could be fewer if there are fewer results)
    top_params = df_sorted.head(min(5, len(df_sorted)))
    
    return df, top_params

def create_comparison_visualizations(df, top_params, output_dir):
    """Create visualization comparing different parameter sets
    
    Args:
        df (DataFrame): DataFrame of metrics
        top_params (DataFrame): Top parameter sets
        output_dir (Path): Output directory
    """
    print("Creating comparison visualizations...")
    
    # Create output directory
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out baseline for some plots
    df_no_baseline = df[df['grid_sampling_factor'] != 'baseline'].copy()
    
    # 1. Parameter Space Heatmap - Mean Cross-Correlation
    if len(df_no_baseline) > 1:
        plt.figure(figsize=(12, 10))
        heatmap_data = df_no_baseline.pivot_table(
            index='grid_sampling_factor', 
            columns=['scale_sampling', 'speed_factor'], 
            values='mean_cc'
        )
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.4f')
        plt.title('Mean Cross-Correlation by Parameter Combination')
        plt.tight_layout()
        plt.savefig(viz_dir / 'parameter_heatmap_cc.png', dpi=150)
        plt.close()
        
        # 2. Parameter Space Heatmap - Mean MSE
        plt.figure(figsize=(12, 10))
        heatmap_data = df_no_baseline.pivot_table(
            index='grid_sampling_factor', 
            columns=['scale_sampling', 'speed_factor'], 
            values='mean_mse'
        )
        sns.heatmap(heatmap_data, annot=True, cmap='rocket_r', fmt='.4f')
        plt.title('Mean MSE by Parameter Combination')
        plt.tight_layout()
        plt.savefig(viz_dir / 'parameter_heatmap_mse.png', dpi=150)
        plt.close()
    
    # 3. Bar chart comparing top parameter sets with baseline
    plt.figure(figsize=(12, 8))
    top_with_baseline = pd.concat([
        top_params, 
        df[df['grid_sampling_factor'] == 'baseline']
    ], ignore_index=True)
    
    # Create parameter set labels
    top_with_baseline['param_label'] = top_with_baseline.apply(
        lambda x: f"grid={x['grid_sampling_factor']}, scale={x['scale_sampling']}, speed={x['speed_factor']}" 
        if x['grid_sampling_factor'] != 'baseline' else 'Baseline',
        axis=1
    )
    
    # Plot CC comparison
    plt.subplot(2, 1, 1)
    sns.barplot(x='param_label', y='mean_cc', data=top_with_baseline)
    plt.title('Mean Cross-Correlation Comparison')
    plt.ylabel('Mean CC')
    plt.xticks(rotation=45, ha='right')
    
    # Plot MSE comparison
    plt.subplot(2, 1, 2)
    sns.barplot(x='param_label', y='mean_mse', data=top_with_baseline)
    plt.title('Mean MSE Comparison')
    plt.ylabel('Mean MSE')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'top_params_comparison.png', dpi=150)
    plt.close()
    
    # 4. Scatter plot of CC vs MSE
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df['mean_cc'], 
        df['mean_mse'], 
        c=df['elapsed_time'], 
        cmap='viridis', 
        s=100, 
        alpha=0.7
    )
    
    # Highlight baseline if present
    baseline = df[df['grid_sampling_factor'] == 'baseline']
    if not baseline.empty:
        plt.scatter(
            baseline['mean_cc'], 
            baseline['mean_mse'], 
            c='red', 
            s=150, 
            marker='*', 
            label='Baseline'
        )
    
    # Highlight top parameter set
    top = df.iloc[df['mean_cc'].idxmax()]
    plt.scatter(
        top['mean_cc'], 
        top['mean_mse'], 
        c='lime', 
        s=150, 
        marker='o', 
        edgecolors='black', 
        label='Best Parameter Set'
    )
    
    plt.colorbar(scatter, label='Elapsed Time (s)')
    plt.xlabel('Mean Cross-Correlation (higher is better)')
    plt.ylabel('Mean MSE (lower is better)')
    plt.title('Registration Quality Trade-off')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / 'cc_vs_mse_scatter.png', dpi=150)
    plt.close()
    
    # 5. Save metrics table as CSV
    df.to_csv(output_dir / 'metrics_comparison.csv', index=False)
    top_params.to_csv(output_dir / 'top_parameter_sets.csv', index=False)

def compare_templates(results_dir, baseline_template, output_dir):
    """Compare templates from different parameter sets
    
    Args:
        results_dir (Path): Path to the results directory
        baseline_template (np.ndarray): Baseline template
        output_dir (Path): Output directory
    """
    print("Comparing templates from different parameter sets...")
    
    # Find all template files
    template_paths = []
    for param_dir in results_dir.iterdir():
        if param_dir.is_dir() and param_dir.name.startswith('grid'):
            template_path = param_dir / 'template.tif'
            if template_path.exists():
                template_paths.append((param_dir.name, template_path))
    
    # Load templates
    templates = {}
    for param_name, path in tqdm(template_paths, desc="Loading templates"):
        templates[param_name] = tifffile.imread(path)
    
    # Create visualization directory
    viz_dir = output_dir / 'template_comparison'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Select middle slice for comparison
    middle_slice = baseline_template.shape[0] // 2
    
    # Create a grid of templates
    num_templates = len(templates) + 1  # +1 for baseline
    cols = min(3, num_templates)
    rows = (num_templates + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    
    # Add baseline template
    plt.subplot(rows, cols, 1)
    plt.imshow(baseline_template[middle_slice], cmap='gray')
    plt.title('Baseline Template')
    plt.axis('off')
    
    # Add registered templates
    for i, (param_name, template) in enumerate(templates.items(), start=2):
        plt.subplot(rows, cols, i)
        plt.imshow(template[middle_slice], cmap='gray')
        plt.title(f'{param_name}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(viz_dir / f'template_comparison_slice_{middle_slice}.png', dpi=150)
    plt.close()
    
    # Create difference maps between best template and baseline
    # First, find the best parameter set
    with open(output_dir / 'top_parameter_sets.csv', 'r') as f:
        top_params = pd.read_csv(f)
    
    best_param_dir = top_params.iloc[0]['output_dir']
    best_param_name = Path(best_param_dir).name
    
    if best_param_name in templates:
        best_template = templates[best_param_name]
        
        # Create difference map for multiple slices
        slice_indices = [0, baseline_template.shape[0]//4, baseline_template.shape[0]//2, 
                         3*baseline_template.shape[0]//4, baseline_template.shape[0]-1]
        
        for slice_idx in slice_indices:
            if slice_idx >= baseline_template.shape[0]:
                continue
                
            plt.figure(figsize=(15, 5))
            
            # Baseline template
            plt.subplot(1, 3, 1)
            plt.imshow(baseline_template[slice_idx], cmap='gray')
            plt.title('Baseline Template')
            plt.axis('off')
            
            # Best template
            plt.subplot(1, 3, 2)
            plt.imshow(best_template[slice_idx], cmap='gray')
            plt.title(f'Best Template ({best_param_name})')
            plt.axis('off')
            
            # Difference
            plt.subplot(1, 3, 3)
            diff = np.abs(baseline_template[slice_idx] - best_template[slice_idx])
            plt.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            plt.title('Difference')
            plt.axis('off')
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(viz_dir / f'baseline_vs_best_slice_{slice_idx}.png', dpi=150)
            plt.close()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Convert paths to Path objects
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / 'comparison'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist!")
        return 1
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist!")
        return 1
    
    # Find brain images
    brain_images = find_brain_images(data_dir)
    if not brain_images:
        print("Error: No brain images found!")
        return 1
    
    print(f"Found {len(brain_images)} brain images:")
    for animal_id, file_path in brain_images.items():
        print(f"  {animal_id}: {file_path}")
    
    # Set up baseline paths
    baseline_template = None
    baseline_metrics = None
    baseline_dir = output_dir / 'baseline'
    baseline_template_path = baseline_dir / 'baseline_template.tif'
    baseline_metrics_path = baseline_dir / 'baseline_metrics.json'
    
    # Check if baseline already exists
    baseline_exists = baseline_dir.exists() and baseline_template_path.exists() and baseline_metrics_path.exists()
    
    # Case 1: User wants to create baseline and it doesn't exist
    if args.create_baseline and not baseline_exists:
        print("Creating new baseline template...")
        baseline_template, unregistered_images, baseline_metrics = create_baseline_template(
            brain_images, args.downscale, output_dir
        )
    # Case 2: User wants to create baseline but it already exists
    elif args.create_baseline and baseline_exists:
        print(f"Baseline already exists at {baseline_dir}, skipping creation")
        # Load existing baseline template and metrics
        baseline_template = tifffile.imread(baseline_template_path)
        with open(baseline_metrics_path, 'r') as f:
            baseline_metrics = json.load(f)
    # Case 3: User didn't request baseline creation but it exists already
    elif not args.create_baseline and baseline_exists:
        print(f"Loading existing baseline from {baseline_dir}")
        # Load existing baseline template and metrics
        baseline_template = tifffile.imread(baseline_template_path)
        with open(baseline_metrics_path, 'r') as f:
            baseline_metrics = json.load(f)
    # Case 4: User didn't request baseline and it doesn't exist - do nothing
    
    # Collect registration results
    results = collect_registration_results(results_dir)
    
    # Analyze results
    df, top_params = analyze_results(results, baseline_metrics)
    
    # Create comparison visualizations
    create_comparison_visualizations(df, top_params, output_dir)
    
    # Compare templates
    if baseline_template is not None:
        compare_templates(results_dir, baseline_template, output_dir)
    
    print(f"\nComparison completed successfully! Results saved to: {output_dir}")
    
    # Print top parameter sets
    print("\nTop 5 parameter sets by cross-correlation:")
    for i, row in top_params.iterrows():
        print(f"{i+1}. grid={row['grid_sampling_factor']}, "
              f"scale={row['scale_sampling']}, "
              f"speed={row['speed_factor']}: "
              f"mean_cc={row['mean_cc']:.4f}, "
              f"mean_mse={row['mean_mse']:.4f}, "
              f"time={row['elapsed_time']:.2f}s")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())