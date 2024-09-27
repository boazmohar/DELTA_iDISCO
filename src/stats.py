import os
import numpy as np
import pandas as pd
from skimage import io
from multiprocessing import Pool, shared_memory
from tqdm import tqdm

# General function to compute stats for a single label
def compute_stats_for_label(args):
    label, annotation_np_shape, intensity_image_shape, annotation_np_dtype, intensity_image_dtype, annotation_np_name, intensity_image_name, funcs = args
    
    # Access shared memory for annotation_np and intensity_image
    annotation_shm = shared_memory.SharedMemory(name=annotation_np_name)
    annotation_np = np.ndarray(annotation_np_shape, dtype=annotation_np_dtype, buffer=annotation_shm.buf)

    intensity_shm = shared_memory.SharedMemory(name=intensity_image_name)
    intensity_image = np.ndarray(intensity_image_shape, dtype=intensity_image_dtype, buffer=intensity_shm.buf)

    # Create a mask for the current label (view of the array)
    mask = annotation_np == label

    # For each channel, apply each function in the funcs list
    results_per_channel = []
    for ch in range(intensity_image.shape[-1]):
        masked_intensity = intensity_image[..., ch][mask]  # Intensity values for this channel and label
        channel_results = {}
        for func in funcs:
            func_name = func.__name__
            if masked_intensity.size > 0:
                channel_results[func_name] = func(masked_intensity)
            else:
                channel_results[func_name] = 0
        results_per_channel.append(channel_results)

    area = mask.sum()  # Number of pixels for the label (same across all channels)

    return label, results_per_channel, area

# Function to process the labels in parallel using shared memory
def parallel_process_labels(annotation_np, intensity_image, funcs, num_cores=8):
    # Get all unique labels, excluding 0 (background)
    unique_labels = np.unique(annotation_np)
    unique_labels = unique_labels[unique_labels != 0]

    # Create shared memory objects for annotation_np and intensity_image
    annotation_shm = shared_memory.SharedMemory(create=True, size=annotation_np.nbytes)
    annotation_np_shared = np.ndarray(annotation_np.shape, dtype=annotation_np.dtype, buffer=annotation_shm.buf)
    np.copyto(annotation_np_shared, annotation_np)  # Copy data into shared memory

    intensity_shm = shared_memory.SharedMemory(create=True, size=intensity_image.nbytes)
    intensity_image_shared = np.ndarray(intensity_image.shape, dtype=intensity_image.dtype, buffer=intensity_shm.buf)
    np.copyto(intensity_image_shared, intensity_image)  # Copy data into shared memory

    # Prepare arguments for each label (passing shapes, dtypes, shared memory names, and functions list)
    args = [
        (
            label,
            annotation_np.shape,
            intensity_image.shape,
            annotation_np.dtype,
            intensity_image.dtype,
            annotation_shm.name,
            intensity_shm.name,
            funcs
        )
        for label in unique_labels
    ]

    # Create a Pool with the specified number of cores
    with Pool(processes=num_cores) as pool:
        # Use tqdm to show the progress bar as tasks complete
        results = list(tqdm(pool.imap_unordered(compute_stats_for_label, args), total=len(args), desc="Processing labels", unit="label"))

    # Cleanup shared memory
    annotation_shm.close()
    annotation_shm.unlink()
    intensity_shm.close()
    intensity_shm.unlink()

    # Convert the results to a list of dictionaries
    stats = []
    for label, results_per_channel, area in results:
        entry = {'Region': label, 'N': area}
        for ch, channel_results in enumerate(results_per_channel):
            for func_name, result in channel_results.items():
                entry[f'{func_name}_ch{ch}'] = result
        stats.append(entry)
    
    return stats

def compute_region_stats(files, output_dir, annotation_np, funcs, num_cores):
    # open reaw files (3 ch)
    image_list = [io.imread(os.path.join(output_dir, name + '.tif')) for name in files.keys()]
    multichannel_image = np.stack(image_list, axis=-1)
    print(funcs)
    # get stats
    stats = parallel_process_labels(annotation_np, multichannel_image, funcs, num_cores=num_cores)
    
    # Convert to DataFrame 
    df_stats = pd.DataFrame(stats)
    df_stats = df_stats.sort_values(by='Region')
    df_stats.to_csv(os.path.join(output_dir, 'region_stats.csv'), index=False)
    print(f"Region statistics saved to {os.path.join(output_dir, 'region_stats.csv')}")
    return df_stats
