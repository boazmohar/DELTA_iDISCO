import os
import re
import logging
import h5py
import numpy as np
from collections import defaultdict


def flip_image(image, axes):
    for axis in axes:
        image = np.flip(image, axis=axis)
    return image

# Function to read h5 files
def read_h5_image(file_path, dataset_name, th=100):
    with h5py.File(file_path, 'r') as h5_file:
        data = h5_file[dataset_name][:]
        data[data < th] = th
        return data

# Match h5 files to their channels
def match_h5_files_by_channels(base_dir):
    file_groups = defaultdict(lambda: {"ch0": None, "ch1": None, "ch2": None})
    for dirpath, dirnames, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith('.h5'):
                dir_name = os.path.basename(dirpath)
                full_path = os.path.join(dirpath, filename)
                if "uni_tp-0_ch-0" in filename:
                    file_groups[dir_name]["ch0"] = full_path
                elif "uni_tp-0_ch-1" in filename:
                    file_groups[dir_name]["ch1"] = full_path
                elif "uni_tp-0_ch-2" in filename:
                    file_groups[dir_name]["ch2"] = full_path
    return {key: value for key, value in file_groups.items() if all(value.values())}


def setup_logging(base_dir, animal):
    """
    Sets up logging to both a file in the base directory with the animal's name and the console.
    """
    log_file = os.path.join(base_dir, f'process_log_{animal}.txt')

    # Create a custom logger
    logger = logging.getLogger(animal)
    logger.setLevel(logging.INFO)
    
    # Check if handlers are already added (to prevent duplicate logs)
    if not logger.hasHandlers():
        # Create handlers
        f_handler = logging.FileHandler(log_file)  # Log to file
        c_handler = logging.StreamHandler()  # Log to console
        
        f_handler.setLevel(logging.INFO)
        c_handler.setLevel(logging.INFO)
        
        # Create formatters and add them to handlers
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        f_handler.setFormatter(log_format)
        c_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(f_handler)
        logger.addHandler(c_handler)
    
    logger.info("Logging initialized for animal processing.")
    
    return logger


def collect_region_stats_paths(base_path):
    """
    Collects the paths of all 'region_stats.csv' files from subdirectories of base_path.
    
    The function assumes that the 6-digit ANM number is part of the directory structure and extracts it.
    
    Parameters:
    - base_path: The base directory to search for 'region_stats.csv' files.
    
    Returns:
    - A dictionary where the keys are 6-digit ANM numbers and values are the paths to the 'region_stats.csv' files.
    """
    region_stats_paths = {}
    
    # Define a regex pattern to capture the 6-digit ANM number
    anm_pattern = re.compile(r"ANM(\d{6})")
    
    # Walk through all subdirectories of base_path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'region_stats.csv':
                # Check if the path contains an ANM number
                match = anm_pattern.search(root)
                if match:
                    anm_number = match.group(1)  # Extract the ANM number
                    full_path = os.path.join(root, file)
                    region_stats_paths[anm_number] = full_path
    
    return region_stats_paths
