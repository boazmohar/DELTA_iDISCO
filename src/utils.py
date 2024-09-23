import os
import h5py
from collections import defaultdict

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
