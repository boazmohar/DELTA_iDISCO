import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import regionprops_table

# Compute region statistics
def compute_region_stats(files, output_dir, annotation_np):
    image_list = [io.imread(os.path.join(output_dir, name + '.tif')) for name in files.keys()]
    multichannel_image = np.stack(image_list, axis=-1)
    
    props = regionprops_table(annotation_np, intensity_image=multichannel_image, properties=['label', 'mean_intensity', 'area'])
    df_stats = pd.DataFrame(props)
    df_stats.rename(columns={
        'mean_intensity-0': 'Mean_ch0',
        'mean_intensity-1': 'Mean_ch1',
        'mean_intensity-2': 'Mean_ch2',
        'area': 'N',
        'label': 'Region'
    }, inplace=True)
    df_stats.to_csv(os.path.join(output_dir, 'region_stats.csv'), index=False)
    print(f"Region statistics saved to {os.path.join(output_dir, 'region_stats.csv')}")
