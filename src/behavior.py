import os
import pandas as pd
import pickle
import time
from vr2p.gimbl.parse import parse_gimbl_log
from utils import behav


def load_and_clean_alyssa_sheet(dirctory, sheet_name, file_name='Notes_Behavior.xlsx'):
    """Load and clean the sheet from Alyssa's excel file  (used for BM27-BM30)."""
    df = pd.read_excel(os.path.join(dirctory, file_name), sheet_name=sheet_name)

    # Drop rows columns that are entirely NaN
    df_cleaned = df.dropna(how='all', axis=1)
    df_cleaned = df_cleaned.dropna(how='all')
    
    if 'Headbar' in df_cleaned.columns:
        # Remove columns starting from 'Headbar' and after
        df_cleaned = df_cleaned.loc[:, :'Headbar']
    
    return df_cleaned


def load_and_clean_mafe_sheet(dirctory, file_name='BM24.csv'):
    """Load and clean the sheet from Mafe (used for BM24,26)."""
    df = pd.read_csv(os.path.join(dirctory, file_name))

    # Drop rows columns that are entirely NaN
    df_cleaned = df.dropna(how='all', axis=1)
    df_cleaned = df_cleaned.dropna(how='all')
    
    return df_cleaned


def pre_proccess(mouse_dir = "E:\\Unbiased\\GluA2\\Behavior data", mice_num=[24,26,27,28,29,30]):
    mice = [f'BM{mouse_num}' for mouse_num in mice_num]
    for mouse in mice:
        print(f'Starting mouse {mouse}.')
        t_start = time.time()
        json_data_dir = os.path.join(mouse_dir, mouse)
        print(json_data_dir)

        # Make directory for processed data (if it doesn't exist)
        processed_data_dir = os.path.join(mouse_dir, 'preprocess_pilot_v1', mouse)
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        
        # Get names of all raw data json files for mouse.
        json_fnames = [fname for fname in os.listdir(json_data_dir) if fname.endswith('.json')]
        num_saved = 0
        for json_fname in json_fnames:
            # Load + preprocess data.
            json_fpath = os.path.join(json_data_dir, json_fname)
            print(f'  Processing {json_fpath}...')
            try:
                log, vr = parse_gimbl_log(json_fpath, verbose=True)
            except:
                print(f'  Failed to parse {json_fpath}.')
                continue
                
            vr = behav.preprocess_vr(log, vr)

            # Save with pickle.
            fname_prefix = json_fname.split('.json')[0]
            processed_fpath = os.path.join(processed_data_dir, fname_prefix + '.pkl')
            with open(processed_fpath, 'wb') as file:
                pickle.dump(vr, file)
                num_saved += 1

            print(f'  Saved {num_saved} of {len(json_fnames)} (time elapsed = {time.time() - t_start} sec)')
        t_end = time.time()
        print(f'Time elapsed: {t_end - t_start}')

