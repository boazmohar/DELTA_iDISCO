import os
import pandas as pd
import pickle
import time

from datetime import datetime, timedelta
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

def get_sessions_with_offset(directory, start_offset, num_days):
    # List to store the filenames and their dates
    sessions = []
    
    # Loop through the files in the given directory
    for filename in os.listdir(directory):
        if filename.startswith("Log") and filename.endswith(".pkl"):
            # Extract the date and session number from the filename
            parts = filename.split()
            date_str = parts[2]
            session_date = datetime.strptime(date_str, "%Y-%m-%d")
            session_number = parts[-1].split('.')[0]
            
            # Append the filename, date, and session number to the list
            sessions.append((filename, session_date, session_number))
    
    # Sort the sessions by date
    sessions.sort(key=lambda x: x[1])
    
    # Get unique dates
    unique_dates = sorted(set(date for _, date, _ in sessions))
    
    # Determine the start and end dates based on the offset and number of days
    if start_offset >= len(unique_dates):
        return {}  # If the offset is out of range, return an empty dict
    
    start_index = max(0, len(unique_dates) - start_offset - num_days)
    end_index = len(unique_dates) - start_offset
    
    start_date = unique_dates[start_index]
    end_date = unique_dates[end_index - 1] + timedelta(days=1)  # Include the end date
    
    # Filter sessions within the date range and store in a dictionary
    filtered_sessions = {}
    for filename, date, session_number in sessions:
        if start_date <= date < end_date:
            filtered_sessions[filename] = {
                'date': date.strftime("%Y-%m-%d") + '-' + str(session_number)
            }
    
    return filtered_sessions

def load_sessions_data(sessions, directory):
    # Dictionary to store the loaded data
    loaded_data = {}
    
    # Loop through the sessions dictionary
    for filename, session_name in sessions.items():
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Load the pickle file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # Store the loaded data in the dictionary
        loaded_data[session_name['date']] = data.task
    
    return loaded_data