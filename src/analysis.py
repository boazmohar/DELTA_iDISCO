import pandas as pd


def load_region_data(csv_paths):
    region_data = {}
    for animal_id, path in csv_paths.items():
        # Load each CSV file into a DataFrame
        df = pd.read_csv(path)
        # Store DataFrame in the dictionary with the animal ID as the key
        region_data[animal_id] = df
    return region_data

def filter_metadata(metadata_df, filters):
    for column, values in filters.items():
        metadata_df = metadata_df[metadata_df[column].isin(values)]
    return metadata_df

def filter_region_columns(region_data, columns):
    filtered_region_data = {}
    for animal_id, df in region_data.items():
        # Filter region DataFrame by specified columns
        filtered_region_data[animal_id] = df[columns]
    return filtered_region_data

def merge_region_with_metadata(metadata_df, region_data, meta_columns=None):
    merged_dfs = []
    
    for animal_id, region_df in region_data.items():
        # Use .loc to safely assign 'AnimalID'
        region_df = region_df.copy()  # Create a copy to avoid warnings
        region_df.loc[:, 'AnimalID'] = int(animal_id)

        # Filter metadata to only one row per AnimalID
        filtered_metadata_df = metadata_df[metadata_df['AnimalID'] == int(animal_id)]
        
        # Filter metadata columns if meta_columns are provided
        if meta_columns:
            filtered_metadata_df = filtered_metadata_df[meta_columns]
        
        # Merge metadata with region data based on AnimalID
        merged_df = pd.merge(filtered_metadata_df, region_df, on='AnimalID', how='left')
        merged_dfs.append(merged_df)
    
    # Combine all animal data into one DataFrame
    unified_df = pd.concat(merged_dfs, ignore_index=True)
    return unified_df

def get_filtered(meta, filters, region_data, columns, meta_columns=['AnimalID']):
    filtered_metadata = filter_metadata(meta, filters)
    filtered_region_data = filter_region_columns(region_data, columns)
    unified_df = merge_region_with_metadata(filtered_metadata, filtered_region_data, meta_columns)
    return unified_df