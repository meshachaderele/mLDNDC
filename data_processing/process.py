import pandas as pd
pd.set_option('display.max_columns', None)
import os
import time
from tqdm import tqdm
import numpy as np
import glob

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def clean_data(CROP):

    df = pd.read_parquet(f"data/ml_ready_{CROP}_outputs.parquet")
    
    def process_df(df):
        # Track ID counts before processing
        initial_counts = df['id'].value_counts()
    
        df.loc[(df['n_org_amount'] == 0), ['n_org_type', 'n_org_replication', 'manu_daytoplanting_1',	'manu_daytoplanting_2',	'manu_daytoplanting_3',	'manu_amount_1',	'manu_amount_2',	'manu_amount_3']] = [None, 0, np.nan, np.nan, np.nan, 0,0,0]
        df.loc[(df['n_synthamount'] == 0), ['n_synth_type', 'n_synth_replication', 'fert_daytoplanting_1',	'fert_daytoplanting_2',	'fert_daytoplanting_3', 'fert_amount_1',	'fert_amount_2',	'fert_amount_3']] = [None, 0, np.nan, np.nan, np.nan,0,0,0]
    
    
        # Converting carbon to dry matter
        df['aC_harvest_export[kgCha-1]'] = df['aC_harvest_export[kgCha-1]'] * (1 / 0.45)
    
        df = df[
            ~(
                (df['aC_harvest_export[kgCha-1]'] < 1000) |
                ((df['n_synthamount'] <= 140) & (df['n_synth_replication'] > 2)) |
                ((df['n_org_amount'] <= 140) & (df['n_org_replication'] > 2)) |
                ((df['n_synthamount'] < 70) & (df['n_synth_replication'] > 1)) |
                (df['n_synthamount'].between(70, 140) & (df['n_synth_replication'] > 2))
            )
        ]
    
        # Set calculated values
        df.loc[df['n_org_replication'] == 1, 'manu_amount_1'] = df['n_org_amount']
        df.loc[df['n_synth_replication'] == 1, 'fert_amount_1'] = df['n_synthamount']
    
    
        df.loc[df['n_org_replication'] == 1, 'manu_daytoplanting_1'] = df['manu_daytoplanting']
        df.loc[df['n_synth_replication'] == 1, 'fert_daytoplanting_1'] = df['fert_daytoplanting']
    
        df.loc[df['n_org_type'] == 'none', 'n_org_type'] = None
        df.loc[df['n_synth_type'] == 'none', 'n_synth_type'] = None
    
    
        # Drop unused columns
        df = df.drop([
             'manu_rate_1', 'manu_rate_2', 'manu_rate_3',
            'fert_rate_1', 'fert_rate_2', 'fert_rate_3',
            'manu_amount', 'fert_amount', 'fert_daytoplanting', 'manu_daytoplanting'
        ], axis=1)
    
        #df = df.fillna(0)
    
        # Rename outputs
        df.rename(columns={
            'dN_n2o_emis[kgNha-1]': 'n2o',
            'dN_no3_leach[kgNha-1]': 'no3',
            'aC_harvest_export[kgCha-1]': 'yield',
            'C_pool_change[kgCha-1]': 'soc'
        }, inplace=True)
    
    
        # Track final counts
        final_counts = df['id'].value_counts()
        print(f"Final valid ID counts (should all be 10):\n{final_counts.value_counts()}")
    
        return df
    
    
    # Step 2: Calculate the number of chunks
    print("Now processing data")
    chunk_size = 1_100_000
    num_chunks = (len(df) + chunk_size - 1) // chunk_size  # Ceiling division
    
    # Step 3: Process the DataFrame in chunks
    for i in tqdm(range(num_chunks), desc="Processing chunks"):
        start = i * chunk_size
        end = start + chunk_size
        chunk = df.iloc[start:end]  # Slice the DataFrame to get the chunk
        processed = process_df(chunk)  # Apply your custom processing function
        
        out_path = f"data/clean/batch_{i}.parquet"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        processed.to_parquet(out_path, index=False)
    
    print("✅ Processing complete. All chunks saved separately.")
    
    
    
    
    start_time = time.time()
    
    # Step 1: Get all batch files (ensure they're sorted)
    batch_files = sorted(glob.glob("data/clean/batch_*.parquet"))
    
    # Step 2: Read and merge all batches
    df_list = [pd.read_parquet(file) for file in batch_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    
    
    # Step 3: More processing on the merged DataFrame
    
    merged_df['total_nitrogen'] = merged_df['n_org_amount'] + merged_df['n_synthamount'] 
    
    df_ready = merged_df.drop(['crops', 'crop_rotation_1', 'crop_rotation_2'], axis = 1)
    
    # Step 4: Encode categorical columns using LabelEncoder
    cat_columns = [
       'soil', 'climate','cropping_systems', 'crop_rotation', 'n_synth_type', 'n_org_type', 'irrigation', 'n_org_replication','n_synth_replication'
    ]
    
    df_ready['planting_dayofyear'] = df_ready['planting_dayofyear'].astype('Int64')
    df_ready['harvest_dayofyear'] = df_ready['harvest_dayofyear'].astype('Int64')
    
    
    df_ready[['n_org_amount', 'n_synthamount', 'total_nitrogen']] = df_ready[['n_org_amount', 'n_synthamount', 'total_nitrogen']].astype('float32')
    
    
    # Step 5 Save merged and proceseed file
    df_ready.to_parquet(f"data/ml_ready_{CROP}_outputs_cleaned.parquet", index=False)
    
    
    # Step 6: Print time taken
    end_time = time.time()
    print(f"✅ Merged {len(batch_files)} files and processed in {end_time - start_time:.2f} seconds.")


