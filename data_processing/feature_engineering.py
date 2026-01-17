import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from datetime import timedelta
import json
import time
#GPU
import cudf
import cupy as cp

from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder

from cuml.preprocessing import StandardScaler

import sys
import warnings

import gc

warnings.filterwarnings("ignore", category=UserWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# # Function

# In[2]:


def daytoplanting_to_date_cpu(df, col):
    """
    CPU version using pandas.
    Planting occurs in the PREVIOUS year relative to df['year'].
    The value inside df[col] is the offset in days relative to planting.
    """

    # planting year is previous year
    planting_year = (df["year"].astype(int) - 1).astype(str)

    # construct planting date
    planting_date = pd.to_datetime(planting_year, format="%Y") \
                     + pd.to_timedelta(df["planting_dayofyear"] - 1, unit="D")

    # offset in days (positive or negative)
    offset_days = df[col].astype(float)

    # final event date
    event_date = planting_date + pd.to_timedelta(offset_days, unit="D")

    return event_date



def compute_prec_before_after_gpu_optimized(df, climate_df, chunk_size=100000):
    """
    Optimized version for large datasets (45M+ rows) with proper NaN handling
    """
    # Step 1: Pre-process climate data ONCE
    print("Pre-processing climate data...")
    climate_cudf = cudf.DataFrame({
        'climate_id': climate_df['climate_id'].values,
        'year': climate_df['dates'].dt.year.astype("int32"),
        'doy_clim': climate_df['dates'].dt.dayofyear.astype("int32"),
        'prec': climate_df['prec'].astype("float32")
    })

    # Step 2: Pre-process main dataframe dates ONCE with NaN handling
    print("Pre-processing main dataframe dates...")
    date_columns = []
    for i in [1, 2, 3]:
        for prefix in ["manu_date", "fert_date"]:
            col = f"{prefix}_{i}"
            year_col = f"{col}_year"
            doy_col = f"{col}_doy"

            # Handle NaN values - use float32 instead of int32 for years with NaN
            df[year_col] = df[col].dt.year.astype("float32")  # Use float to preserve NaN
            df[doy_col] = df[col].dt.dayofyear.astype("float32")  # Use float to preserve NaN
            date_columns.extend([year_col, doy_col])

    # Keep only the columns we need
    keep_columns = ['id', 'climate_id'] + date_columns
    df_processed = df[keep_columns].copy()

    # Step 3: Process in chunks
    print(f"Processing {len(df_processed)} rows in chunks of {chunk_size}...")
    all_results = []

    for chunk_start in range(0, len(df_processed), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df_processed))
        print(f"Processing chunk {chunk_start}-{chunk_end}...")

        df_chunk = df_processed.iloc[chunk_start:chunk_end]
        chunk_results = process_chunk(df_chunk, climate_cudf)

        if chunk_results is not None and len(chunk_results) > 0:
            all_results.append(chunk_results)

        # Clear memory between chunks
        del df_chunk
        clear_gpu_memory()

    # Step 4: Combine all results
    print("Combining results...")
    if all_results:
        final_results = combine_chunk_results(all_results, df['id'].unique())
    else:
        final_results = create_empty_results(df['id'].unique())

    return final_results

def process_chunk(df_chunk, climate_cudf):
    """Process a single chunk of data"""
    # Convert chunk to cuDF
    df_cudf = cudf.from_pandas(df_chunk)

    chunk_results = []

    # Process each date type
    for i in [1, 2, 3]:
        # Manufacturing dates
        manu_results = process_single_date_type(
            df_cudf, climate_cudf, f"manu_date_{i}", i, "manu"
        )
        if manu_results is not None and len(manu_results) > 0:
            chunk_results.append(manu_results)

        # Fertilization dates  
        fert_results = process_single_date_type(
            df_cudf, climate_cudf, f"fert_date_{i}", i, "fert"
        )
        if fert_results is not None and len(fert_results) > 0:
            chunk_results.append(fert_results)

    if chunk_results:
        return cudf.concat(chunk_results, ignore_index=True)
    return None

def process_single_date_type(df_cudf, climate_cudf, date_prefix, idx, event_type):
    """Process precipitation for a single date type (manu/fert)"""
    # Check if this date column has any non-null values
    year_col = f"{date_prefix}_year"
    doy_col = f"{date_prefix}_doy"

    if year_col not in df_cudf.columns:
        return None

    # Filter out rows where this date is null
    valid_dates = df_cudf[df_cudf[year_col].notnull()]
    if len(valid_dates) == 0:
        return None

    results = []

    # Define windows: (start_offset, end_offset, label_suffix)
    windows = [
        (-7, 0, f"total_precipitation_7_before_{event_type}_{idx}"),
        (0, 3, f"total_precipitation_3_after_{event_type}_{idx}")
    ]

    for start_offset, end_offset, label in windows:
        window_result = compute_precipitation_window(
            valid_dates, climate_cudf, year_col, doy_col, 
            start_offset, end_offset, label
        )
        if window_result is not None and len(window_result) > 0:
            results.append(window_result)

    if results:
        return cudf.concat(results, ignore_index=True)
    return None

def compute_precipitation_window(df_cudf, climate_cudf, year_col, doy_col, 
                               start_offset, end_offset, label):
    """Compute precipitation for a specific date window"""
    # Create window boundaries
    window_data = df_cudf[['id', 'climate_id', year_col, doy_col]].copy()
    window_data['start_doy'] = window_data[doy_col] + start_offset
    window_data['end_doy'] = window_data[doy_col] + end_offset
    window_data['label'] = label

    # Rename for merge
    window_data = window_data.rename(columns={year_col: 'year', doy_col: 'doy'})

    # Merge with climate data
    merged = window_data.merge(
        climate_cudf, 
        on=['climate_id', 'year'], 
        how='inner'
    )

    if len(merged) == 0:
        return None

    # Filter for dates in our window
    in_window = merged[
        (merged['doy_clim'] >= merged['start_doy']) & 
        (merged['doy_clim'] <= merged['end_doy'])
    ]

    if len(in_window) == 0:
        return None

    # Aggregate precipitation
    aggregated = in_window.groupby(['id', 'label'], as_index=False).agg({
        'prec': 'sum'
    })

    return aggregated

def combine_chunk_results(all_results, all_ids):
    """Combine results from all chunks"""
    if not all_results:
        return create_empty_results(all_ids)

    # Concatenate all results
    combined = cudf.concat(all_results, ignore_index=True)

    # Use pivot for better performance
    pivoted = combined.pivot(index='id', columns='label', values='prec')
    pivoted = pivoted.reset_index()

    # Ensure all original IDs are present
    all_ids_cudf = cudf.DataFrame({'id': all_ids})
    final = all_ids_cudf.merge(pivoted, on='id', how='left')

    return final

def create_empty_results(all_ids):
    """Create empty results with correct structure"""
    empty_df = cudf.DataFrame({'id': all_ids})

    # Add all expected columns
    columns = []
    for i in [1, 2, 3]:
        for event in ['manu', 'fert']:
            columns.extend([
                f"total_precipitation_7_before_{event}_{i}",
                f"total_precipitation_3_after_{event}_{i}"
            ])

    for col in columns:
        empty_df[col] = float('nan')

    return empty_df

def clear_gpu_memory():
    """Clear GPU memory between chunks"""
    import gc
    gc.collect()

def process_in_batches(df, climate_df, batch_size=100000):
    """Process extremely large datasets in memory-managed batches"""
    results = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        print(f"Processing batch {i} to {i + batch_size}...")

        batch_result = compute_prec_before_after_gpu_optimized(batch, climate_df, chunk_size=50000)
        results.append(batch_result.to_pandas() if hasattr(batch_result, 'to_pandas') else batch_result)

        # Clear GPU memory
        clear_gpu_memory()

    # Combine all results
    final_result = pd.concat(results, ignore_index=True)
    return final_result


# In[3]:


def add_engineered_features(df):

    # synthetic to organic ratio
    df["synth_org_ratio"] = df["n_synthamount"] / (df["n_synthamount"] + df["n_org_amount"])

    df["synth_org_ratio"] = df["synth_org_ratio"].fillna(0)

    # precipitation x clay
    df["precipitation_clay_interaction"] = df["total_precipitation_growing_season"] * df["clay"]



    # precipitation x fertilizer
    df["precip_n_interaction"] = df["total_precipitation_growing_season"] * df["total_nitrogen"]

    return df



def process_data_gpu(gdf):

    # 1. drop predictors
    #predictors = ['soc', 'no3', 'n2o']
    #gdf = gdf.drop(columns=predictors)

    fill_value = "No Application"

    gdf.loc[gdf["n_org_type"].isna(), "n_org_type"] = fill_value

    gdf.loc[gdf["n_synth_type"].isna(), "n_synth_type"] = fill_value



    amount_cols = [
        'no3', 'n2o', 'yield', 'soc',
    ]



    lab_cols = [
        'soil','climate','cropping_systems','crop_rotation',
        'n_synth_type','n_org_type','n_org_replication',
        'n_synth_replication','irrigation','manu_depth'
    ]

    ord_cols = ['n_org_amount','n_synthamount']

    # 2. impute amount columns with zero
    gdf[amount_cols] = gdf[amount_cols].fillna(0)


    # 5. categorical encoding on GPU
    cat_cols = lab_cols + ord_cols
    # Training data
    gdf, cat_maps = encode_cats_train(gdf, cat_cols)

    # Save cat_maps if you want

    with open("encoders/cat_maps.json", "w") as f:
        json.dump(cat_maps, f)

    # 6. compute passthrough columns
    passthrough_cols = [
        c for c in gdf.columns
        if c not in amount_cols + lab_cols + ord_cols
    ]

    # 7. final logical ordering
    final_cols = (
        amount_cols +
        lab_cols +
        ord_cols +
        passthrough_cols
    )

    return gdf[final_cols]


# In[4]:


def encode_cats_train(gdf, cat_cols):
    """
    Fit categorical encoders on TRAIN data.

    Parameters
    ----------
    gdf : cudf.DataFrame
    cat_cols : list of str
        Columns to treat as categorical.

    Returns
    -------
    gdf_encoded : cudf.DataFrame
        DataFrame with categorical columns encoded as int32 codes.
    cat_maps : dict
        {col_name: list_of_categories_in_training_order}
    """
    gdf = gdf.copy()
    cat_maps = {}

    for col in cat_cols:
        s_cat = gdf[col].astype("category")
        cats_idx = s_cat.cat.categories                # cuDF Index

        cats_ser = cudf.Series(cats_idx)
        cat_maps[col] = cats_ser.to_pandas().tolist()
        gdf[col] = s_cat.cat.codes.astype("int32")

    return gdf, cat_maps


def encode_cats_apply(gdf, cat_maps):
    """
    Apply TRAIN encoders to NEW data (val/test) in STRICT mode.

    Behavior:
    - If a column in cat_maps is missing in gdf -> raises ValueError.
    - If gdf has categories not seen in training -> raises ValueError
      listing the offending values.

    Parameters
    ----------
    gdf : cudf.DataFrame
        New data (validation/test).
    cat_maps : dict
        {col_name: list_of_categories_in_training_order}

    Returns
    -------
    gdf_encoded : cudf.DataFrame
        DataFrame with categorical columns encoded as int32 codes.
    """
    gdf = gdf.copy()

    for col, cats in cat_maps.items():
        if col not in gdf.columns:
            raise ValueError(
                f"Column '{col}' not found in provided DataFrame. "
                f"This column was present during training and is required."
            )

        # cast to category first
        s = gdf[col].astype("category")

        # find unseen categories (ignoring nulls)
        mask_non_null = s.notnull()
        mask_known = s.isin(cats)
        mask_unknown = mask_non_null & (~mask_known)

        if bool(mask_unknown.any()):
            unknown_values = (
                s[mask_unknown]
                .unique()
                .to_pandas()
                .tolist()
            )
            raise ValueError(
                f"Column '{col}' contains categories not seen during training: "
                f"{unknown_values}. "
                f"Update your training encoders or clean your data before encoding."
            )

        # all categories are known at this point, safe to enforce ordering
        s = s.cat.set_categories(cats)
        gdf[col] = s.cat.codes.astype("int32")

    return gdf



# # Process

def process_training(CROP):

    # ---- Load data ----
    t0 = time.perf_counter()
    df = pd.read_parquet(f"data/df_{CROP}.parquet")
    climate_df = pd.read_csv("../data/climate_10km_2010_2020.csv")
    climate_df = climate_df.rename(columns={"id": "climate_id", "date": "dates"})
    climate_df["dates"] = pd.to_datetime(climate_df["dates"])
    t1 = time.perf_counter()
    print(f"[{CROP}] Load + preprocess climate data took {t1 - t0:.2f} seconds")

    # ---- Compute event dates ----
    t0 = time.perf_counter()
    df["manu_date_1"] = daytoplanting_to_date_cpu(df, "manu_daytoplanting_1")
    df["manu_date_2"] = daytoplanting_to_date_cpu(df, "manu_daytoplanting_2")
    df["manu_date_3"] = daytoplanting_to_date_cpu(df, "manu_daytoplanting_3")
    df["fert_date_1"] = daytoplanting_to_date_cpu(df, "fert_daytoplanting_1")
    df["fert_date_2"] = daytoplanting_to_date_cpu(df, "fert_daytoplanting_2")
    df["fert_date_3"] = daytoplanting_to_date_cpu(df, "fert_daytoplanting_3")
    t1 = time.perf_counter()
    print(f"[{CROP}] Compute event dates took {t1 - t0:.2f} seconds")

    # ---- Compute precipitation events (GPU batches) ----
    t0 = time.perf_counter()
    prec_event_df = process_in_batches(
        df.drop_duplicates(subset=["id"]), climate_df, batch_size=100_000
    )
    t1 = time.perf_counter()
    print(f"[{CROP}] process_in_batches took {t1 - t0:.2f} seconds")

    prec_event_df.to_parquet("data/prec_events.parquet", index=False)

    # ---- Drop unused columns ----
    unused_cols = [
        "planting_dayofyear",
        "harvest_dayofyear",
        "manu_daytoplanting_1",
        "manu_daytoplanting_2",
        "manu_daytoplanting_3",
        "fert_daytoplanting_1",
        "fert_daytoplanting_2",
        "fert_daytoplanting_3",
        "setup_id",
        "climate_id",
        "event_id",
        "site_id",
        "Class",
        "harvest_year",
        "JB_Classes",
        "manu_date_1",
        "manu_date_2",
        "manu_date_3",
        "fert_date_1",
        "fert_date_2",
        "fert_date_3",
    ]

    df.drop(unused_cols, axis=1, inplace=True, errors="ignore")

    # ---- Merge precipitation events ----
    df = df.merge(prec_event_df, on="id")
    del prec_event_df
    gc.collect()

    # ---- Main feature engineering pipeline ----
    starttime = time.time()

    plot_df_transformed = add_engineered_features(df)
    plot_df_transformed = process_data_gpu(plot_df_transformed)

    duration = (time.time() - starttime) / 60
    print(f"[{CROP}] Data processing done in {duration:.2f} minutes")

    # ---- Final transforms for training ----
    cols = [
        "total_precipitation_7_before_fert_1",
        "total_precipitation_7_before_fert_2",
        "total_precipitation_7_before_fert_3",
        "total_precipitation_7_before_manu_1",
        "total_precipitation_7_before_manu_2",
        "total_precipitation_7_before_manu_3",
        "total_precipitation_3_after_fert_1",
        "total_precipitation_3_after_fert_2",
        "total_precipitation_3_after_fert_3",
        "total_precipitation_3_after_manu_1",
        "total_precipitation_3_after_manu_2",
        "total_precipitation_3_after_manu_3",
    ]

    # 1) Create any missing columns, filled with -1
    for c in cols:
        if c not in plot_df_transformed.columns:
            plot_df_transformed[c] = -1
    
    # 2) Standardise: fill remaining NaNs with -1 in all these columns
    plot_df_transformed[cols] = plot_df_transformed[cols].fillna(-1)


    amount_cols = [
        "fert_amount_1",
        "fert_amount_2",
        "fert_amount_3",
        "manu_amount_1",
        "manu_amount_2",
        "manu_amount_3",
        "total_nitrogen",
    ]

    # 1) Ensure all amount columns exist, filled with 0 if missing
    for col in amount_cols:
        if col not in plot_df_transformed.columns:
            plot_df_transformed[col] = 0
    
    # 2) Create squared features
    for col in amount_cols:
        plot_df_transformed[f"{col}_sq"] = plot_df_transformed[col] ** 2
    
    # 3) Fill remaining NaNs in these columns (and their squares) with 0
    cols_to_fill = amount_cols + [f"{col}_sq" for col in amount_cols]
    plot_df_transformed[cols_to_fill] = plot_df_transformed[cols_to_fill].fillna(0)

    plot_df_transformed.to_parquet(
        f"data/df_{CROP}_training.parquet", index=False
    )

    print(f"[{CROP}] Done")

if __name__ == "__main__":
    process_training("WIWH") #WinterWheat








