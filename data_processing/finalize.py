import cudf
import gc


def finalize(CROP):
    df = cudf.read_parquet(f'data/ml_ready_{CROP}_outputs_cleaned.parquet') 
    clim = cudf.read_parquet(f'data/simulation_ids_{CROP}_with_climate.parquet')
    soil = cudf.read_parquet('../data/soil_data_with_classes.parquet')
    climate_df = cudf.read_csv('../data/climate_10km_2010_2020.csv')
    prec_days = cudf.read_csv('data/climate_prec_days.csv')
    df = df.merge(clim, left_on=['id', 'year'], right_on=['event_id', 'harvest_year'])
    del clim
    gc.collect()
    df = df.merge(soil, on='site_id')
    del soil
    gc.collect()
    df = df.merge(prec_days, on=['climate_id', 'year'])
    pdf = df.to_pandas()
    del df
    gc.collect()
    pdf.to_parquet(f"data/df_{CROP}.parquet", index=False)
    
    
    print("All Done")