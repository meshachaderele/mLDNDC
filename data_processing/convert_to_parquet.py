import cudf

def convert_to_parquet(CROP):

    df = cudf.read_csv(f"data/ml_ready_{CROP}_outputs.csv")
    
    df.to_parquet(f"data/ml_ready_{CROP}_outputs.parquet", index = False)

