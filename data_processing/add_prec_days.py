import pandas as pd

def get_precipitation_days():

    df = pd.read_csv("../data/climate_10km_2010_2020.csv")
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year   
    
    df_climate = (
        df.groupby(['id', 'year'])
          .agg(prec_days=('prec', lambda s: (s > 0).sum()))
          .reset_index()
          .rename(columns={'id': 'climate_id'})
    )
    
    
    df_climate.to_csv("data/climate_prec_days.csv", index=False)