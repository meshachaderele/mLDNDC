This script joins GPU-accelerated simulation outputs with climate, soil, and precipitation-day features, then converts the result to a CPU pandas DataFrame and stores it as a Parquet file.[1][2][3]

***

## Overview

The workflow performs the following steps:

- Load multiple feature datasets (simulation outputs, climate features, soil data, precipitation-day counts) using cuDF for GPU-accelerated processing.[4][1]
- Perform a series of joins to attach climate and soil context to each simulation record.  
- Convert the final cuDF DataFrame to pandas and persist it as a Parquet file `data/df_SBAR.parquet` for downstream modeling or analysis.[3][5]

***

## Inputs

- `data/ml_ready_SBAR_outputs_090126.parquet`  
  - Simulation-level outputs, expected to contain: `id`, `year`, `site_id`.  
- `data/simulation_ids_SBAR_with_climate.parquet`  
  - Simulation IDs enriched with climate features; expected columns include: `event_id`, `harvest_year`, `climate_id`, and additional climate feature columns.  
- `../data/soil_data_with_classes.parquet`  
  - Soil attributes per `site_id`, including soil classes and related properties.  
- `../data/climate_10km_2010_2020.csv`  
  - Full climate time series; loaded but not used further in this script.  
- `data/climate_prec_days.csv`  
  - Annual precipitation-day counts per `climate_id` and `year`.

All paths are relative to the script’s working directory and should point to existing files readable by cuDF.[6][1]

***

## Data Loading

The script loads all datasets onto the GPU using cuDF:

```python
df = cudf.read_parquet('data/ml_ready_SBAR_outputs_090126.parquet') 
clim = cudf.read_parquet('data/simulation_ids_SBAR_with_climate.parquet')
soil = cudf.read_parquet('../data/soil_data_with_classes.parquet')
climate_df = cudf.read_csv('../data/climate_10km_2010_2020.csv')
prec_days = cudf.read_csv('data/climate_prec_days.csv')
```

- `cudf.read_parquet` and `cudf.read_csv` load Parquet and CSV data directly into GPU memory for fast subsequent joins.[1][4]
- `climate_df` is read but not used downstream; it can be removed if no future extension requires it.

***

## Merging Feature Tables

### Attach climate features to simulation outputs

```python
df = df.merge(clim, left_on=['id', 'year'], right_on=['event_id', 'harvest_year'])
del clim
gc.collect()
```

- Inner join between simulation outputs (`df`) and climate-enriched simulation IDs (`clim`).  
- Join keys:
  - Left: `id`, `year`.  
  - Right: `event_id`, `harvest_year`.  
- After merging, `clim` is deleted and `gc.collect()` is called to free GPU memory, which is important for large datasets.[2][7]

### Attach soil information

```python
df = df.merge(soil, on='site_id')
del soil
gc.collect()
```

- Join on `site_id` adds soil attributes (e.g., soil class, texture) to each simulation record.  
- The soil DataFrame is deleted afterward to release memory.

### Attach precipitation-day counts

```python
df = df.merge(prec_days, on=['climate_id', 'year'])
```

- Join on `climate_id` and `year` attaches annual `prec_days` features to each record, providing an additional climate indicator per simulation-climate-year combination.

***

## Conversion to pandas and Export

```python
pdf = df.to_pandas()
del df
gc.collect()
pdf.to_parquet("data/df_SBAR.parquet", index=False)
```

- `df.to_pandas()` transfers the GPU DataFrame to CPU as a standard pandas DataFrame for compatibility with downstream tools that expect pandas.[7][2]
- The cuDF DataFrame is deleted and garbage collected to free GPU memory.  
- `pdf.to_parquet(...)` writes the final dataset to a columnar Parquet file, which is efficient in terms of storage and I/O for analytical workloads.[5][8][3]

***

## Output

- **File:** `data/df_SBAR.parquet`  
- **Content:** Simulation outputs enriched with:
  - Climate features by event and harvest year.  
  - Soil features by site.  
  - Annual precipitation-day counts by climate grid and year.

This file serves as a consolidated, model-ready dataset optimized for subsequent machine learning experiments or statistical analyses.
