This script builds seasonal climate features for spring or winter crops, joins them with aridity-based climate classes, and attaches the resulting features to simulation IDs, saving the enriched dataset as a Parquet file.[1][2]

***

## Overview

The workflow can be summarized as:
- Aggregate daily climate data into seasonal and annual features per grid cell and **harvest_year**.[1]
- Classify locations into aridity-based classes using an aridity index.[3]
- Merge climate features and classes with simulation IDs and save as `simulation_ids_SBAR_with_climate.parquet`.[2]

***

## Inputs

- `../data/climate_10km_2010_2020.csv`  
  - Expected columns: `id`, `date`, `tavg`, `prec`.  
- `../data/climate_ai_classified.csv`  
  - Expected columns: `id`, `latitude`, `longitude`, `temperature average`, `annual precipitation`, `Aridity index`.  
- `data/simulation_ids_SBAR.parquet`  
  - Expected columns: at least `climate_id` and `folder` plus simulation identifiers.

All inputs must be consistent in spatial IDs (`id` vs `climate_id`) and cover overlapping years for meaningful joins.[4]

***

## Climate Feature Construction

### Function purpose

`build_climate_features(clim: pd.DataFrame, crop_type: str) -> pd.DataFrame` constructs annual and seasonal temperature and precipitation features per grid cell and **harvest_year**.[1]

- `clim` must contain: `id`, `date`, `tavg`, `prec`.  
- `crop_type` controls growing-season definitions:
  - `"winter"`: October (Y−1) to July (Y) for harvest year Y.  
  - `"spring"`: March–August of calendar year Y for harvest year Y.

The function returns one row per `(id, harvest_year)` with aggregated sums over different periods.

### Winter crops logic

For `crop_type == "winter"`:

- `date` is converted to datetime; `year` and `month` are extracted.[4]
- `harvest_year`:
  - Months Oct–Dec are assigned to `year + 1`.  
  - Other months use the calendar year.  
- Boolean flags:
  - `is_autumn`: months 10–11.  
  - `is_winter`: months 12, 1, 2.  
  - `is_spring`: months 3–5.  
  - `is_gs`: growing season months 10–12 and 1–7.

Aggregations (all grouped by `id`, `harvest_year`):[1]

- `yearly`:
  - `total_precipitation_year`: sum of `prec` over full calendar year of `harvest_year`.  
  - `total_average_temperature_year`: sum of `tavg` over full calendar year.  
- `gs` (growing season):
  - `total_precipitation_growing_season`.  
  - `total_average_temperature_growing_season`.  
- `autumn`:
  - `total_precipitation_autumn`.  
  - `total_average_temperature_autumn`.  
- `winter`:
  - `total_precipitation_winter`.  
  - `total_average_temperature_winter`.  
- `spring`:
  - `total_precipitation_spring`.  
  - `total_average_temperature_spring`.

These tables are joined on `["id", "harvest_year"]`, then the index is reset.

### Spring crops logic

For `crop_type == "spring"`:

- `harvest_year` equals calendar `year`.  
- Seasonal flags:
  - `is_spring`: months 3–5.  
  - `is_summer`: months 6–8.  
  - `is_gs`: months 3–8.

Aggregations (grouped by `id`, `harvest_year`):[1]

- `yearly`:
  - `total_precipitation_year`.  
  - `total_average_temperature_year`.  
- `gs` (growing season Mar–Aug):
  - `total_precipitation_growing_season`.  
  - `total_average_temperature_growing_season`.  
- `spring`:
  - `total_precipitation_spring`.  
  - `total_average_temperature_spring`.  
- `summer`:
  - `total_precipitation_summer`.  
  - `total_average_temperature_summer`.

The result is similarly joined and returned with a flat index.

***

## Aridity Class Assignment

The script defines `assign_class(value)` to discretize `Aridity index` into categorical classes:

- Class 1: `28 <= value <= 31.05`.  
- Class 2: `31.06 <= value <= 35`.  
- Class 3: `35 < value <= 45`.  
- Class 4: `45 < value <= 55`.  
- Any value outside these ranges is labelled `"Unknown"`.

`df_class_climate['Class']` is created by applying `assign_class` to the `Aridity index` column.[3]

This adds a coarse climate-type label that can be used as a categorical feature in models.

***

## Merging and Output

### Climate feature filtering

- `new_df = build_climate_features(df, "spring")` builds features for spring crops.  
- `new_df` rows with `harvest_year` 2010 and 2021 are removed, keeping only years with complete seasonal coverage in the 2010–2020 input period.

### Joining climate classes and features

- `climate_df = pd.merge(df_class_climate, new_df, on='id')` performs an inner join on `id`, retaining only grid cells present in both datasets.[1]
- Columns dropped from `climate_df`:
  - `latitude`, `longitude`, `temperature average`, `annual precipitation`, `Aridity index`.  
  These are superseded by the newly derived features and the `Class` label.

### Attaching to simulation IDs

- `sim_ids_df['climate_id']` is cast to integer to match the climate grid ID type.  
- `sim_ids_climate = sim_ids_df.merge(climate_df, left_on="climate_id", right_on="id", how="left")` attaches climate features to each simulation row.[4]
- Columns dropped:
  - `folder`: simulation folder path not needed in final output.  
  - `id`: redundant, as `climate_id` remains.

### Writing the final dataset

The enriched simulation dataset is saved as a Parquet file:

```python
sim_ids_climate.to_parquet("data/simulation_ids_SBAR_with_climate.parquet", index=False)
```

The `to_parquet` method writes a columnar binary file suitable for efficient storage and I/O in downstream workflows.[2]

***