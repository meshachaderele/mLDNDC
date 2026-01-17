Here's a structured, professional-style Markdown documentation for your script. It explains the purpose, workflow, and logic clearly (ideal for repositories, team handovers, or reproducible pipelines).  

***

# Climate Precipitation Days Aggregation Script

## Overview
This script processes gridded climate data to compute the **number of precipitation days per grid cell and year**. It converts daily precipitation records into annual summaries, where each record represents one grid cell (`id`) and one calendar year.

The output is a summarized dataset ready for downstream statistical modeling or visualization.

***

## Table of Contents
1. [Requirements](#requirements)  
2. [Input Data](#input-data)  
3. [Processing Steps](#processing-steps)  
4. [Output Data](#output-data)  
5. [Code Walkthrough](#code-walkthrough)  
6. [Notes and Best Practices](#notes-and-best-practices)

***

## Requirements

**Dependencies:**
- Python ≥ 3.8  
- pandas ≥ 1.3  

Install required dependencies using:
```bash
pip install pandas
```

**Input file location:**
```
../data/climate_10km_2010_2020.csv
```

**Output directory:**
```
data/
```

Ensure that the output folder exists before running the script.

***

## Input Data

**Expected structure of `climate_10km_2010_2020.csv`:**

| Column | Type | Description |
|---------|-------|-------------|
| `id` | int or str | Unique grid cell identifier |
| `date` | str (YYYY-MM-DD) | Observation date |
| `prec` | float | Daily precipitation amount (e.g., in mm) |

***

## Processing Steps

1. **Load climate data:**  
   The script reads the raw climate data from a CSV file into a pandas DataFrame.

2. **Parse and extract date components:**  
   The `date` column is converted to datetime format to enable time-based operations.  
   A `year` column is created from the date.

3. **Aggregate precipitation days:**  
   For each grid cell (`id`) and year, the script counts the number of days with precipitation greater than zero.

4. **Rename and save output:**  
   The aggregated data is saved as a CSV file with standardized column names (`climate_id`, `year`, `prec_days`).

***

## Output Data

**Output file:**  
```
data/climate_prec_days.csv
```

**Output columns:**

| Column | Type | Description |
|---------|------|-------------|
| `climate_id` | int or str | Grid cell identifier |
| `year` | int | Year of aggregation |
| `prec_days` | int | Number of days with precipitation > 0 mm |

***

## Code Walkthrough

```python
import pandas as pd

# 1. Load data
df = pd.read_csv("../data/climate_10km_2010_2020.csv")

# 2. Parse datetime and extract year
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year   

# 3. Group and aggregate by grid cell and year
df_climate = (
    df.groupby(['id', 'year'])
      .agg(prec_days=('prec', lambda s: (s > 0).sum()))
      .reset_index()
      .rename(columns={'id': 'climate_id'})
)

# 4. Save output
df_climate.to_csv("data/climate_prec_days.csv", index=False)
```

**Key computation:**
\[
\text{prec\_days} = \sum_{d=1}^{N} 1_{\text{prec}_d > 0}
\]
where \( N \) is the number of daily records in a given year for a given grid cell.

***

## Notes and Best Practices

- **Missing data:** This script does not explicitly handle missing values. If your dataset may include `NaN` in `prec`, consider cleaning it first.
- **Custom thresholds:** To define precipitation days based on a threshold (e.g., >1 mm), modify the lambda function accordingly.
- **Performance:** For very large datasets, ensure sufficient memory or consider using chunked reading with `pd.read_csv(..., chunksize=...)`.

***