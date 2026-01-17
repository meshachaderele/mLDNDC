
# mLDNDCv.1.0 Documentation  
**High-Performance Climate & Soil Feature Engineering and ML Training Pipeline for the production of (Machine Learning LandscapeDNDC) mLDNDCv1.0**


## 1. Overview

This codebase implements a **scalable GPU-accelerated data processing and machine learning pipeline** designed for very large agro-climatic datasets (tens of millions of rows).  
It supports end-to-end processing from raw climate and management data to model-ready features and trained ML models.

The system is optimized for:
- Extremely large tabular datasets (45M+ rows)
- GPU acceleration using RAPIDS (cuDF, cuML, CuPy)
- Reproducible model training with lightweight experiment tracking
- Scientific publication and methodological transparency



## 2. Project Structure

```
code/
├── data_processing/
│   ├── pipeline.py
│   ├── process.py
│   ├── feature_engineering.py
│   ├── add_prec_days.py
│   ├── add_prec_temp.py
│   ├── convert_to_parquet.py
│   ├── finalize.py
│   ├── ReadME.md
│   ├── 1_add_prec_days.md
│   ├── 2_add_prec_temp.md
│   └── 3_finalize.md
│
├── model_training/
│   ├── helper.py
│   ├── tiny_mlflow.py
│   ├── train_xgb.ipynb
│   ├── train_xgb_cv.ipynb
│   ├── train_lgb.ipynb
│   └── train_catboost.ipynb

```


## 3. Data Processing Module

### 3.1 Design Philosophy

The data processing layer is designed around:
- **Chunk-based GPU processing** to avoid memory overflow
- **Single-pass climate preprocessing** for efficiency
- **Explicit handling of missing dates**
- **Separation of concerns** between feature creation steps



### 3.2 `pipeline.py`

**Purpose**  
Acts as the orchestration layer for the entire preprocessing workflow.

**Responsibilities**
- Executes processing steps in the correct order
- Ensures intermediate outputs are written correctly
- Provides a reproducible processing pipeline



### 3.3 `process.py`

**Purpose**  
Main execution script that connects raw inputs to engineered outputs.

**Key Tasks**
- Loads management and climate datasets
- Applies data cleaning functions
- Handles dataset merging and integrity checks



### 3.4 `feature_engineering.py`

**Purpose**  
Core feature engineering logic optimized for very large datasets.

**Key Features**
- GPU-accelerated precipitation aggregation
- Temporal alignment of management events and climate data
- Robust NaN-safe date handling
- Chunk-based processing to support 45M+ rows

#### Key Functions

##### `daytoplanting_to_date_cpu`
- Converts day offsets relative to planting into absolute dates
- Handles planting occurring in the previous year
- CPU-based for accuracy and reliability

##### `compute_prec_before_after_gpu_optimized`
- Computes precipitation statistics around management events
- Uses cuDF and CuPy for GPU acceleration
- Processes data in configurable chunks
- Designed to minimize GPU memory fragmentation



### 3.5 `add_prec_days.py`

**Purpose**  
Adds precipitation-based features computed over day windows.

**Examples**
- Count of days it rained in a given year 



### 3.6 `add_prec_temp.py`

**Purpose**  
Extends precipitation features with temperature-based signals.

**Outputs**
- Joint precipitation–temperature indicators



### 3.7 `convert_to_parquet.py`

**Purpose**  
Optimizes storage and I/O performance.

**Why Parquet**
- Columnar format
- GPU-friendly
- Efficient compression
- Faster downstream training



### 3.8 `finalize.py`

**Purpose**  
Final cleanup and export stage.

**Tasks**
- Column validation
- Feature normalization
- Dataset consistency checks
- Writing final model-ready datasets



## 4. Model Training Module

### 4.1 Training Philosophy

The training pipeline focuses on:
- Strong baseline models
- Transparent evaluation
- Minimal overhead experiment tracking
- Reproducibility for scientific use



### 4.2 `helper.py`

**Purpose**  
Shared utilities for model training.

**Includes**
- Dataset loading helpers
- Feature selection logic
- Metric computation
- Consistent train-test splitting



### 4.3 `tiny_mlflow.py`

**Purpose**  
Lightweight experiment tracking utility.

**Why**
- Avoids heavy MLflow dependencies
- Logs parameters and metrics
- Suitable for HPC and cluster environments



### 4.4 Training Notebooks

#### `train_xgb.ipynb`
- XGBoost single-run training
- Baseline performance benchmarking

#### `train_xgb_cv.ipynb`
- Cross-validated XGBoost training
- Robust performance estimation
- Hyperparameter stability analysis

#### `train_lgb.ipynb`
- LightGBM training workflow
- Optimized for large tabular data

#### `train_catboost.ipynb`
- CatBoost training
- Native categorical feature handling



## 5. Hardware and Performance Considerations

### GPU Stack
- cuDF
- cuML
- CuPy
- RAPIDS ecosystem

### Key Optimizations
- Chunk-based GPU execution
- Explicit memory cleanup
- Minimal CPU–GPU transfers
- Float32 precision where appropriate







