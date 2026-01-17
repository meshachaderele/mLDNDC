import mlflow
import mlflow.xgboost  # for xgboost models
import mlflow.catboost
import mlflow.lightgbm
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

TARGETS = ["n2o", "no3", "yield", "soc"]

import json

def log_multi_target_run(
    model_family_name,
    models,
    metrics,
    feature_names,
    transformers,
    experiment_name="multi_target_experiment",
    run_name=None,
):
    """
    models: dict[target] -> model object
    metrics: dict[target] -> {"rmse": ..., "mae": ..., "r2": ...}
    feature_names: list of feature column names
    transformers: dict[target] -> None or PowerTransformer for soc
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name or model_family_name):
        # Log feature list
        mlflow.log_dict({"features": feature_names}, "features.json")

        # Log metrics per target
        for target, m in metrics.items():
            mlflow.log_metric(f"{target}_rmse", m["rmse"])
            mlflow.log_metric(f"{target}_mae",  m["mae"])
            mlflow.log_metric(f"{target}_r2",   m["r2"])

        # Log model parameters and models per target
        for target, model in models.items():
            params = model.get_params() if hasattr(model, "get_params") else {}
            mlflow.log_params({f"{target}_{k}": v for k, v in params.items()})

            # XGBoost example
            if model_family_name == "xgboost":
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path=f"{target}_model",
                )
            # similar blocks for catboost and lightgbm below
            
            if model_family_name == "catboost":
                
                mlflow.catboost.log_model(
                    catboost_model=model,
                    artifact_path=f"{target}_model",
                )
            if model_family_name == "lightgbm":
                mlflow.lightgbm.log_model(
                    lgbm_model=model,
                    artifact_path=f"{target}_model",
                )

        # log soc transformer if you want to reuse it in deployment
        if transformers.get("soc") is not None:
            
            joblib.dump(transformers["soc"], "soc_yeojohnson.pkl")
            mlflow.log_artifact("soc_yeojohnson.pkl", artifact_path="preprocessors")





def transform_targets(y_train_cudf):
    """
    y_train_cudf: cuDF with columns n2o, no3, yield, soc in ORIGINAL scale.
    
    Returns:
      y_orig_np: dict[target] -> original numpy 1d
      y_trans_np: dict[target] -> transformed numpy 1d
      transformers: dict[target] -> None or fitted PowerTransformer (for soc)
    """
    y_orig_np = {}
    y_trans_np = {}
    transformers = {}

    for target in TARGETS:
        y = y_train_cudf[target].to_numpy().astype("float32")
        y_orig_np[target] = y

        if target in ["n2o", "no3", "yield"]:
            # log1p transform
            y_trans = np.log1p(y)
            y_trans_np[target] = y_trans
            transformers[target] = None     # no transformer needed
        else:
            # soc uses Yeo Johnson
            pt = PowerTransformer(method="yeo-johnson")
            y_trans = pt.fit_transform(y.reshape(-1, 1)).astype("float32").ravel()
            y_trans_np[target] = y_trans
            transformers[target] = pt

    return y_orig_np, y_trans_np, transformers


def compute_metrics(target, preds_trans, y_orig, transformer):
    """
    Convert preds_trans back to original scale and compute RMSE, MAE, R2.
    """
    if target in ["n2o", "no3", "yield"]:
        preds_orig = np.expm1(preds_trans)
    else:
        preds_orig = transformer.inverse_transform(
            preds_trans.reshape(-1, 1)
        ).ravel()

    rmse = np.sqrt(np.mean((preds_orig - y_orig) ** 2))
    mae = mean_absolute_error(y_orig, preds_orig)
    r2 = r2_score(y_orig, preds_orig)

    return {"rmse": rmse, "mae": mae, "r2": r2}, preds_orig


def compute_metrics(
    target,
    preds_trans,
    y_orig,
    transformer,
    model_name,
    save_dir="data",
):
    """
    Convert preds_trans back to original scale, compute RMSE, MAE, R2,
    and save y_orig and preds_orig to Parquet.

    - Columns: {target}, {target}_pred
    - File:    data/{model_name}_{target}_predictions.parquet
    """
    # Inverse transform
    if target in ["n2o", "no3", "yield"]:
        preds_orig = np.expm1(preds_trans)
    else:
        preds_orig = transformer.inverse_transform(
            preds_trans.reshape(-1, 1)
        ).ravel()

    # Ensure y_orig is 1d
    y_orig_arr = np.asarray(y_orig).ravel()

    # Metrics
    rmse = np.sqrt(np.mean((preds_orig - y_orig_arr) ** 2))
    mae = mean_absolute_error(y_orig_arr, preds_orig)
    r2 = r2_score(y_orig_arr, preds_orig)

    # Save predictions
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir,
        f"{model_name}_{target}_predictions.parquet",  
       
    )

    df_out = pd.DataFrame(
        {
            target: y_orig_arr,
            f"{target}_pred": preds_orig,
        }
    )
    df_out.to_parquet(out_path, index=False)

    return {"rmse": rmse, "mae": mae, "r2": r2}, preds_orig



