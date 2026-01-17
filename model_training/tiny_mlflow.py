import os
import json
import uuid
import datetime as dt
from pathlib import Path

import joblib  # pip install joblib if you do not have it


def _json_default(obj):
    """Fallback serializer for numpy, datetime, etc."""
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()

    # Last resort: string representation
    return str(obj)


def log_multi_target_run_local(
    model_family_name,
    models,
    metrics,
    feature_names,
    transformers,
    experiment_name,
    run_name,
    base_dir="experiments_logs",
):
    """
    Lightweight experiment logger that mimics MLflow style logging
    but stores runs as JSON and model files on disk.

    Parameters
    ----------
    model_family_name : str
        Short name for the model family, for example "xgboost".
    models : dict or list
        Dict like {target_name: model} or list of models.
    metrics : dict
        Any nested dict of metrics. Must be JSON serializable or convertible.
    feature_names : list
        List of feature names used for training.
    transformers : object or dict or None
        Preprocessing transformers. Saved as a separate artifact if provided.
    experiment_name : str
        Name of the experiment, used as a folder name.
    run_name : str
        Name of the run, used in folder and JSON.
    base_dir : str or Path
        Root folder where experiments and artifacts are stored.
    """
    base_dir = Path(base_dir)
    experiment_dir = base_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex
    timestamp = dt.datetime.utcnow().isoformat() + "Z"

    # Run directory for artifacts
    safe_run_name = run_name.replace(" ", "_")
    run_dir = experiment_dir / f"{safe_run_name}_{run_id[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    models_info = {}
    if isinstance(models, dict):
        model_items = models.items()
    else:
        # If it is a list or any sequence, index it
        model_items = enumerate(models)

    for target, model in model_items:
        model_filename = f"model_{target}.pkl"
        model_path = run_dir / model_filename
        joblib.dump(model, model_path)

        models_info[str(target)] = {
            "artifact_path": str(model_path),
            "repr": repr(model),
            "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        }

    # Save transformers if provided
    transformers_info = None
    if transformers is not None:
        transformers_filename = "transformers.pkl"
        transformers_path = run_dir / transformers_filename
        joblib.dump(transformers, transformers_path)
        transformers_info = {
            "artifact_path": str(transformers_path),
            "repr": repr(transformers),
            "class": f"{transformers.__class__.__module__}.{transformers.__class__.__name__}",
        }

    # Build run record
    run_record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "model_family_name": model_family_name,
        "feature_names": list(feature_names) if feature_names is not None else None,
        "models": models_info,
        "transformers": transformers_info,
        "metrics": metrics,
        "artifacts_dir": str(run_dir),
    }

    # Append to experiment JSONL log
    log_path = experiment_dir / "experiments.jsonl"
    with open(log_path, "a", encoding="utf8") as f:
        json.dump(run_record, f, default=_json_default)
        f.write("\n")

    return run_record
