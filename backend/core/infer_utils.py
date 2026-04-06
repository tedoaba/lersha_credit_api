"""ML inference utilities for the Lersha Credit Scoring backend.

Provides model loading, SHAP explanation generation, and prediction helpers.
Feature engineering and preprocessing are intentionally extracted to sibling
modules (feature_engineering.py and preprocessing.py) to keep this module
focused on ML inference concerns.

Dead code removed in monorepo refactor (2026-03-29):
  - generate_shap_value_summary_plotsss (triple-s, duplicate of the clean version below)
  - load_prediction_model() (singular — referenced removed _34_ model config attributes)
"""

import contextlib
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive, thread-safe — must be before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap

from backend.config.config import config
from backend.core.feature_engineering import apply_feature_engineering  # noqa: F401 (re-exported for pipeline)
from backend.core.preprocessing import load_features, preprocessing_categorical_features  # noqa: F401
from backend.logger.logger import get_logger

logger = get_logger(__name__)


def get_candidate_data(df: pd.DataFrame, columns_to_select: list) -> pd.DataFrame:
    """Drop the ID column and select only the required feature columns.

    Args:
        df: Input DataFrame containing all engineered features.
        columns_to_select: List of column names to keep.

    Returns:
        pd.DataFrame: Subset of ``df`` without the farmer UID column.
    """
    sample_data = df.drop(columns=[config.id_column], errors="ignore")
    sample_data = sample_data[columns_to_select]
    logger.info("Candidate data shape: %s", sample_data.shape)
    return sample_data


def load_prediction_models(model_name: str) -> object:
    """Load a trained prediction model by name.

    Attempts to load from the MLflow Model Registry first
    (``models:/lersha-{model_name}/Production``). If the registry is
    unreachable or the model has not yet been promoted to Production stage,
    falls back to loading from the configured local ``.pkl`` path.

    Supported model names: ``"xgboost"``, ``"random_forest"``, ``"catboost"``.

    Args:
        model_name: Identifier for the model to load.

    Returns:
        Loaded model object (sklearn Pipeline or CatBoostClassifier).

    Raises:
        ValueError: If ``model_name`` is not recognised.
        RuntimeError: If both the registry and the local ``.pkl`` fail to load.
    """
    pkl_path_map: dict[str, str] = {
        "xgboost": config.xgb_model_36,
        "random_forest": config.rf_model_36,
        "catboost": config.cab_model_36,
    }

    if model_name not in pkl_path_map:
        raise ValueError(f"Unknown model_name '{model_name}'. Expected: xgboost | random_forest | catboost")

    # ── 1. Try MLflow Model Registry ──────────────────────────────────────────
    registry_uri = f"models:/lersha-{model_name}/Production"
    try:
        model = mlflow.sklearn.load_model(registry_uri)
        mlflow.set_tag("model_source", f"registry:{registry_uri}")
        logger.info("Model '%s' loaded from MLflow registry (%s)", model_name, registry_uri)
        return model
    except Exception as exc:  # noqa: BLE001 — intentional broad catch for registry fallback
        logger.warning(
            "MLflow registry unavailable for '%s' (%s); falling back to local pkl.",
            model_name,
            exc,
        )

    # ── 2. Fallback: load from local .pkl file ────────────────────────────────
    pkl_path = pkl_path_map[model_name]
    try:
        model = joblib.load(pkl_path)
        mlflow.set_tag("model_source", f"local:{pkl_path}")
        logger.info("Model '%s' loaded from local pkl: %s", model_name, pkl_path)
        return model
    except Exception as exc:
        logger.error("Failed to load model '%s' from local pkl '%s'", model_name, pkl_path, exc_info=True)
        raise RuntimeError(
            f"Failed to load model '{model_name}' from both MLflow registry and local pkl '{pkl_path}'"
        ) from exc


def predict_single_sample_data(model, sample_data: pd.DataFrame, target_column: str, model_name: str):
    """Predict the credit class for exactly ONE row of data.

    Args:
        model: Trained model object.
        sample_data: Single-row DataFrame (after preprocessing).
        target_column: Path to the ``36_label_classes.pkl`` file.
        model_name: Name of the model (used for logging only).

    Returns:
        tuple: ``(prediction_class_index, prediction_class_name, class_probabilities, confidence_score)``
            - class_probabilities: dict mapping each class label to its probability.
            - confidence_score: probability of the predicted class (0.0–1.0).

    Raises:
        ValueError: If ``model`` is ``None`` or ``sample_data`` has != 1 row.
    """
    if model is None:
        raise ValueError("Model is None. Check model loading.")

    if isinstance(sample_data, pd.Series):
        sample_data = sample_data.to_frame().T

    if len(sample_data) != 1:
        raise ValueError(f"predict_single_sample_data supports ONE row only; got {len(sample_data)}")

    label_classes = load_features(target_column)

    # Use predict_proba() as the single source of truth when available.
    # This ensures the predicted class always matches the highest probability.
    class_probabilities: dict[str, float] = {}
    confidence_score: float = 0.0
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample_data)
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("model", None), "predict_proba"):
            from sklearn.pipeline import Pipeline as SkPipeline

            preprocess_steps = [
                (name, step) for name, step in model.named_steps.items() if name != "model"
            ]
            preprocess_pipeline = SkPipeline(preprocess_steps)
            X_transformed = preprocess_pipeline.transform(sample_data)
            proba = model.named_steps["model"].predict_proba(X_transformed)
    except Exception as exc:
        logger.warning("Could not extract probabilities (model=%s): %s", model_name, exc)

    if proba is not None:
        prob_array = proba[0]
        prediction_class_index = int(np.argmax(prob_array))
        prediction_class_name = label_classes[prediction_class_index]
        class_probabilities = {label_classes[i]: float(prob_array[i]) for i in range(len(label_classes))}
        confidence_score = float(prob_array[prediction_class_index])
    else:
        # Fallback to model.predict() when probabilities are unavailable
        prediction = model.predict(sample_data)
        prediction_class_index = int(np.argmax(prediction))
        prediction_class_name = label_classes[prediction_class_index]

    logger.info(
        "Prediction: index=%d, class=%s, confidence=%.3f (model=%s)",
        prediction_class_index, prediction_class_name, confidence_score, model_name,
    )
    return prediction_class_index, prediction_class_name, class_probabilities, confidence_score


def build_contribution_table(sample_data: pd.DataFrame, shap_values, pred_class_index: int) -> pd.DataFrame:
    """Build a feature contribution table sorted by absolute SHAP value.

    Supports:
    - CatBoost SHAP (list of per-class arrays)
    - ``shap.Explanation`` objects (XGBoost / Random Forest, 2D or 3D)

    Args:
        sample_data: Single-row DataFrame (after preprocessing).
        shap_values: SHAP values — either a list (CatBoost) or a
            ``shap.Explanation`` with a ``.values`` ndarray.
        pred_class_index: Index of the predicted class (used for multiclass).

    Returns:
        pd.DataFrame: Columns ``["Feature", "SHAP Value", "Feature Value"]``
        sorted descending by ``|SHAP Value|``.

    Raises:
        ValueError: If lengths of features, SHAP values, and feature values differ.
        TypeError: If ``shap_values`` type is not supported.
    """
    if isinstance(sample_data, pd.Series):
        sample_data = sample_data.to_frame().T

    feature_names = list(sample_data.columns)

    if isinstance(shap_values, list):
        # CatBoost returns a list of per-class 2D arrays
        class_shap_values = shap_values[pred_class_index][0]
    elif hasattr(shap_values, "values"):
        values = shap_values.values
        if values.ndim == 3:  # multiclass: (samples, features, classes)
            class_shap_values = values[0, :, pred_class_index]
        elif values.ndim == 2:  # binary: (samples, features)
            class_shap_values = values[0]
        else:
            raise ValueError(f"Unsupported SHAP ndarray shape: {values.shape}")
    else:
        raise TypeError(f"Unsupported shap_values type: {type(shap_values)}")

    class_shap_values = np.asarray(class_shap_values).ravel()
    feature_values = sample_data.iloc[0].to_numpy().ravel()

    if not (len(feature_names) == len(class_shap_values) == len(feature_values)):
        raise ValueError(
            f"Length mismatch: features={len(feature_names)}, "
            f"shap={len(class_shap_values)}, values={len(feature_values)}"
        )

    return (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP Value": class_shap_values,
                "Feature Value": feature_values,
            }
        )
        .sort_values(by="SHAP Value", key=lambda x: x.abs(), ascending=False)
        .reset_index(drop=True)
    )


def generate_shap_value_summary_plots(model, X_shap, feature_names: list, model_name: str):
    """Compute SHAP values and save summary plots + JSON per class.

    Args:
        model: Trained model (the ``"model"`` step from the sklearn Pipeline).
        X_shap: Preprocessed feature matrix (numpy array or DataFrame).
        feature_names: Ordered list of feature column names.
        model_name: Used in output file names.

    Returns:
        tuple: ``(explainer, shap_per_class)`` where ``shap_per_class`` is a
        list of 2D ``np.ndarray`` with shape ``(num_samples, num_features)``.
    """
    explainer = shap.TreeExplainer(model)
    shap_sample = X_shap[: min(config.hyperparams.get("inference", {}).get("shap_max_samples", 100), X_shap.shape[0])]

    shap_values = explainer.shap_values(shap_sample)

    if isinstance(shap_values, list):
        shap_per_class = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_per_class = [shap_values[:, :, cls] for cls in range(shap_values.shape[2])]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_per_class = [shap_values]
    else:
        raise ValueError(
            f"Unsupported SHAP output: type={type(shap_values)}, shape={getattr(shap_values, 'shape', None)}"
        )

    for class_idx, class_shap in enumerate(shap_per_class):
        shap_values_dict = [
            {"feature": feature_names[idx], "shap_value": float(value)}
            for idx, feature_col in enumerate(feature_names)
            for value in class_shap[:, idx]
        ]

        shap_json_path = f"{config.output_dir}/shap_values_{model_name}__class_{class_idx}.json"
        with open(shap_json_path, "w") as f:
            json.dump(shap_values_dict, f, indent=4)

        with contextlib.suppress(Exception):
            mlflow.log_artifact(shap_json_path)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(class_shap, shap_sample, feature_names=feature_names, plot_type="bar", show=False)

        shap_plot_path = f"{config.output_dir}/shap_summary_{model_name}__class_{class_idx}.png"
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()

        with contextlib.suppress(Exception):
            mlflow.log_artifact(shap_plot_path)

    return explainer, shap_per_class
