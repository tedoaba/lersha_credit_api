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
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

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


def get_featured_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ID column and return all remaining feature columns.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without the farmer UID column.
    """
    sample_data = df.drop(columns=[config.id_column], errors="ignore")
    logger.info("Featured data shape: %s", sample_data.shape)
    return sample_data


def load_prediction_models(model_name: str):
    """Load a trained prediction model by name.

    Supported model names: ``"xgboost"``, ``"random_forest"``, ``"catboost"``.

    Args:
        model_name: Identifier for the model to load.

    Returns:
        Loaded model object (sklearn Pipeline or CatBoostClassifier).

    Raises:
        ValueError: If ``model_name`` is not recognised.
        RuntimeError: If the model file cannot be loaded (wraps the original exception).
    """
    try:
        if model_name == "xgboost":
            model = joblib.load(config.xgb_model_36)
        elif model_name == "random_forest":
            model = joblib.load(config.rf_model_36)
        elif model_name == "catboost":
            model = joblib.load(config.cab_model_36)
        else:
            raise ValueError(f"Unknown model_name '{model_name}'. Expected: xgboost | random_forest | catboost")

        mlflow.set_tag("model_source", model_name)
        logger.info("Model '%s' loaded successfully", model_name)
        return model
    except ValueError:
        raise
    except Exception as e:
        logger.error("Error loading model '%s': %s", model_name, str(e))
        raise RuntimeError(f"Failed to load model '{model_name}'") from e


def xgb_model_evaluation(model, X, y, dataset_name: str = "Dataset"):
    """Evaluate an XGBoost model and log metrics to the logger.

    Args:
        model: Trained model with ``predict`` and ``predict_proba`` methods.
        X: Feature matrix.
        y: True target labels.
        dataset_name: Label for log messages (e.g. ``"Train"`` or ``"Test"``).

    Returns:
        tuple: ``(y_pred, accuracy, f1, roc_auc)``
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    roc_auc = roc_auc_score(y, model.predict_proba(X), multi_class="ovr")

    logger.info("%s — Accuracy: %.4f | F1: %.4f | ROC-AUC: %.4f", dataset_name, accuracy, f1, roc_auc)
    return y_pred, accuracy, f1, roc_auc


def generate_classification_report(y_true, y_pred) -> dict:
    """Return sklearn classification report as a dictionary.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        dict: Classification report from sklearn.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    logger.info("Classification Report generated")
    return report


def log_model_reports(report: dict) -> None:
    """Log per-class metrics to MLflow.

    Args:
        report: Dictionary from ``generate_classification_report``.
    """
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
    logger.info("Model reports logged to MLflow")


def generate_confusion_matrix(y_true, y_pred):
    """Compute and log the confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        np.ndarray: Confusion matrix.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    logger.info("Confusion Matrix:\n%s", conf_matrix)
    return conf_matrix


def define_shap_explainer(model, X_encoded):
    """Create a SHAP Explainer and compute SHAP values.

    Args:
        model: Trained model.
        X_encoded: Encoded feature matrix.

    Returns:
        tuple: ``(explainer, shap_values)``
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_encoded)
    logger.info("SHAP explainer created")
    return explainer, shap_values


def predict_single_sample_data(model, sample_data: pd.DataFrame, target_column: str, model_name: str):
    """Predict the credit class for exactly ONE row of data.

    Args:
        model: Trained model object.
        sample_data: Single-row DataFrame (after preprocessing).
        target_column: Path to the ``36_label_classes.pkl`` file.
        model_name: Name of the model (used for logging only).

    Returns:
        tuple: ``(prediction_class_index: int, prediction_class_name: str)``

    Raises:
        ValueError: If ``model`` is ``None`` or ``sample_data`` has != 1 row.
    """
    if model is None:
        raise ValueError("Model is None. Check model loading.")

    if isinstance(sample_data, pd.Series):
        sample_data = sample_data.to_frame().T

    if len(sample_data) != 1:
        raise ValueError(f"predict_single_sample_data supports ONE row only; got {len(sample_data)}")

    prediction = model.predict(sample_data)
    prediction_class_index = int(np.argmax(prediction))

    label_classes = load_features(target_column)
    prediction_class_name = label_classes[prediction_class_index]

    logger.info("Prediction: index=%d, class=%s (model=%s)", prediction_class_index, prediction_class_name, model_name)
    return prediction_class_index, prediction_class_name


def generate_shap_for_sample(sample_data: pd.DataFrame, shap_values, model_name: str) -> None:
    """Save a SHAP summary bar plot for a single sample prediction.

    Args:
        sample_data: Single-row DataFrame used for prediction.
        shap_values: SHAP Explanation object from the explainer.
        model_name: Used when constructing the output file path.
    """
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values.values, sample_data, plot_type="bar", show=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shap_path = f"{config.shap_path}_{model_name}_single_sample_{timestamp}.png"

    plt.savefig(shap_path, bbox_inches="tight")
    plt.close()
    logger.info("SHAP sample plot saved: %s", shap_path)

    with contextlib.suppress(Exception):
        mlflow.log_artifact(shap_path)


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
        pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": class_shap_values,
            "Feature Value": feature_values,
        })
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
        raise ValueError(f"Unsupported SHAP output: type={type(shap_values)}, shape={getattr(shap_values, 'shape', None)}")

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
