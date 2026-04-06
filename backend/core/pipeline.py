"""Inference pipeline orchestration.

Handles data fetching, model loading, SHAP computation, and result assembly.
RAG explanation is live — the commented-out import and placeholder string
have been replaced with real ``get_rag_explanation`` calls.
"""

from __future__ import annotations

from datetime import datetime

import mlflow
from sklearn.pipeline import Pipeline as SkPipeline

from backend.chat.rag_engine import get_rag_explanation
from backend.config.config import config
from backend.core.feature_engineering import apply_feature_engineering
from backend.core.infer_utils import (
    build_contribution_table,
    generate_shap_value_summary_plots,
    get_candidate_data,
    load_prediction_models,
    predict_single_sample_data,
)
from backend.core.preprocessing import preprocessing_categorical_features
from backend.logger.logger import get_logger
from backend.services.db_utils import fetch_multiple_raw_data, fetch_raw_data, save_batch_evaluations

logger = get_logger(__name__)


def match_inputs(
    source: str | None = None,
    filters: str | None = None,
    number_of_rows: int | None = None,
    *,
    gender: str | None = None,
    age_min: int | None = None,
    age_max: int | None = None,
):
    """Fetch and engineer the input data for inference.

    Args:
        source: ``"Single Value"`` or ``"Batch Prediction"``.
        filters: ``farmer_uid`` value for single-value requests.
        number_of_rows: Number of rows for batch requests.
        gender: Optional gender filter for batch requests.
        age_min: Optional minimum age filter for batch requests.
        age_max: Optional maximum age filter for batch requests.

    Returns:
        tuple: ``(original_data: pd.DataFrame, selected_data_36: pd.DataFrame)``

    Raises:
        ValueError: If ``source`` is not a recognised value.
    """
    if source == "Single Value":
        original_data = fetch_raw_data(table_name=config.farmer_data_all, filters=filters)
        initial_data_36 = apply_feature_engineering(original_data)
        selected_data_36 = get_candidate_data(initial_data_36, config.columns_36)
    elif source == "Batch Prediction":
        original_data = fetch_multiple_raw_data(
            table_name=config.farmer_data_all,
            n_rows=number_of_rows,
            gender=gender,
            age_min=age_min,
            age_max=age_max,
        )
        initial_data_36 = apply_feature_engineering(original_data)
        selected_data_36 = get_candidate_data(initial_data_36, config.columns_36)
    else:
        raise ValueError(f"Invalid source '{source}'. Expected 'Single Value' or 'Batch Prediction'.")

    return original_data, selected_data_36


def run_inferences(
    model_name: str,
    original_data,
    selected_data,
    feature_column: str,
    target_column: str,
) -> dict:
    """Run end-to-end inference for all rows in ``selected_data``.

    Iterates row-by-row, computes predictions and SHAP contributions,
    generates a RAG explanation per prediction, and saves results to the DB.

    Args:
        model_name: One of ``"xgboost"``, ``"random_forest"``, ``"catboost"``.
        original_data: Raw DataFrame with farmer identity columns.
        selected_data: Engineered 36-feature DataFrame.
        feature_column: Path to ``36_feature_columns.pkl``.
        target_column: Path to ``36_label_classes.pkl``.

    Returns:
        dict: ``{"status": ..., "records_processed": int, "evaluations": list}``
    """
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    logger.info("Starting inference run — model: %s, rows: %d", model_name, len(selected_data))
    run_name = f"{model_name}_inference_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("run_type", config.inference_tag)
        mlflow.log_param("records_processed", len(selected_data))
        mlflow.log_param("num_columns", selected_data.shape[1])

        sample_encoded = preprocessing_categorical_features(selected_data, feature_column)
        model = load_prediction_models(model_name)
        evaluation_results = []

        for idx in range(len(sample_encoded)):
            row = sample_encoded.iloc[[idx]]

            prediction_class_index, prediction_class_name = predict_single_sample_data(
                model, row, target_column, model_name
            )
            mlflow.log_metric(f"prediction_{idx}", float(prediction_class_index))
            mlflow.log_param(f"prediction_class_name_{idx}", prediction_class_name)

            shap_values = None
            if model_name != "catboost":
                preprocess_steps = [
                    (name, step) for name, step in model.named_steps.items() if name in ("inf_cleaner", "imputer")
                ]
                preprocess_pipeline = SkPipeline(preprocess_steps)
                X_shap = preprocess_pipeline.transform(row)
                feature_names = (
                    row.columns.tolist()
                    if hasattr(row, "columns")
                    else [f"feature_{i}" for i in range(X_shap.shape[1])]
                )
                _, shap_values = generate_shap_value_summary_plots(
                    model.named_steps["model"], row, feature_names, model_name
                )

            contribution_table = build_contribution_table(row, shap_values, prediction_class_index)
            top10 = contribution_table.head(10)
            top10_list = [{"feature": str(r["Feature"]), "value": float(r["SHAP Value"])} for _, r in top10.iterrows()]
            shap_dict = {r["Feature"]: float(r["SHAP Value"]) for _, r in top10.iterrows()}

            # RAG explanation — FIXED: was a literal placeholder string "rag_explanation"
            try:
                rag_explanation = get_rag_explanation(prediction_class_name, shap_dict, model_name=model_name)
            except Exception as rag_exc:
                logger.warning("RAG explanation failed for record %d: %s", idx, rag_exc)
                rag_explanation = "[RAG unavailable]"

            evaluation_results.append(
                {
                    "predicted_class_name": str(prediction_class_name),
                    "top_feature_contributions": top10_list,
                    "rag_explanation": rag_explanation,
                    "model_name": model_name,
                }
            )

        save_batch_evaluations(original_data, evaluation_results)
        logger.info("Inference complete — model: %s, records: %d", model_name, len(selected_data))

        return {
            "status": "batch_evaluation_completed",
            "records_processed": len(selected_data),
            "evaluations": evaluation_results,
        }
