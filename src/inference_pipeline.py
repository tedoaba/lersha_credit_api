import mlflow
from datetime import datetime
from sklearn.pipeline import Pipeline as SkPipeline

from services.db_utils import fetch_multiple_raw_data, fetch_raw_data, save_batch_evaluations
from src.infer_utils import (
    preprocessing_categorical_features, get_candidate_data, load_prediction_models, predict_single_sample_data, 
    build_contribution_table, apply_feature_engineering,
    generate_shap_value_summary_plots
)
from config.config import config
from src.logger import get_logger
# from chat.rag_engine import get_rag_explanation

logger = get_logger(__name__)


def match_inputs(source=None, filters=None, number_of_rows=None):
    if source == "Single Value":
        original_data = fetch_raw_data(table_name=config.farmer_data_all, filters=filters)
        initial_data_34 = apply_feature_engineering(original_data)
        selected_data_34 = get_candidate_data(initial_data_34, config.columns_34)
    elif source == "Batch Prediction":
        # number_of_rows = 3
        original_data = fetch_multiple_raw_data(table_name=config.farmer_data_all, n_rows=number_of_rows)
        initial_data_34 = apply_feature_engineering(original_data)
        selected_data_34 = get_candidate_data(initial_data_34, config.columns_34)
    else:
        raise ValueError("Invalid data source. Choose 'database' or 'csv'.")

    return original_data, selected_data_34


def run_inferences(model_name, original_data, selected_data, feature_column, target_column):
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    logger.info(f"Running predictions using model: {model_name}")
    
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

            prediction_class_index, prediction_class_name = predict_single_sample_data(model, row, target_column, model_name)
            mlflow.log_metric(f"prediction_{idx}", float(prediction_class_index))
            mlflow.log_param(f"prediction_class_name_{idx}", prediction_class_name)

            if model_name == "catboost":
                pass
            else:
                preprocess_steps = [
                    (name, step)
                    for name, step in model.named_steps.items()
                    if name in ("inf_cleaner", "imputer")
                ]

                preprocess_pipeline = SkPipeline(preprocess_steps)
                X_shap = preprocess_pipeline.transform(row)
                
                feature_names = (
                    row.columns.tolist()
                    if hasattr(row, "columns")
                    else [f"feature_{i}" for i in range(X_shap.shape[1])]
                )

                _, shap_values = generate_shap_value_summary_plots(model.named_steps["model"], row, feature_names, model_name)

            contribution_table = build_contribution_table(row, shap_values, prediction_class_index)

            top10 = contribution_table.head(10)
            top10_list = [
                {"feature": str(r["Feature"]), "value": float(r["SHAP Value"])}
                for _, r in top10.iterrows()
            ]
            shap_dict = {r["Feature"]: float(r["SHAP Value"]) for _, r in top10.iterrows()}

            # rag_explanation = get_rag_explanation(prediction_class_name, shap_dict)

            evaluation_results.append({
                "predicted_class_name": str(prediction_class_name),
                "top_feature_contributions": top10_list,
                "rag_explanation": "rag_explanation",
                "model_name": model_name
            })

        save_batch_evaluations(original_data, evaluation_results)
        
        logger.info(f"Completed predictions using model: {model_name}")

        return {
            "status": "batch_evaluation_completed",
            "records_processed": len(selected_data),
            "evaluations": evaluation_results
        }
