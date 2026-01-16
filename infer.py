from src.inference_pipeline import match_inputs, run_inferences
from config.config import config
from src.logger import get_logger

logger = get_logger(__name__)


def infer(source, farmer_uid=None, number_of_rows=None):
    logger.info("Inference Initialized...")

    if source == "Single Value":
        farmer_uid = "Farmer UID"
        if farmer_uid:
            original_df, selected_df_34 = match_inputs(source=source, filters=farmer_uid)


    elif source == "Batch Prediction":
        original_df, selected_df_34 = match_inputs(source=source, number_of_rows=number_of_rows)
        logger.info(f"Fetched {number_of_rows} rows from database for evaluation.")

    result_xgboost = run_inferences(model_name="xgboost", original_data=original_df, selected_data=selected_df_34, feature_column=config.feature_column_34, target_column=config.target_column_34)
    result_random_forest = run_inferences(model_name="random_forest", original_data=original_df, selected_data=selected_df_34, feature_column=config.feature_column_34, target_column=config.target_column_34)

    logger.info(f"Batch Evaluation Completed! Records processed: {result_xgboost['records_processed']}")

    evaluations_xgboost = result_xgboost["evaluations"]
    evaluations_random_forest = result_random_forest["evaluations"]

    for i, eval_row in enumerate(evaluations_xgboost):
        logger.info("Predicted Class: %s", eval_row['predicted_class_name'])
        logger.info("Result Explanation: %s\n")
        logger.info(eval_row['rag_explanation'])

    for i, eval_row in enumerate(evaluations_random_forest):
        logger.info("Predicted Class: %s", eval_row['predicted_class_name'])
        logger.info("Result Explanation: %s\n")
        logger.info(eval_row['rag_explanation'])

    return result_xgboost, result_random_forest


if __name__ == "__main__":
    infer(source="Batch Prediction", number_of_rows=3)