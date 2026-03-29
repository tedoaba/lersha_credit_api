"""Model registration documentation for the Lersha Credit Scoring System.

This script contains no executable code. It documents the training-time
MLflow Model Registry registration pattern for operators and data scientists.

WHEN TO USE:
    Run the registration code below at the end of a training notebook or
    training script, after fitting your pipeline, to register the model in
    the MLflow Model Registry.

TRAINING-TIME REGISTRATION PATTERN:
    import mlflow.sklearn

    with mlflow.start_run():
        # ... train your pipeline ...

        mlflow.sklearn.log_model(
            pipeline,                                          # fitted sklearn Pipeline
            artifact_path="model",                             # artifact subdirectory
            registered_model_name=f"lersha-{model_name}",     # e.g. "lersha-xgboost"
        )
        # Supported model_name values: "xgboost", "random_forest", "catboost"

PROMOTING TO PRODUCTION:
    After logging, the model lands in the registry at the "None" stage.
    To promote it:
      1. Open the MLflow UI: http://mlflow:5000 (dev) or https://your-domain.com/mlflow
      2. Navigate to: Models → lersha-{model_name}
      3. Click the version you want to promote
      4. Transition: None → Staging → Production

    Once a version is in the "Production" stage, the prediction service
    (backend/core/infer_utils.py:load_prediction_models) will automatically
    load it from the registry on next startup.

FALLBACK BEHAVIOUR:
    If the MLflow registry is unreachable at startup (network issue, MLflow
    server down, model not promoted yet), the prediction service gracefully
    falls back to loading from the configured local .pkl path:
      - xgboost:      config.xgb_model_36  (XGB_MODEL_36 env var)
      - random_forest: config.rf_model_36  (RF_MODEL_36 env var)
      - catboost:     config.cab_model_36  (CAB_MODEL_36 env var)

    A WARNING log is emitted when the fallback path is taken, so operators
    can monitor via log aggregation for unexpected registry unavailability.
"""
