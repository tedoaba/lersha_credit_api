import shap
import mlflow
import joblib
import pickle
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import Pool

import numpy as np
import json

from config.config import config
from src.logger import get_logger

logger = get_logger(__name__)


def get_candidate_data(df, columns_to_select):
    sample_data = df.drop(columns=[config.id_column], errors='ignore')
    sample_data = sample_data[columns_to_select]

    logger.info("Sample data extracted for prediction with shape:\n%s", sample_data.shape)

    return sample_data


def get_featured_data(df):
    sample_data = df.drop(columns=[config.id_column], errors='ignore')

    logger.info("Sample data extracted for prediction with shape:\n%s", sample_data.shape)

    return sample_data


def load_features(feature_path: str):

    with open(feature_path, "rb") as f:
        feature_columns = pickle.load(f)

    return feature_columns

def preprocessing_categorical_features(data, feature_columns):
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    feature_columns = load_features(feature_columns)
    data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)
    logger.info("Categorical features one-hot encoded. New shape: %s", data_encoded.shape)
    return data_encoded


def load_prediction_model(model_name: str):
    try:
        if model_name == "model_18":
            model = joblib.load(config.xgb_model_18)
        elif model_name == "model_44":
            model = joblib.load(config.xgb_model_44)
        elif model_name == "model_36":
            model = joblib.load(config.xgb_model_36)
        elif model_name == "feature_engineered_model":
            model = joblib.load(config.xgb_engineered_model)
        else:
            model = None
        
        mlflow.set_tag("model_source", model_name)
        logger.info("Model loaded from %s", model_name)
        return model
    except Exception as e:
        logger.error("Error loading model %s: %s", model_name, str(e))
        raise RuntimeError(f"Failed to load model {model_name}") from e


def load_prediction_models(model_name: str):
    try:
        if model_name == "xgboost":
            model = joblib.load(config.xgb_model_36)
        elif model_name == "random_forest":
            model = joblib.load(config.rf_model_36)
        elif model_name == "catboost":
            model = joblib.load(config.cab_model_36)
        else:
            model = None
        
        mlflow.set_tag("model_source", model_name)
        logger.info("Model loaded from %s", model_name)
        return model
    except Exception as e:
        logger.error("Error loading model %s: %s", model_name, str(e))
        raise RuntimeError(f"Failed to load model {model_name}") from e


def xgb_model_evaluation(model, X, y, dataset_name="Dataset"):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    roc_auc = roc_auc_score(y, model.predict_proba(X), multi_class='ovr')

    logger.info("%s Evaluation:", dataset_name)
    logger.info("Accuracy: %.4f", accuracy)
    logger.info("F1 Score: %.4f", f1)
    logger.info("ROC AUC Score: %.4f", roc_auc)

    return y_pred, accuracy, f1, roc_auc


def generate_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    logger.info("Classification Report:\n%s", report)
    return report


def log_model_reports(report):
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for m, value in metrics.items():
                mlflow.log_metric(f"{label}_{m}", value)

    logger.info("Model reports logged.")


def generate_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    logger.info("Confusion Matrix:\n%s", conf_matrix)
    return conf_matrix


def define_shap_explainer(model, X_encoded):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_encoded)
    logger.info("SHAP explainer defined for the model.")
    return explainer, shap_values


def generate_shap_summary(model, X_encoded):
    explainer = shap.Explainer(model)
    shap_sample = X_encoded[:100]
    shap_values = explainer(shap_sample)

    for cls in range(shap_values.values.shape[2]):
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values.values[:, :, cls], shap_sample, plot_type="bar", show=False)
        shap_path = f"{config.shap_path}_xgb_{cls}.png"
        plt.savefig(shap_path, bbox_inches="tight")
        logger.info("SHAP summary plot saved at %s", shap_path)
        mlflow.log_artifact(shap_path)

    logger.info("SHAP summary plots generated and logged.")


def predict_single_sample(model, sample_data, target_column):
    sample_data = xgb.DMatrix(sample_data)
    prediction_class_index = model.predict(sample_data)[0]
    target_column = load_features(target_column)
    prediction_class_name = target_column[prediction_class_index]
    logger.info("Prediction for the sample data: %s", prediction_class_index)
    logger.info("Predicted class name: %s", prediction_class_name)
    return prediction_class_index, prediction_class_name


def predict_single_sample_data(model, sample_data, target_column, model_name):
    """
    Predict class for exactly ONE row of data.
    """

    if model is None:
        raise ValueError("Model is None. Check model loading.")

    if isinstance(sample_data, pd.Series):
        sample_data = sample_data.to_frame().T

    if len(sample_data) != 1:
        raise ValueError("This function supports ONLY a single row of data.")

    prediction = model.predict(sample_data)
    prediction_class_index = int(np.argmax(prediction))

    target_column = load_features(target_column)
    prediction_class_name = target_column[prediction_class_index]

    logger.info("Prediction index: %s", prediction_class_index)
    logger.info("Predicted class name: %s", prediction_class_name)

    return prediction_class_index, prediction_class_name


def generate_shap_for_sample(sample_data, shap_values, model_name):

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values.values, sample_data, plot_type="bar", show=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shap_path = f"{config.shap_path}_{model_name}_single_sample_{timestamp}.png"

    plt.savefig(shap_path, bbox_inches="tight")
    logger.info("SHAP summary plot for single sample saved at %s", shap_path)
    mlflow.log_artifact(shap_path)

    logger.info("SHAP summary plot for single sample generated and logged.")


def build_contribution_tables(sample_data, shap_values, pred_class_index):

    contribution_table = pd.DataFrame({
        'Feature': sample_data.columns,
        'SHAP Value': shap_values.values[0, :, pred_class_index],
        'Feature Value': sample_data.iloc[0].values
    }).sort_values(by='SHAP Value', key=abs, ascending=False)

    return contribution_table


def build_contribution_table(sample_data, shap_values, pred_class_index):
    """
    Build contribution table for a single sample.
    Supports:
    - CatBoost SHAP (list of per-class arrays)
    - shap.Explanation (XGB / RF)
    """

    if isinstance(sample_data, pd.Series):
        sample_data = sample_data.to_frame().T

    feature_names = list(sample_data.columns)

    if isinstance(shap_values, list):
        class_shap_values = shap_values[pred_class_index][0]
    elif hasattr(shap_values, "values"):
        values = shap_values.values
        if values.ndim == 3:  # multiclass
            class_shap_values = values[0, :, pred_class_index]
        elif values.ndim == 2:
            class_shap_values = values[0]
        else:
            raise ValueError(f"Unsupported SHAP shape: {values.shape}")
    else:
        raise TypeError(f"Unsupported shap_values type: {type(shap_values)}")

    class_shap_values = np.asarray(class_shap_values).ravel()
    feature_values = sample_data.iloc[0].to_numpy().ravel()

    if not (
        len(feature_names)
        == len(class_shap_values)
        == len(feature_values)
    ):
        raise ValueError(
            f"Length mismatch:\n"
            f"features={len(feature_names)}, "
            f"shap={len(class_shap_values)}, "
            f"values={len(feature_values)}"
        )

    contribution_table = (
        pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": class_shap_values,
            "Feature Value": feature_values,
        })
        .sort_values(by="SHAP Value", key=lambda x: x.abs(), ascending=False)
        .reset_index(drop=True)
    )

    return contribution_table

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    AGE_BINS = [18, 30, 40, 50, 65, 100]

    df['total_estimated_income'] = (df['estimated_income'] + df['estimated_income_another_farm']) 
    df['total_estimated_cost'] = (df['estimated_expenses'] + df['estimated_cost'])
    df['net_income'] = np.round((df['total_estimated_income'] - df['total_estimated_cost']), 3)

    df['income_per_family_member'] = np.round((
        df['total_estimated_income'] / df['family_size']
    ), 3)
    
    df['agriculture_experience'] = np.round(np.log1p(df['agricultureexperience']), 3)

    df['institutional_support_score'] = (
        df['hasmemberofmicrofinance']
        + df['hascooperativeassociation']
        + df['agriculturalcertificate']
        + df['hascommunityhealthinsurance']
    )

    df['yield_per_hectare'] = np.round((
        df['expectedyieldquintals'] / df['farmsizehectares']
    ), 3)

    df['input_intensity'] = np.round((
        (df['seedquintals']
        + df['ureafertilizerquintals']
        + df['dapnpsfertilizerquintals'])
        / df['farmsizehectares']
    ), 3) 

    try:
        df['age_group'] = pd.qcut(
            df['age'],
            q=4,
            labels=['Young', 'Early_Middle', 'Late_Middle', 'Senior']
        )
    except ValueError:
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 20, 35, 45, 55],
            labels=['Young', 'Early_Middle', 'Late_Middle', 'Senior']
        )

    columns_to_drop_after_feature_engineering = ['age', 'value_chain', 'estimated_cost', 'estimated_income', 'estimated_expenses',
        'estimated_income_another_farm', 'total_farmland_size', 'land_size', 'childrenunder12', 
        'elderlymembersover60', 'agricultureexperience', 'agriculturalcertificate', 'hasmemberofmicrofinance',
        'hascooperativeassociation', 'hascommunityhealthinsurance', 'maincrops', 'lastyearaverageprice']

    df = df.drop(columns=columns_to_drop_after_feature_engineering)

    return df

def generate_shap_value_summary_plotsss(model, X_shap, feature_names, model_name):
    explainer = shap.TreeExplainer(model)
    shap_sample = X_shap[: min(100, X_shap.shape[0])]

    shap_values = explainer.shap_values(shap_sample)

    if isinstance(shap_values, list):
        shap_per_class = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_per_class = [
            shap_values[:, :, class_idx]
            for class_idx in range(shap_values.shape[2])
        ]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_per_class = [shap_values]
    else:
        raise ValueError(
            f"Unsupported SHAP output shape: {type(shap_values)}, "
            f"shape={getattr(shap_values, 'shape', None)}"
        )

    for class_idx, class_shap in enumerate(shap_per_class):
        shap_values_dict = []

        for idx, feature_name in enumerate(feature_names):
            feature_shap = class_shap[:, idx]
            shap_values_dict.extend(
                {
                    "feature": feature_name,
                    "shap_value": float(value),
                }
                for value in feature_shap
            )

        shap_json_path = (
            f"{config.output_dir}/shap_values_"
            f"{model_name}__class_{class_idx}.json"
        )

        with open(shap_json_path, "w") as f:
            json.dump(shap_values_dict, f, indent=4)

        mlflow.log_artifact(shap_json_path)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            class_shap,
            shap_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )

        shap_plot_path = (
            f"{config.output_dir}/shap_summary_"
            f"{model_name}__class_{class_idx}.png"
        )
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(shap_plot_path)
    
    return explainer, shap_per_class

def generate_shap_value_summary_plots(model, X_shap, feature_names, model_name):
    """
    Compute SHAP values and save summary plots and JSON per class.
    Returns:
        explainer: SHAP explainer object
        shap_per_class: list of 2D np.ndarray, each shape (num_samples, num_features)
    """
    explainer = shap.TreeExplainer(model)
    shap_sample = X_shap[: min(100, X_shap.shape[0])]

    shap_values = explainer.shap_values(shap_sample)

    # --- Standardize to list of 2D arrays ---
    if isinstance(shap_values, list):
        shap_per_class = shap_values  # already a list
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_per_class = [
            shap_values[:, :, class_idx] for class_idx in range(shap_values.shape[2])
        ]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_per_class = [shap_values]
    else:
        raise ValueError(
            f"Unsupported SHAP output shape: {type(shap_values)}, "
            f"shape={getattr(shap_values, 'shape', None)}"
        )

    # --- Save JSON and summary plots ---
    for class_idx, class_shap in enumerate(shap_per_class):
        shap_values_dict = []
        for idx, feature_name in enumerate(feature_names):
            feature_shap = class_shap[:, idx]
            shap_values_dict.extend(
                {"feature": feature_name, "shap_value": float(value)}
                for value in feature_shap
            )

        shap_json_path = (
            f"{config.output_dir}/shap_values_{model_name}__class_{class_idx}.json"
        )
        with open(shap_json_path, "w") as f:
            json.dump(shap_values_dict, f, indent=4)

        # Optional MLflow logging
        try:
            mlflow.log_artifact(shap_json_path)
        except Exception:
            pass

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            class_shap,
            shap_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        shap_plot_path = f"{config.output_dir}/shap_summary_{model_name}__class_{class_idx}.png"
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()

        try:
            mlflow.log_artifact(shap_plot_path)
        except Exception:
            pass

    return explainer, shap_per_class
