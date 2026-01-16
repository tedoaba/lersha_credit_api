import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.logger import get_logger
from config.config import config

logger = get_logger(__name__)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def get_sample_data(csv_path, n_rows=3):
    df = pd.read_csv(csv_path, nrows=n_rows)
    sample = df.iloc[[0]]
    sample_data = sample.drop(columns=[config.id_column], errors='ignore')
    logger.info("Sample data extracted for prediction with shape:\n%s", sample_data.shape)

    return sample_data


def get_random_sample_data(csv_path, n_rows=3):
    df = pd.read_csv(csv_path)
    sample = df.sample(n=n_rows, replace=False, random_state=None)
    sample_data = sample.drop(columns=[config.id_column], errors='ignore')

    logger.info("Sample data extracted for prediction with shape: %s", sample_data.shape)
    return sample_data


def get_sample_df(df):
    # df = df.iloc[[0]]
    sample_data = df.drop(columns=[config.id_column], errors='ignore')
    logger.info("Sample data extracted for prediction with shape:\n%s", sample_data.shape)

    return sample_data

def get_candidate_data(df, columns_to_select):
    sample_data = df.drop(columns=[config.id_column], errors='ignore')
    sample_data = sample_data[columns_to_select]

    logger.info("Sample data extracted for prediction with shape:\n%s", sample_data.shape)

    return sample_data

def merge_dataset_on_farmer_uid(csv_path_1, csv_path_2, output_path, how='inner'):
    """
    Merge two DataFrames on the 'farmer_uid' column.

    Parameters:
        csv_path_1 (str): First dataset path
        csv_path_2 (str): Second dataset path
        how (str): Type of merge ('inner', 'left', 'right', 'outer')

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    duplicate_cols = [col for col in df1.columns if col in df2.columns and col != 'farmer_uid']
    logger.info(df1.shape)
    logger.info(df1.columns)
    logger.info(df2.shape)
    logger.info(df2.columns)

    df2 = df2.drop(columns=duplicate_cols)

    merged_df = pd.merge(df1, df2, on='farmer_uid', how=how)
    merged_df.to_csv(output_path, index=False)
    logger.info(merged_df.columns)
    logger.info(f"data merged and saved on {output_path} Successfully!")
    logger.info(duplicate_cols)
    return merged_df


def get_test_data(input_path, output_path):
    df = pd.read_csv(input_path)
    test_data = df.iloc[:500]

    test_data = test_data.drop(columns=['decision'])
    test_data.to_csv(output_path, index=False)
    logger.info(test_data.shape)
    logger.info(f"Successfully saved test data to {output_path}")
    return test_data


def apply_one_hot_encoding(features):

    categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()

    features_encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

    joblib.dump(features_encoded.columns.tolist(), config.feature_column_18)

    logger.info("Feature columns after one-hot encoding:")
    logger.info(features_encoded.columns)
    logger.info(features_encoded.shape)

    return features_encoded


def load_model(model_path: str):
    model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
    return model
