"""Preprocessing utilities for the Lersha Credit Scoring backend.

Handles categorical one-hot encoding, infinite value replacement,
and feature column alignment. Like feature_engineering.py, this module
avoids ML framework imports so unit tests run without model artifacts.

Usage:
    from backend.core.preprocessing import preprocessing_categorical_features, load_features
"""
import pickle
from pathlib import Path

import pandas as pd

from backend.logger.logger import get_logger

logger = get_logger(__name__)


def load_features(feature_path: str) -> list:
    """Load a pickled list of feature names or label classes.

    Args:
        feature_path: Absolute or relative path to a ``.pkl`` file containing
            a list of strings (feature column names or label class names).

    Returns:
        list: The unpickled object (expected to be a list of strings).

    Raises:
        FileNotFoundError: If ``feature_path`` does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled.
    """
    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    with open(path, "rb") as f:
        feature_columns = pickle.load(f)

    logger.info("Loaded %d features from %s", len(feature_columns), feature_path)
    return feature_columns


def preprocessing_categorical_features(data: pd.DataFrame, feature_columns: str) -> pd.DataFrame:
    """One-hot encode categorical columns and align to the canonical feature list.

    Categorical and object-dtype columns are one-hot encoded with pandas
    ``get_dummies`` (``drop_first=True``). The result is then reindexed to
    exactly match the saved feature column list, filling any missing columns
    with ``0`` (handles unseen categories at inference time).

    Args:
        data: Input DataFrame (typically the output of ``apply_feature_engineering``).
        feature_columns: Path to the ``.pkl`` file containing the canonical
            ordered list of feature column names.

    Returns:
        pd.DataFrame: Encoded and aligned DataFrame ready for model inference.
    """
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    canonical_columns = load_features(feature_columns)
    data_encoded = data_encoded.reindex(columns=canonical_columns, fill_value=0)

    logger.info("Categorical encoding complete. Output shape: %s", data_encoded.shape)
    return data_encoded


def replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Replace positive and negative infinity values with NaN.

    This is a hygiene step applied before imputation to ensure infinite
    values introduced by division (e.g. ``yield_per_hectare`` when
    ``farmsizehectares == 0``) do not propagate to model predictions.

    Args:
        df: Input DataFrame that may contain infinite float values.

    Returns:
        pd.DataFrame: Same DataFrame with ``np.inf`` and ``-np.inf``
        replaced by ``pd.NA``.
    """
    import numpy as np  # local import keeps module-level deps clean

    result = df.replace([np.inf, -np.inf], pd.NA)
    inf_count = (df == np.inf).sum().sum() + (df == -np.inf).sum().sum()
    if inf_count > 0:
        logger.warning("Replaced %d infinite values with NaN", inf_count)
    return result
