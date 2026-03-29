"""Unit tests for backend/core/preprocessing.py."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.core.preprocessing import load_features, preprocessing_categorical_features, replace_inf


@pytest.fixture
def feature_pkl(tmp_path) -> str:
    """Write a canonical feature list to a temp .pkl file and return its path."""
    feature_columns = ["gender_Male", "age_group_Early_Middle", "net_income", "family_size"]
    pkl_path = tmp_path / "test_features.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(feature_columns, f)
    return str(pkl_path)


def test_load_features_returns_list(feature_pkl):
    """load_features should return a list of strings."""
    result = load_features(feature_pkl)
    assert isinstance(result, list)
    assert len(result) == 4


def test_load_features_missing_file():
    """load_features raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_features("/nonexistent/path/to/file.pkl")


def test_preprocessing_output_columns_match_pkl(feature_pkl):
    """After encoding, columns should match the canonical feature list exactly."""
    df = pd.DataFrame(
        {
            "gender": ["Male"],
            "age_group": ["Early_Middle"],
            "net_income": [12000.0],
            "family_size": [5],
        }
    )
    result = preprocessing_categorical_features(df, feature_pkl)
    canonical = load_features(feature_pkl)
    assert list(result.columns) == canonical


def test_missing_columns_filled_with_zero(feature_pkl):
    """Columns in the canonical list but absent from input should be filled with 0."""
    df = pd.DataFrame(
        {
            "net_income": [5000.0],
            "family_size": [3],
            # gender and age_group are missing → should be filled as 0
        }
    )
    result = preprocessing_categorical_features(df, feature_pkl)
    assert result["gender_Male"].iloc[0] == 0
    assert result["age_group_Early_Middle"].iloc[0] == 0


def test_output_shape_equals_feature_list(feature_pkl):
    """Output DataFrame should have exactly len(feature_columns) columns."""
    df = pd.DataFrame(
        {
            "gender": ["Female"],
            "age_group": ["Young"],
            "net_income": [8000.0],
            "family_size": [4],
        }
    )
    result = preprocessing_categorical_features(df, feature_pkl)
    canonical = load_features(feature_pkl)
    assert result.shape[1] == len(canonical)


def test_replace_inf_removes_positive_inf():
    """replace_inf should convert np.inf to NaN."""
    df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [np.inf, 2.0, -np.inf]})
    result = replace_inf(df)
    assert not np.any(np.isinf(result.select_dtypes(include=[float]).values))


def test_replace_inf_preserves_normal_values():
    """replace_inf should not alter finite values."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = replace_inf(df)
    assert list(result["a"]) == [1.0, 2.0, 3.0]
