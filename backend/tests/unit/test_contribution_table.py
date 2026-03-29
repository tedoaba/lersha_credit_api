"""Unit tests for build_contribution_table in backend/core/infer_utils.py."""
import numpy as np
import pandas as pd
import pytest

from backend.core.infer_utils import build_contribution_table


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    """Single-row sample DataFrame with 3 features."""
    return pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0], "feature_c": [3.0]})


def make_shap_explanation(values_2d: np.ndarray):
    """Wrap a 2D numpy array in a minimal shap.Explanation-like object."""

    class FakeExplanation:
        def __init__(self, v):
            self.values = v

    return FakeExplanation(values_2d)


def test_catboost_list_input(single_row_df):
    """CatBoost returns a list of per-class 2D arrays; class index 1 is selected."""
    # Simulate CatBoost: list of (1 sample, 3 features) arrays
    shap_values = [
        np.array([[0.1, -0.2, 0.05]]),  # class 0
        np.array([[0.4, 0.3, -0.1]]),   # class 1
    ]
    result = build_contribution_table(single_row_df, shap_values, pred_class_index=1)
    assert len(result) == 3
    assert "SHAP Value" in result.columns
    assert "Feature" in result.columns


def test_xgb_3d_multiclass(single_row_df):
    """XGBoost multiclass returns a 3D ndarray (samples, features, classes)."""
    shap_3d = np.zeros((1, 3, 3))
    shap_3d[0, :, 2] = [0.5, -0.3, 0.1]  # class 2 values
    explanation = make_shap_explanation(shap_3d)
    result = build_contribution_table(single_row_df, explanation, pred_class_index=2)
    assert len(result) == 3


def test_2d_binary_ndarray(single_row_df):
    """Binary classification returns a 2D ndarray (samples, features)."""
    shap_2d = np.array([[0.6, -0.4, 0.2]])
    explanation = make_shap_explanation(shap_2d)
    result = build_contribution_table(single_row_df, explanation, pred_class_index=0)
    assert len(result) == 3


def test_length_mismatch_raises_value_error():
    """Mismatched feature count between df and SHAP values raises ValueError."""
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})  # 2 features
    shap_wrong = make_shap_explanation(np.array([[0.1, 0.2, 0.3]]))  # 3 SHAP values
    with pytest.raises(ValueError, match="Length mismatch"):
        build_contribution_table(df, shap_wrong, pred_class_index=0)


def test_unsupported_type_raises_type_error(single_row_df):
    """Passing an unsupported shap_values type raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported shap_values type"):
        build_contribution_table(single_row_df, "not_a_valid_type", pred_class_index=0)


def test_sorted_descending_by_abs_shap(single_row_df):
    """Output must be sorted by absolute SHAP value, descending."""
    shap_2d = np.array([[-0.9, 0.1, 0.5]])  # abs: 0.9, 0.1, 0.5
    explanation = make_shap_explanation(shap_2d)
    result = build_contribution_table(single_row_df, explanation, pred_class_index=0)
    abs_vals = result["SHAP Value"].abs().tolist()
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_series_input_converted(single_row_df):
    """A pd.Series input is coerced to a single-row DataFrame."""
    shap_2d = np.array([[0.1, 0.2, 0.3]])
    explanation = make_shap_explanation(shap_2d)
    series_input = single_row_df.iloc[0]  # pd.Series
    result = build_contribution_table(series_input, explanation, pred_class_index=0)
    assert len(result) == 3
