"""Unit tests for backend/core/feature_engineering.py.

These tests are intentionally dependency-light — no ML models, no DB,
no environment variables required. They verify pure data transformation logic.
"""

import numpy as np
import pandas as pd
import pytest

from backend.core.feature_engineering import apply_feature_engineering


@pytest.fixture
def raw_farmer_row() -> pd.DataFrame:
    """Return a single-row DataFrame with all raw source columns."""
    return pd.DataFrame(
        {
            "farmer_uid": ["F-001"],
            "gender": ["Male"],
            "age": [32],
            "family_size": [5],
            "estimated_income": [12000.0],
            "estimated_income_another_farm": [2000.0],
            "estimated_expenses": [4000.0],
            "estimated_cost": [2000.0],
            "agricultureexperience": [8],
            "hasmemberofmicrofinance": [1],
            "hascooperativeassociation": [0],
            "agriculturalcertificate": [1],
            "hascommunityhealthinsurance": [1],
            "farmsizehectares": [2.5],
            "expectedyieldquintals": [30.0],
            "seedquintals": [3.0],
            "ureafertilizerquintals": [1.5],
            "dapnpsfertilizerquintals": [1.0],
            "value_chain": ["maize"],
            "total_farmland_size": [3.0],
            "land_size": [2.5],
            "childrenunder12": [2],
            "elderlymembersover60": [1],
            "maincrops": ["maize"],
            "lastyearaverageprice": [500.0],
            "decision": ["Eligible"],
        }
    )


def test_net_income_formula(raw_farmer_row):
    """net_income = (income + income_another) - (expenses + cost)."""
    result = apply_feature_engineering(raw_farmer_row)
    expected = round((12000.0 + 2000.0) - (4000.0 + 2000.0), 3)
    assert result["net_income"].iloc[0] == pytest.approx(expected, rel=1e-3)


def test_institutional_support_score(raw_farmer_row):
    """institutional_support_score is the sum of the 4 binary flags."""
    result = apply_feature_engineering(raw_farmer_row)
    # 1 + 0 + 1 + 1 = 3
    assert result["institutional_support_score"].iloc[0] == 3


def test_dropped_columns_absent(raw_farmer_row):
    """Source columns used in feature derivation must be dropped from output."""
    columns_to_drop = [
        "age",
        "value_chain",
        "estimated_cost",
        "estimated_income",
        "estimated_expenses",
        "estimated_income_another_farm",
        "total_farmland_size",
        "land_size",
        "childrenunder12",
        "elderlymembersover60",
        "agricultureexperience",
        "agriculturalcertificate",
        "hasmemberofmicrofinance",
        "hascooperativeassociation",
        "hascommunityhealthinsurance",
        "maincrops",
        "lastyearaverageprice",
    ]
    result = apply_feature_engineering(raw_farmer_row)
    for col in columns_to_drop:
        assert col not in result.columns, f"Column '{col}' should have been dropped"


def test_agriculture_experience_log_transform(raw_farmer_row):
    """agriculture_experience = round(log1p(raw_value), 3)."""
    result = apply_feature_engineering(raw_farmer_row)
    expected = round(np.log1p(8), 3)
    assert result["agriculture_experience"].iloc[0] == pytest.approx(expected, rel=1e-3)


def test_age_group_fallback_single_row(raw_farmer_row):
    """fallback pd.cut is used when qcut fails on insufficient data."""
    result = apply_feature_engineering(raw_farmer_row)
    # Should produce a valid age_group without raising
    assert "age_group" in result.columns
    assert result["age_group"].iloc[0] in ["Young", "Early_Middle", "Late_Middle", "Senior"]


def test_age_group_binning_two_row_edge_case():
    """age_group binning must complete without error on a 2-row DataFrame.

    With only 2 distinct age values ``pd.qcut`` (4 quantiles) raises a
    ValueError; the implementation must fall back to ``pd.cut`` with fixed
    bins and return a valid categorical column with no NaN values.
    """
    df = pd.DataFrame(
        {
            "farmer_uid": ["F-001", "F-002"],
            "gender": ["Male", "Female"],
            "age": [25, 55],  # two distinct ages — triggers qcut ValueError
            "family_size": [4, 6],
            "estimated_income": [10000.0, 14000.0],
            "estimated_income_another_farm": [1000.0, 2000.0],
            "estimated_expenses": [3000.0, 5000.0],
            "estimated_cost": [1500.0, 2500.0],
            "agricultureexperience": [5, 18],
            "hasmemberofmicrofinance": [1, 0],
            "hascooperativeassociation": [0, 1],
            "agriculturalcertificate": [1, 1],
            "hascommunityhealthinsurance": [1, 0],
            "farmsizehectares": [2.0, 3.5],
            "expectedyieldquintals": [24.0, 42.0],
            "seedquintals": [2.5, 4.5],
            "ureafertilizerquintals": [1.0, 2.0],
            "dapnpsfertilizerquintals": [0.8, 1.5],
            "value_chain": ["maize", "wheat"],
            "total_farmland_size": [2.5, 4.0],
            "land_size": [2.0, 3.5],
            "childrenunder12": [1, 3],
            "elderlymembersover60": [0, 1],
            "maincrops": ["maize", "wheat"],
            "lastyearaverageprice": [480.0, 610.0],
            "decision": ["Eligible", "Review"],
        }
    )
    result = apply_feature_engineering(df)
    assert "age_group" in result.columns, "age_group column must be present"
    assert result["age_group"].isna().sum() == 0, "age_group must not contain NaN"
    valid_labels = {"Young", "Early_Middle", "Late_Middle", "Senior"}
    assert set(result["age_group"].astype(str)) <= valid_labels


def test_yield_per_hectare(raw_farmer_row):
    """yield_per_hectare = round(expectedyieldquintals / farmsizehectares, 3)."""
    result = apply_feature_engineering(raw_farmer_row)
    expected = round(30.0 / 2.5, 3)
    assert result["yield_per_hectare"].iloc[0] == pytest.approx(expected, rel=1e-3)


def test_input_intensity(raw_farmer_row):
    """input_intensity = round((seeds + urea + dap) / farmsize, 3)."""
    result = apply_feature_engineering(raw_farmer_row)
    expected = round((3.0 + 1.5 + 1.0) / 2.5, 3)
    assert result["input_intensity"].iloc[0] == pytest.approx(expected, rel=1e-3)


def test_total_estimated_income(raw_farmer_row):
    """total_estimated_income = estimated_income + estimated_income_another_farm."""
    result = apply_feature_engineering(raw_farmer_row)
    assert result["total_estimated_income"].iloc[0] == 14000.0


def test_income_per_family_member(raw_farmer_row):
    """income_per_family_member = round(total_income / family_size, 3)."""
    result = apply_feature_engineering(raw_farmer_row)
    expected = round(14000.0 / 5, 3)
    assert result["income_per_family_member"].iloc[0] == pytest.approx(expected, rel=1e-3)
