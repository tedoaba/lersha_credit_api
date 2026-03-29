"""Feature engineering transformations for agricultural credit scoring.

This module is intentionally dependency-light: it uses only numpy and pandas.
No ML framework imports (joblib, xgboost, shap, etc.) are permitted here,
enabling fast unit testing without model artifacts.

Usage:
    from backend.core.feature_engineering import apply_feature_engineering
    engineered_df = apply_feature_engineering(raw_farmer_df)
"""
import numpy as np
import pandas as pd

from backend.logger.logger import get_logger

logger = get_logger(__name__)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature transformations to raw farmer data.

    Computes derived financial, agricultural, and demographic features,
    then drops the source columns that were used in the computation.

    Args:
        df: Raw farmer DataFrame from the database. Expected to contain the
            columns listed in the 'columns to drop' list below.

    Returns:
        DataFrame with engineered features. The output has 36 columns
        matching ``config.columns_36`` (minus 'decision' which is the target).

    Raises:
        KeyError: If a required source column is missing from ``df``.
    """
    df = df.copy()

    # ── Financial features ─────────────────────────────────────────────────
    df["total_estimated_income"] = df["estimated_income"] + df["estimated_income_another_farm"]
    df["total_estimated_cost"] = df["estimated_expenses"] + df["estimated_cost"]
    df["net_income"] = np.round(df["total_estimated_income"] - df["total_estimated_cost"], 3)

    df["income_per_family_member"] = np.round(
        df["total_estimated_income"] / df["family_size"],
        3,
    )

    # ── Agricultural features ──────────────────────────────────────────────
    df["agriculture_experience"] = np.round(np.log1p(df["agricultureexperience"]), 3)

    df["institutional_support_score"] = (
        df["hasmemberofmicrofinance"]
        + df["hascooperativeassociation"]
        + df["agriculturalcertificate"]
        + df["hascommunityhealthinsurance"]
    )

    df["yield_per_hectare"] = np.round(df["expectedyieldquintals"] / df["farmsizehectares"], 3)

    df["input_intensity"] = np.round(
        (df["seedquintals"] + df["ureafertilizerquintals"] + df["dapnpsfertilizerquintals"])
        / df["farmsizehectares"],
        3,
    )

    # ── Demographic features ───────────────────────────────────────────────
    try:
        df["age_group"] = pd.qcut(
            df["age"],
            q=4,
            labels=["Young", "Early_Middle", "Late_Middle", "Senior"],
            duplicates="drop",
        )
    except ValueError:
        # Fallback for DataFrames too small for quartile-based binning (e.g. single row)
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 40, 50, 200],
            labels=["Young", "Early_Middle", "Late_Middle", "Senior"],
            include_lowest=True,
        )

    # ── Drop source columns consumed by the derived features ─────────────
    columns_to_drop = [
        "age", "value_chain", "estimated_cost", "estimated_income",
        "estimated_expenses", "estimated_income_another_farm",
        "total_farmland_size", "land_size", "childrenunder12",
        "elderlymembersover60", "agricultureexperience",
        "agriculturalcertificate", "hasmemberofmicrofinance",
        "hascooperativeassociation", "hascommunityhealthinsurance",
        "maincrops", "lastyearaverageprice",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    logger.info("Feature engineering complete. Output shape: %s", df.shape)
    return df
