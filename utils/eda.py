import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text

from config.config import config


@st.cache_data
def get_dataset_from_local(csv_path):
    return pd.read_csv(csv_path)

@st.cache_data(show_spinner=True, ttl=300)
def load_table(table_name):
    """Load table data from the database with caching."""
    try:
        engine = create_engine(config.db_uri)
        query = text(f"SELECT * FROM {table_name}")

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        return df

    except Exception as e:
        st.error(f"Failed to load data from database: {e}")
        return pd.DataFrame()


def style_decision(val):
    """
    Returns CSS style rules for decision highlighting.
    Automatically adjusts text color for dark/light themes.
    """
    color_map = {
        "Not Eligible": "#690707",   # red
        "Review":       "#8f7408",   # yellow
        "Eligible":     "#03450d",   # green
    }
    
    bg = color_map.get(val, "transparent")

    text = "black"
    if val in ["Not Eligible", "Eligible"]:  
        text = "white"

    return f"background-color: {bg}; color:{text}; font-weight:600; border-radius:6px;"


def missing_values(df):
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Values": missing, "Percentage": percent})
    missing_df = missing_df[missing_df["Missing Values"] > 0]

    st.dataframe(missing_df)


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:

        df[col] = df[col].apply(
            lambda x: x if isinstance(x, (int, float, bool, np.number, type(None))) else str(x)
        )

        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("object") 

    return df


def summarize_dataset(df: pd.DataFrame):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isna().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "memory_usage": df.memory_usage(deep=True).sum() / (1024**2),
    }


def numeric_categorization(df):
    numeric = df.select_dtypes(include="number").columns.tolist()
    categoricals = df.select_dtypes(exclude="number").columns.tolist()
    return numeric, categoricals
