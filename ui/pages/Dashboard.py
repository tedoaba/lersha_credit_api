"""Dashboard page — display historical credit scoring results via the backend API.

All backend communication is handled exclusively through LershaAPIClient.
No imports from backend.*, config.*, utils.eda, or src.* are permitted in this file.
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests
import streamlit as st

from ui.utils.api_client import LershaAPIClient

st.set_page_config(
    page_title="Dashboard",
    layout="wide",
)

client = LershaAPIClient()


# ── Inline helper (previously imported from utils.eda) ──────────────────────

def style_decision(val: str) -> str:
    """Return a CSS background-color style string based on the decision value.

    Args:
        val: The decision class name (e.g. ``"Eligible"``, ``"Review"``, ``"Not Eligible"``).

    Returns:
        str: CSS ``background-color`` property string for use with ``DataFrame.style.map``.
    """
    colors: dict[str, str] = {
        "Eligible": "background-color: #d4edda",
        "Review": "background-color: #fff3cd",
        "Not Eligible": "background-color: #f8d7da",
    }
    return colors.get(val, "")


# ── Data Loading ────────────────────────────────────────────────────────────

try:
    response = client.get_results(limit=500)
    df = pd.DataFrame(response.get("records", []))
except requests.exceptions.ConnectionError:
    st.error(
        "🔌 Backend unavailable. Is the API server running? "
        "Start it with `make api` or `uvicorn backend.main:app --reload --port 8000`."
    )
    st.stop()
except requests.exceptions.HTTPError as exc:
    if exc.response is not None and exc.response.status_code == 403:
        st.error("🔐 Authentication failed. Check that API_KEY matches the backend configuration.")
    else:
        st.error(f"⚠ API error: {exc}")
    st.stop()


# ── Page Layout ─────────────────────────────────────────────────────────────

st.markdown("## 📊 **Farmer Credit Worthiness Dashboard**")

if df.empty:
    st.info("No data available.")
    st.stop()

df = df.drop(columns=["id", "top_feature_contributions", "timestamp"], errors="ignore")

df = df.rename(columns={
    "farmer_uid": "Farmer ID",
    "age": "Age",
    "gender": "Gender",
    "predicted_class_name": "Decision",
    "rag_explanation": "RAG Explanation",
    "model_name": "Model Name",
})

cols = [c for c in df.columns if c != "Decision"] + ["Decision"]
df = df[cols]

st.markdown("---")

eligible_count = (df["Decision"] == "Eligible").sum()
review_count = (df["Decision"] == "Review").sum()
not_eligible_count = (df["Decision"] == "Not Eligible").sum()

col1, col2, col3 = st.columns(3)
col1.metric("✔ Eligible", eligible_count)
col2.metric("⚠ Review", review_count)
col3.metric("✖ Not Eligible", not_eligible_count)


st.markdown("---")

st.subheader("🔎 Filter & Search")
search_text = st.text_input("Search by Farmer ID or Decision:")

decision_filter = st.multiselect(
    "Filter by Decision:",
    options=df["Decision"].unique(),
    default=df["Decision"].unique(),
)


filtered_df = df.copy()

if search_text:
    search_text = search_text.lower()
    filtered_df = filtered_df[
        filtered_df.apply(lambda row: row.astype(str).str.lower().str.contains(search_text).any(), axis=1)
    ]

filtered_df = filtered_df[filtered_df["Decision"].isin(decision_filter)]


rows_per_page = 15
total_rows = len(filtered_df)
total_pages = max(1, (total_rows // rows_per_page) + (1 if total_rows % rows_per_page else 0))

page = st.number_input("Page:", min_value=1, max_value=total_pages, step=1)

start = (page - 1) * rows_per_page
end = start + rows_per_page

ILLEGAL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def clean_illegal_chars(val: object) -> object:
    """Remove control characters that are illegal in Excel/CSV exports.

    Args:
        val: Any cell value; non-strings are returned unchanged.

    Returns:
        Cleaned string, or the original value if not a string.
    """
    if isinstance(val, str):
        return ILLEGAL_CHARACTERS_RE.sub("", val)
    return val


page_df = filtered_df.iloc[start:end]
csv_bytes = page_df.to_csv(index=False).encode("utf-8")
clean_page_df = page_df.map(clean_illegal_chars)

excel_buffer = io.BytesIO()
clean_page_df.to_excel(excel_buffer, index=False)
excel_bytes = excel_buffer.getvalue()

st.markdown("---")
st.subheader("Farmers List")

styled_df = (
    page_df.style
        .map(style_decision, subset=["Decision"])
        .set_properties(
            **{"border": "1px solid rgba(150,150,150,0.2)", "border-radius": "4px", "width": "100%"}
        )
)


st.write(styled_df)

colA, colB = st.columns([1, 1])

with colA:
    st.download_button(
        label="⬇ Download CSV",
        data=csv_bytes,
        file_name="credit_scoring_data.csv",
        mime="text/csv",
    )

with colB:
    st.download_button(
        label="⬇ Download Excel",
        data=excel_bytes,
        file_name="credit_scoring_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption(f"Showing {len(page_df)} of {total_rows} records. Page {page}/{total_pages}.")


# ── Footer ──────────────────────────────────────────────────────────────────

footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    color: grey;
    padding: 10px;
}
</style>

<div class="footer">
    © 2025 Lersha — All Rights Reserved
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
