"""New Prediction page — submit and poll an inference job via the backend API.

All backend communication is handled exclusively through LershaAPIClient.
No imports from backend.*, config.*, or src.* are permitted in this file.
"""

import pandas as pd
import requests
import streamlit as st

from ui.utils.api_client import LershaAPIClient

st.set_page_config(page_title="New Prediction", layout="wide")

client = LershaAPIClient()

st.title("New Prediction")

# ── Data Source Selection ───────────────────────────────────────────────────

source = st.radio(
    "Select Data Source",
    ["Single Value", "Batch Prediction"],
    index=0,
    horizontal=True,
)

farmer_uid: str | None = None
number_of_rows: int | None = None

if source == "Single Value":
    farmer_uid = st.text_input("Farmer UID") or None
    st.info("Enter a Farmer UID above. Data will be fetched from the backend on job submission.")

elif source == "Batch Prediction":
    number_of_rows = int(st.number_input("Number of rows to process", min_value=1, max_value=100, step=1))
    st.info(
        f"Batch prediction will process {number_of_rows} randomly selected farmer record(s) "
        "from the database via the backend API."
    )

# ── Prediction Submission ───────────────────────────────────────────────────

if st.button("Run Prediction", type="primary"):
    # Validate inputs before submission
    if source == "Single Value" and not farmer_uid:
        st.warning("Please enter a Farmer UID before running a Single Value prediction.")
        st.stop()

    st.write("---")

    try:
        # Step 1: Submit the job
        with st.spinner("Submitting inference job…"):
            response = client.submit_prediction(
                source=source,
                farmer_uid=farmer_uid,
                number_of_rows=number_of_rows,
            )

        job_id: str = response["job_id"]
        st.success(f"✅ Job accepted — ID: `{job_id}`")

        # Step 2: Poll until complete
        with st.spinner("Running inference… this may take up to 5 minutes."):
            job = client.poll_until_complete(job_id, poll_interval=2.0, max_wait=300.0)

    except requests.exceptions.ConnectionError:
        st.error(
            "🔌 Backend unavailable. Is the API server running? "
            "Start it with `make api` or `uvicorn backend.main:app --reload --port 8000`."
        )
        st.stop()
    except TimeoutError:
        st.error("⏱ Inference timed out after 5 minutes. The job may still be running — check the Dashboard later.")
        st.stop()
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            st.error("🔐 Authentication failed. Check that API_KEY matches the backend configuration.")
        else:
            st.error(f"⚠ API error: {exc}")
        st.stop()

    # ── Result Rendering ────────────────────────────────────────────────────

    if job["status"] == "failed":
        st.error(f"❌ Inference failed: {job.get('error', 'Unknown error')}")
        st.stop()

    result: dict = job.get("result") or {}
    xgb_result: dict = result.get("result_xgboost", {})
    rf_result: dict = result.get("result_random_forest", {})

    records_processed = xgb_result.get("records_processed", 0)
    st.success(f"✅ Batch Evaluation Completed! Records processed: {records_processed}")
    st.write("---")

    evaluations_xgboost: list = xgb_result.get("evaluations", [])
    evaluations_random_forest: list = rf_result.get("evaluations", [])

    st.header("Evaluation Output")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Result with XGBoost Model")
        if not evaluations_xgboost:
            st.info("No XGBoost evaluation results available.")
        for i, eval_row in enumerate(evaluations_xgboost):
            with st.expander(
                f"Record #{i + 1} — Prediction with {eval_row.get('model_name', 'xgboost')}: "
                f"{eval_row.get('predicted_class_name', 'N/A')}"
            ):
                st.subheader("Prediction")
                st.write(f"**Model Name**: {eval_row.get('model_name', 'xgboost')}")
                st.write(f"**Predicted Class**: {eval_row.get('predicted_class_name', 'N/A')}")

                st.subheader("Top Feature Contributions")
                contributions = eval_row.get("top_feature_contributions", [])
                if contributions:
                    contrib_df = pd.DataFrame(contributions)
                    st.table(contrib_df)
                else:
                    st.write("No feature contributions available.")

                st.subheader("Explanation")
                st.write(eval_row.get("rag_explanation", "No explanation available."))

    with col2:
        st.subheader("Result with Random Forest Model")
        if not evaluations_random_forest:
            st.info("No Random Forest evaluation results available.")
        for i, eval_row in enumerate(evaluations_random_forest):
            with st.expander(
                f"Record #{i + 1} — Prediction with {eval_row.get('model_name', 'random_forest')}: "
                f"{eval_row.get('predicted_class_name', 'N/A')}"
            ):
                st.subheader("Prediction")
                st.write(f"**Model Name**: {eval_row.get('model_name', 'random_forest')}")
                st.write(f"**Predicted Class**: {eval_row.get('predicted_class_name', 'N/A')}")

                st.subheader("Top Feature Contributions")
                contributions = eval_row.get("top_feature_contributions", [])
                if contributions:
                    contrib_df = pd.DataFrame(contributions)
                    st.table(contrib_df)
                else:
                    st.write("No feature contributions available.")

                st.subheader("Explanation")
                st.write(eval_row.get("rag_explanation", "No explanation available."))

    st.write("---")


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
