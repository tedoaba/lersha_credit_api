import streamlit as st
import pandas as pd

from src.inference_pipeline import match_inputs, run_inferences
from config.config import config

st.set_page_config(page_title="New Prediction", layout="wide")

st.title("New Prediction")

source = st.radio(
    "Select Data Source",
    ["Single Value", "Batch Prediction"],
    index=0,
    horizontal=True
)

original_df = None


if source == "Single Value":
    filters = st.text_input("Farmer UID")
    if filters:
        original_df, selected_df_34 = match_inputs(source=source, filters=filters)
        st.dataframe(original_df.head())


elif source == "Batch Prediction":
    st.info("Evaluation will run using the CSV file defined inside your backend config.")
    number_of_rows = st.number_input("Number of rows to process", min_value=1, max_value=10, step=1)
    original_df, selected_df_34 = match_inputs(source=source, number_of_rows=number_of_rows)
    st.success(f"Fetched {number_of_rows} rows from database for evaluation.")
    st.dataframe(original_df.head())


if st.button("Run Prediction"):
    st.info("Running evaluation… please wait.")

    try:
        result_xgboost = run_inferences(model_name="xgboost", original_data=original_df, selected_data=selected_df_34, feature_column=config.feature_column_34, target_column=config.target_column_34)
        result_random_forest = run_inferences(model_name="random_forest", original_data=original_df, selected_data=selected_df_34, feature_column=config.feature_column_34, target_column=config.target_column_34)

        st.success(f"Batch Evaluation Completed! Records processed: {result_xgboost['records_processed']}")
        st.write("---")

        evaluations_xgboost = result_xgboost["evaluations"]
        evaluations_random_forest = result_random_forest["evaluations"]

        st.header("Evaluation Output")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Result with Xgboost Model")
            for i, eval_row in enumerate(evaluations_xgboost):
                with st.expander(f"Record #{i+1} — Prediction with {eval_row['model_name']}: {eval_row['predicted_class_name']}"):
                    
                    st.subheader("Prediction")
                    st.write(f"**Model Name**: {eval_row['model_name']}")
                    st.write(f"**Predicted Class**: {eval_row['predicted_class_name']}")

                    st.subheader("Top Feature Contributions")
                    contrib_df = pd.DataFrame(eval_row["top_feature_contributions"])
                    st.table(contrib_df)

                    st.subheader("Explanation")
                    st.write(eval_row["rag_explanation"])

        with col2:
            st.subheader("Result with Random Forest Model")
            for i, eval_row in enumerate(evaluations_random_forest):
                with st.expander(f"Record #{i+1} — Prediction with {eval_row['model_name']}: {eval_row['predicted_class_name']}"):
                    
                    st.subheader("Prediction")
                    st.write(f"**Model Name**: {eval_row['model_name']}")
                    st.write(f"**Predicted Class**: {eval_row['predicted_class_name']}")

                    st.subheader("Top Feature Contributions")
                    contrib_df = pd.DataFrame(eval_row["top_feature_contributions"])
                    st.table(contrib_df)

                    st.subheader("Explanation")
                    st.write(eval_row["rag_explanation"])

        st.write("---")

    except Exception as e:
        st.error(f"Error during evaluation: {e}")


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
