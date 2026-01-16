import streamlit as st
import pandas as pd

from src.inference_pipeline import match_inputs, run_predictions
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
        result_34 = run_predictions(model_name="model_34", original_data=original_df, selected_data=selected_df_34, feature_column=config.feature_column_34, target_column=config.target_column_34)

        st.success(f"Batch Evaluation Completed! Records processed: {result_34['records_processed']}")
        st.write("---")

        evaluations_34 = result_34["evaluations"]

        st.header("Evaluation Output")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Result with 18 Data Points")
            for i, eval_row in enumerate(evaluations_34):
                with st.expander(f"Record #{i+1} — Prediction with {eval_row['model_name']}: {eval_row['predicted_class_name']}"):
                    
                    st.subheader("Prediction")
                    st.write(f"**Model Name**: {eval_row['model_name']}")
                    st.write(f"**Predicted Class**: {eval_row['predicted_class_name']}")

                    st.subheader("Top Feature Contributions (SHAP)")
                    contrib_df = pd.DataFrame(eval_row["top_feature_contributions"])
                    st.table(contrib_df)

                    st.subheader("RAG Explanation")
                    st.write(eval_row["rag_explanation"])

        with col2:
            st.subheader("Result with 44 Data Points")
            for i, eval_row in enumerate(evaluations_34):
                with st.expander(f"Record #{i+1} — Prediction with {eval_row['model_name']}: {eval_row['predicted_class_name']}"):
                    
                    st.subheader("Prediction")
                    st.write(f"**Model Name**: {eval_row['model_name']}")
                    st.write(f"**Predicted Class**: {eval_row['predicted_class_name']}")

                    st.subheader("Top Feature Contributions (SHAP)")
                    contrib_df = pd.DataFrame(eval_row["top_feature_contributions"])
                    st.table(contrib_df)

                    st.subheader("RAG Explanation")
                    st.write(eval_row["rag_explanation"])

        with col3:
            st.subheader("Result with 44 Data Points")
            for i, eval_row in enumerate(evaluations_34):
                with st.expander(f"Record #{i+1} — Prediction with {eval_row['model_name']}: {eval_row['predicted_class_name']}"):
                    
                    st.subheader("Prediction")
                    st.write(f"**Model Name**: {eval_row['model_name']}")
                    st.write(f"**Predicted Class**: {eval_row['predicted_class_name']}")

                    st.subheader("Top Feature Contributions (SHAP)")
                    contrib_df = pd.DataFrame(eval_row["top_feature_contributions"])
                    st.table(contrib_df)

                    st.subheader("RAG Explanation")
                    st.write(eval_row["rag_explanation"])

        with col4:
            st.subheader("Result with Feature Engineered Data Points")
            for i, eval_row in enumerate(evaluations_34):
                with st.expander(f"Record #{i+1} — Prediction with {eval_row['model_name']}: {eval_row['predicted_class_name']}"):
                    
                    st.subheader("Prediction")
                    st.write(f"**Model Name**: {eval_row['model_name']}")
                    st.write(f"**Predicted Class**: {eval_row['predicted_class_name']}")

                    st.subheader("Top Feature Contributions (SHAP)")
                    contrib_df = pd.DataFrame(eval_row["top_feature_contributions"])
                    st.table(contrib_df)

                    st.subheader("RAG Explanation")
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
