import streamlit as st
from typing import List, Dict

st.set_page_config(
    page_title="Introduction",
    layout="centered",
)

st.title("🌱 Lersha Credit Scoring System")

st.markdown("""
<style>
/* Light Mode */
@media (prefers-color-scheme: light) {
    .card {
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 20px;
        background-color: #fafafa;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        color: #000000;
    }
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    .card {
        border: 1px solid #444444;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 20px;
        background-color: #1e1e1e;
        box-shadow: 0 2px 6px rgba(255,255,255,0.07);
        color: #ffffff;
    }
    .card h3 {
        color: #ffffff !important;
    }
    .card p, .card ul, .card li {
        color: #dddddd !important;
    }
}
.card h3 {
    margin-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1.1, 1.1])

with col1:
    st.markdown("""
    <div class="card">
        <h3>📊 Interactive Data Visualizations</h3>
        <p>Understand customer profiles and risk indicators through dynamic charts and visual summaries designed to reveal insights at a glance.</p>
    </div>

    <div class="card">
        <h3>📈 Live Prediction Dashboard</h3>
        <ul>
            <li>Track individual and batch predictions</li>
            <li>Visualize risk distributions</li>
            <li>Export results for reporting and decision-making</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


with col2:
    st.markdown("""
    <div class="card">
        <h3>🤖 Real-Time Credit Predictions</h3>
        <p>Upload customer data (CSV) or retrieve records from the database to generate instant, ML-powered credit scores using Lersha’s model.</p>
    </div>

    <div class="card">
        <h3>🗄️ Fully Integrated Database</h3>
        <p>All predictions, customer details, and model outputs are securely stored in a production-ready database which ensures accuracy and collaboration for analysts and field agents.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="card">
        <h3>🧪 Purpose</h3>
        <p>Built to accelerate credit evaluations, strengthen risk assessment, and support sustainable agricultural financing in Ethiopia.</p>
    </div>
        """, unsafe_allow_html=True)


st.info("This platform uses a replica of Lersha's expert scorecard model in machine learning.")


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
