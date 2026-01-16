import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st


def plot_distribution(df, column):
    fig = px.histogram(df, x=column, nbins=40, title=f"Distribution of {column}")
    st.plotly_chart(fig, width='stretch')


def plot_correlation(df):
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    st.plotly_chart(fig, width='stretch')


def plot_missing_values(df):
    missing = df.isna().sum().sort_values(ascending=False)
    fig = px.bar(
        missing[missing > 0],
        title="Missing Values per Feature",
        labels={"value": "Missing Count", "index": "Feature"}
    )
    st.plotly_chart(fig, width='stretch')


def plot_histograms(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if len(numerical_cols) == 0:
        st.warning("No numerical columns found.")
        return

    for col in numerical_cols:
        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            marginal="box",
            title=f"Histogram of {col}",
            color_discrete_sequence=["#636EFA"]
        )
        st.plotly_chart(fig, width='content')


def plot_boxplots(df, target_col=None):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numerical_cols:
        if target_col and target_col in df.columns:
            fig = px.box(
                df,
                x=target_col,
                y=col,
                points="suspectedoutliers",
                title=f"Boxplot of {col} by {target_col}",
                color=target_col
            )
        else:
            fig = px.box(
                df,
                y=col,
                points="suspectedoutliers",
                title=f"Boxplot of {col}"
            )

        st.plotly_chart(fig, width='content')


def plot_pairplot(df):
    cols_to_drop = ['rented_farm_land', 'own_farmland_size', 'farm_mechanization', 'total_farmland_size', 'flaw', 'family_farmland_size', 'hasrusacco', 'haslocaledir', 'totalfamilymembers', 
     'holdsleadershiprole', 'agricultureexperience', 'agriculturalcertificate', 'hasmemberofmicrofinance', 
     'hascooperativeassociation', 'hascommunityhealthinsurance', 'estimated_cost', 'estimated_income', 'estimated_expenses',
       'estimated_income_another_farm']
    
    df = df.drop(columns=cols_to_drop)
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if len(numerical_cols) < 2:
        st.warning("Not enough numerical columns for pair plots.")
        return

    fig = px.scatter_matrix(
        df[numerical_cols],
        title="Interactive Pairplot",
        dimensions=numerical_cols,
        color=numerical_cols[0]
    )

    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig, width='content')


def plot_categorical_distributions(df):
    df = df.drop(columns=['farmer_uid'], errors='ignore')
    categorical_cols = df.select_dtypes(include=['object']).columns

    if len(categorical_cols) == 0:
        st.warning("No categorical columns found.")
        return

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, col in enumerate(categorical_cols):
        with cols[i % 3]:
            fig = px.histogram(
                df,
                y=col,
                color=col,
                title=f"Distribution of {col}",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            st.plotly_chart(fig, width='content')


def plot_categorical_vs_target(df, target):
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        fig = px.histogram(
            df,
            x=col,
            color=target,
            barmode="group",
            title=f"{col} by {target}",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig, width='content')


def correlation_heatmap(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if len(numerical_cols) < 2:
        st.warning("Not enough numerical columns for correlation matrix.")
        return

    corr = df[numerical_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    st.plotly_chart(fig, width='content')
