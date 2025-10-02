import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page config
st.set_page_config(page_title="📊 EDA App For E-Commerce Dataset", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Upload your dataset and explore insights step by step.")

# Title
st.title("📊 Exploratory Data Analysis (EDA) Dashboard")
st.markdown("Built for quick data insights.")

# Upload dataset
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset Info
    st.header("📌 Dataset Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))

    # Data types
    st.subheader("🔍 Column Data Types")
    st.write(df.dtypes)

    # Preview
    st.subheader("👀 Preview of Data")
    st.dataframe(df.head(10))

    # Missing Values
    st.header("🧹 Data Quality Check")
    missing_info = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2)
    })
    st.write(missing_info)

    # Duplicates
    st.write("**Duplicate Rows:**", df.duplicated().sum())

    # Summary Statistics
    st.header("📈 Summary Statistics")
    st.write(df.describe(include="all").T)

    # Correlation Heatmap (with column selection)
    st.header("🔥 Correlation Analysis")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        selected_corr_cols = st.multiselect(
            "Select numeric columns for correlation", 
            numeric_df.columns, 
            default=numeric_df.columns[:5]
        )
        if selected_corr_cols:
            corr = numeric_df[selected_corr_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
    else:
        st.info("No numeric columns available for correlation heatmap.")

    # Column-wise Analysis
    st.header("📊 Column-wise Analysis")
    column = st.selectbox("Select a column to analyze", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(df[column].describe())
        
        # Histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[column], kde=True, bins=20, ax=ax, color="skyblue")
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)

        # Boxplot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[column], ax=ax, color="orange")
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)

    else:
        value_counts = df[column].value_counts()
        if len(value_counts) > 20:
            top_values = value_counts[:20]
            top_values["Other"] = value_counts[20:].sum()
            plot_data = top_values
        else:
            plot_data = value_counts

        st.write(plot_data)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=plot_data.index, y=plot_data.values, ax=ax, palette="Set2")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(f"Count plot of {column}")
        st.pyplot(fig)

    # Multivariate Analysis (lightweight alternative to pairplot)
    st.header("📌 Multivariate Analysis")
    if not numeric_df.empty:
        cols = st.multiselect("Select two numeric columns for scatter plot", numeric_df.columns, default=numeric_df.columns[:2])
        if len(cols) == 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=df[cols[0]], y=df[cols[1]], alpha=0.6, ax=ax)
            ax.set_title(f"Scatter Plot: {cols[0]} vs {cols[1]}")
            st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for scatter plot.")

    # Outlier Detection (improved per-column)
    st.header("🚨 Outlier Detection")
    if not numeric_df.empty:
        z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        outlier_counts = (z_scores > 3).sum()
        st.write("**Potential outliers per column (|Z| > 3):**")
        st.write(outlier_counts[outlier_counts > 0])
        st.write(f"**Total outliers across all numeric columns:** {outlier_counts.sum()}")

else:
    st.warning("👆 Upload a CSV file from the sidebar to begin.")
