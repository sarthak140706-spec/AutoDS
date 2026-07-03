import streamlit as st
import pandas as pd


def show_preview(dataframe):
    """Display a preview of the uploaded dataset."""

    st.subheader("📄 Dataset Preview")

    rows, cols = dataframe.shape

    st.write(f"**Rows:** {rows} | **Columns:** {cols}")

    st.dataframe(dataframe.head())

    st.subheader("📋 Dataset Information")

    info_df = pd.DataFrame({
        "Column Name": dataframe.columns,
        "Data Type": dataframe.dtypes.astype(str)
    })
    st.dataframe(info_df)
    
    st.subheader("📊 Dataset Summary")

    total_missing = dataframe.isnull().sum().sum()
    total_duplicates = dataframe.duplicated().sum()
    memory_usage = round(dataframe.memory_usage(deep=True).sum() / 1024, 2)

    summary_df = pd.DataFrame({
        "Metric": [
            "Missing Values",
            "Duplicate Rows",
            "Memory Usage (KB)"
        ],
        "Value": [
            total_missing,
            total_duplicates,
            memory_usage
        ]
    })

    st.dataframe(summary_df)