import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def show_eda_page(dataframe,eda_results):
    """Display EDA results."""

    st.subheader("📊 Exploratory Data Analysis")

    # ---------------- NUMERICAL ----------------
    num_count = eda_results["num_count"]
    numerical_columns = eda_results["columns"]
    desc_stats = eda_results["stats"]

    st.subheader("📈 Numerical Features")
    st.write(f"**Number of Numerical Features:** {num_count}")
    st.write("**Numerical Feature Names:**")
    st.write(numerical_columns)
    st.dataframe(desc_stats)

    # ---------------- CATEGORICAL ----------------
    categorical_columns = eda_results["categorical_columns"]
    cat_unique_counts = eda_results["cat_unique_counts"]
    cat_most_frequent = eda_results["cat_most_frequent"]

    st.subheader("📋 Categorical Features")
    st.write(f"**Number of Categorical Features:** {len(categorical_columns)}")
    st.write("**List of Categorical Features:**")
    st.write(categorical_columns)

    st.write("**Unique Counts:**")
    st.dataframe(cat_unique_counts)

    st.write("**Most Frequent Values:**")
    st.dataframe(cat_most_frequent)

    # ---------------- MISSING VALUES ----------------
    total_missing_values = eda_results["total_missing"]
    missing_values_per_col = eda_results["missing_counts"]
    missing_percentage = eda_results["missing_percentage"]

    st.subheader("❌ Missing Values Analysis")
    st.write(f"**Total Missing Values:** {total_missing_values}")

    st.write("**Missing Values Per Column:**")
    st.dataframe(missing_values_per_col)

    st.write("**Missing Percentage Per Column:**")
    st.dataframe(missing_percentage)

    # ---------------- CORRELATION ----------------
    st.subheader("🔗 Correlation Analysis")

    correlation_matrix = eda_results["correlation_matrix"]

    if correlation_matrix is not None:
        st.write("**Correlation Matrix:**")
        st.dataframe(correlation_matrix)
    else:
        st.info("Not enough numerical columns to compute correlation.")
    
    # ---------------- VISUALIZATIONS ----------------

    st.subheader("📊 Data Visualizations")

    # Histograms
    st.markdown("### 📈 Histograms")

    for column in numerical_columns:

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(dataframe[column].dropna(), bins=20)

        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)

    # Box Plots
    st.markdown("### 📦 Box Plots")

    for column in numerical_columns:

        fig, ax = plt.subplots(figsize=(6, 2))

        ax.boxplot(dataframe[column].dropna(), vert=False)

        ax.set_title(f"Box Plot of {column}")

        st.pyplot(fig)
        plt.close(fig)

    # ---------------- CORRELATION HEATMAP ----------------

    st.subheader("🔥 Correlation Heatmap")

    correlation_matrix = eda_results["correlation_matrix"]

    if correlation_matrix is not None:

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )

        ax.set_title("Feature Correlation Heatmap")

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Not enough numerical columns to generate heatmap.")