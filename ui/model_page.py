import streamlit as st
import pandas as pd


def show_model_page(models_dict, evaluation_results, y_test):

    st.subheader("🤖 Model Comparison Dashboard")

    st.success("Multi-model training completed successfully.")

    # ---------------- MODEL DETAILS ----------------
    st.subheader("📌 Individual Model Results")

    for model_name, model_data in models_dict.items():

        st.markdown(f"### {model_name}")

        st.write("**First 5 Predictions:**")
        st.write(model_data["predictions"][:5])

        st.write("**First 5 Actual Values:**")
        st.write(y_test[:5])

        st.markdown("---")

    # ---------------- COMPARISON TABLE ----------------
    st.subheader("📊 Model Performance Comparison")

    comparison_data = []

    for model_name, metrics in evaluation_results.items():

        row = {"Model": model_name}

        row.update(metrics)

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    st.dataframe(comparison_df)

    # ---------------- BEST MODEL SELECTION ----------------
    st.subheader("🏆 Best Model")

    metric_to_compare = None

    # Detect metric type (Regression vs Classification)
    first_model_metrics = list(evaluation_results.values())[0]

    if "R2 Score" in first_model_metrics:
        metric_to_compare = "R2 Score"
        best_model = comparison_df.loc[comparison_df[metric_to_compare].idxmax()]
    else:
        metric_to_compare = "Accuracy"
        best_model = comparison_df.loc[comparison_df[metric_to_compare].idxmax()]

    st.success(f"Best Model: {best_model['Model']}")
    st.write(f"**{metric_to_compare}:** {best_model[metric_to_compare]}")