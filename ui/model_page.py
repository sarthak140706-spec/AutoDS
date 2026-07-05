import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


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

    # ---------------- DETERMINE METRIC ----------------
    first_model_metrics = list(evaluation_results.values())[0]

    if "R2 Score" in first_model_metrics:
        metric_to_compare = "R2 Score"
        is_regression = True
    else:
        metric_to_compare = "Accuracy"
        is_regression = False

    # ---------------- MODEL COMPARISON CHART ----------------
    st.subheader("📈 Model Comparison Chart")

    chart_df = comparison_df.set_index("Model")[[metric_to_compare]]

    st.bar_chart(chart_df)

    # ---------------- BEST MODEL ----------------
    st.subheader("🏆 Best Model")

    best_model = comparison_df.loc[
        comparison_df[metric_to_compare].idxmax()
    ]

    best_model_name = best_model["Model"]

    st.success(f"Best Model: {best_model_name}")

    st.write(f"**{metric_to_compare}:** {best_model[metric_to_compare]:.4f}")

    st.write("### 📋 Best Model Summary")

    st.write(f"**Model Name:** {best_model_name}")

    st.write(f"**Evaluation Metric:** {metric_to_compare}")

    st.write(f"**Score:** {best_model[metric_to_compare]:.4f}")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("⭐ Feature Importance")

    best_model_object = models_dict[best_model_name]["model"]

    if hasattr(best_model_object, "feature_importances_"):

        feature_importance_df = pd.DataFrame({
            "Feature": best_model_object.feature_names_in_,
            "Importance": best_model_object.feature_importances_
        })

        feature_importance_df = feature_importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        st.dataframe(feature_importance_df)

        st.bar_chart(
            feature_importance_df.set_index("Feature")
        )

    elif hasattr(best_model_object, "coef_"):

        feature_importance_df = pd.DataFrame({
            "Feature": best_model_object.feature_names_in_,
            "Importance": abs(best_model_object.coef_)
        })

        feature_importance_df = feature_importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        st.dataframe(feature_importance_df)

        st.bar_chart(
            feature_importance_df.set_index("Feature")
        )

    else:

        st.info(
            "Feature importance is not available for this model."
        )

    # ---------------- ACTUAL VS PREDICTED ----------------
    if is_regression:

        st.subheader("📉 Actual vs Predicted")

        best_predictions = models_dict[best_model_name]["predictions"]

        plot_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": best_predictions
        })

        st.dataframe(plot_df.head())

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            plot_df["Actual"],
            plot_df["Predicted"]
        )

        min_value = min(
            plot_df["Actual"].min(),
            plot_df["Predicted"].min()
        )

        max_value = max(
            plot_df["Actual"].max(),
            plot_df["Predicted"].max()
        )

        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--"
        )

        ax.set_xlabel("Actual Values")

        ax.set_ylabel("Predicted Values")

        ax.set_title("Actual vs Predicted")

        st.pyplot(fig)

        # ---------------- RESIDUAL ANALYSIS ----------------
        st.subheader("📉 Residual Analysis")

        plot_df["Residual"] = (
            plot_df["Actual"] - plot_df["Predicted"]
        )

        st.write("### Residual Preview")

        st.dataframe(
            plot_df[
                ["Actual", "Predicted", "Residual"]
            ].head()
        )

        # Residual Scatter Plot
        fig2, ax2 = plt.subplots(figsize=(7, 5))

        ax2.scatter(
            plot_df["Predicted"],
            plot_df["Residual"]
        )

        ax2.axhline(
            y=0,
            linestyle="--"
        )

        ax2.set_xlabel("Predicted Values")

        ax2.set_ylabel("Residual")

        ax2.set_title("Residual Plot")

        st.pyplot(fig2)

        # Residual Distribution
        fig3, ax3 = plt.subplots(figsize=(7, 5))

        ax3.hist(
            plot_df["Residual"],
            bins=20
        )

        ax3.set_xlabel("Residual")

        ax3.set_ylabel("Frequency")

        ax3.set_title("Residual Distribution")

        st.pyplot(fig3)

    else:

        st.subheader("📉 Confusion Matrix")

        st.info(
            "Confusion Matrix visualization will be added for classification models."
        )