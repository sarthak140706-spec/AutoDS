
# NOTE:
# This version has been cleaned for Step 10.9 preparation.
# - Removed duplicate model_monitor import.
# - Interactive dashboard helper imports retained.
# - Ready for replacing Streamlit charts with Plotly helper functions.
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from utils.history import (
    get_history,
    clear_history
)
from utils.shap_explainer import (
    generate_shap_explanation,
    show_shap_summary,
    show_shap_bar,
    get_top_shap_features
)
from utils.permutation_importance import (
    calculate_permutation_importance,
    plot_permutation_importance
)
from utils.explainability_report import (
    generate_explainability_report
)
from utils.fairness import (
    generate_fairness_report
)
from utils.model_monitor import (
    generate_model_monitoring_report
)
from utils.interactive_dashboard import (
    create_model_comparison_chart,
    create_cv_chart,
    create_feature_importance_chart,
    create_residual_chart
)
import shap
from sklearn.inspection import PartialDependenceDisplay

def show_model_page(
    models_dict,
    evaluation_results,
    X_train,
    X_test,
    y_test
):

    st.subheader("🤖 Model Comparison Dashboard")

    st.success(
        "Multi-model training completed successfully."
    )

    # ---------------- MODEL DETAILS ----------------

    st.subheader(
        "📌 Individual Model Results"
    )

    for model_name, model_data in models_dict.items():

        st.markdown(
            f"### {model_name}"
        )

        st.write(
            "**First 5 Predictions:**"
        )

        st.write(
            model_data["predictions"][:5]
        )

        st.write(
            "**First 5 Actual Values:**"
        )

        st.write(
            y_test[:5]
        )

        # ---------------- Hyperparameter Tuning ----------------

        if model_data.get("best_params"):

            st.write(
                "### 🔧 Best Hyperparameters"
            )

            best_params_df = pd.DataFrame(
                {
                    "Parameter": model_data[
                        "best_params"
                    ].keys(),
                    "Value": model_data[
                        "best_params"
                    ].values()
                }
            )

            st.dataframe(
                best_params_df,
                use_container_width=True
            )

        if model_data.get("best_cv_score") is not None:

            st.write(
                f"**Best Cross Validation Score:** "
                f"{model_data['best_cv_score']:.4f}"
            )

        st.markdown("---")

    # ---------------- COMPARISON TABLE ----------------

    st.subheader(
        "📊 Model Performance Comparison"
    )

    comparison_data = []

    for model_name, metrics in evaluation_results.items():

        row = {

            "Model": model_name

        }

        row.update(
            metrics
        )

        comparison_data.append(
            row
        )

    comparison_df = pd.DataFrame(
        comparison_data
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    # ---------------- CROSS VALIDATION SUMMARY ----------------

    st.subheader("📋 Cross Validation Summary")

    cv_summary = []

    for model_name, model_data in models_dict.items():

        cv_summary.append({

            "Model": model_name,

            "CV Mean": model_data["cv_mean"],

            "CV Std": model_data["cv_std"],

            "Best GridSearch CV": model_data["best_cv_score"]

        })

    cv_summary_df = pd.DataFrame(
        cv_summary
    )

    st.dataframe(
        cv_summary_df
    )

    # ---------------- DETERMINE METRIC ----------------

    first_model_metrics = list(
        evaluation_results.values()
    )[0]

    if "R2 Score" in first_model_metrics:

        metric_to_compare = "R2 Score"

        is_regression = True

    else:

        metric_to_compare = "Accuracy"

        is_regression = False

    # ---------------- MODEL COMPARISON CHART ----------------

    st.subheader(
        "📈 Model Comparison Chart"
    )

    chart_df = comparison_df.set_index(
        "Model"
    )[[metric_to_compare]]

    st.bar_chart(
        chart_df
    )

    # ---------------- CROSS VALIDATION COMPARISON ----------------

    st.subheader(
        "📈 Cross Validation Comparison"
    )

    cv_chart_df = cv_summary_df.set_index(
        "Model"
    )[["CV Mean"]]

    st.bar_chart(
        cv_chart_df
    )

    # ---------------- MODEL STABILITY ----------------

    st.subheader(
        "🏅 Model Stability Analysis"
    )

    stable_model = cv_summary_df.loc[
        cv_summary_df["CV Std"].idxmin()
    ]

    st.success(
        f"Most Stable Model: {stable_model['Model']}"
    )

    st.write(
        f"**Cross Validation Mean:** "
        f"{stable_model['CV Mean']:.4f}"
    )

    st.write(
        f"**Cross Validation Standard Deviation:** "
        f"{stable_model['CV Std']:.4f}"
    )

    st.info(
        "A lower Cross Validation Standard Deviation indicates "
        "that the model performs more consistently across "
        "different folds of the dataset."
    )

    # ---------------- BEST MODEL ----------------

    st.subheader(
        "🏆 Best Model"
    )

    best_model = comparison_df.loc[
        comparison_df[
            metric_to_compare
        ].idxmax()
    ]

    best_model_name = best_model[
        "Model"
    ]

    st.success(
        f"Best Model: {best_model_name}"
    )

    st.write(
        f"**{metric_to_compare}:** "
        f"{best_model[metric_to_compare]:.4f}"
    )

    st.write(
        "### 📋 Best Model Summary"
    )

    st.write(
        f"**Model Name:** {best_model_name}"
    )

    st.write(
        f"**Evaluation Metric:** {metric_to_compare}"
    )

    st.write(
        f"**Score:** {best_model[metric_to_compare]:.4f}"
    )

    # ---------------- Best Hyperparameters ----------------

    best_model_data = models_dict[
        best_model_name
    ]

    if best_model_data.get(
        "best_params"
    ):

        st.write(
            "### 🔧 Optimized Hyperparameters"
        )

        best_params_df = pd.DataFrame(
            {
                "Parameter": best_model_data[
                    "best_params"
                ].keys(),
                "Value": best_model_data[
                    "best_params"
                ].values()
            }
        )

        st.dataframe(
            best_params_df,
            use_container_width=True
        )

    if best_model_data.get(
        "best_cv_score"
    ) is not None:

        st.write(
            f"**Best Cross Validation Score:** "
            f"{best_model_data['best_cv_score']:.4f}"
        )

    # ---------------- FEATURE IMPORTANCE ----------------

    st.subheader(
        "⭐ Feature Importance"
    )

    best_model_object = best_model_data[
        "model"
    ]

    if hasattr(
        best_model_object,
        "feature_importances_"
    ):

        feature_importance_df = pd.DataFrame(
            {
                "Feature":
                    best_model_object.feature_names_in_,

                "Importance":
                    best_model_object.feature_importances_
            }
        )

        feature_importance_df = (
            feature_importance_df.sort_values(
                by="Importance",
                ascending=False
            )
        )

        st.dataframe(
            feature_importance_df,
            use_container_width=True
        )

        st.bar_chart(
            feature_importance_df.set_index(
                "Feature"
            )
        )

    elif hasattr(
        best_model_object,
        "coef_"
    ):

        feature_importance_df = pd.DataFrame(
            {
                "Feature":
                    best_model_object.feature_names_in_,

                "Importance":
                    abs(
                        best_model_object.coef_
                    )
            }
        )

        feature_importance_df = (
            feature_importance_df.sort_values(
                by="Importance",
                ascending=False
            )
        )

        st.dataframe(
            feature_importance_df,
            use_container_width=True
        )

        st.bar_chart(
            feature_importance_df.set_index(
                "Feature"
            )
        )

    else:

        st.info(
            "Feature importance is not available for this model."
        )


    # ---------------- SHAP EXPLAINABILITY ----------------

    st.subheader(
        "🧠 SHAP Explainability"
    )

    shap_results = generate_shap_explanation(
        best_model_object,
        X_train,
        X_test
    )

    if shap_results is None:

        st.info(
            "SHAP Explainability is not supported for this model."
        )

    else:

        st.write(
            "### SHAP Summary Plot"
        )

        summary_fig = show_shap_summary(
            shap_results["shap_values"],
            X_test
        )

        st.pyplot(
            summary_fig
        )

        st.write(
            "### SHAP Feature Importance"
        )

        bar_fig = show_shap_bar(
            shap_results["shap_values"],
            X_test
        )

        st.pyplot(
            bar_fig
        )

        st.write(
            "### Top SHAP Features"
        )

        top_features = get_top_shap_features(
            shap_results["shap_values"],
            X_test
        )

        st.dataframe(
            top_features,
            use_container_width=True
        )

        # ---------------- SINGLE PREDICTION EXPLANATION ----------------

        st.subheader(
            "🔍 Explain Individual Prediction"
        )

        sample_index = st.selectbox(
            "Select Test Sample",
            options=range(len(X_test)),
            format_func=lambda x: f"Sample {x}"
        )

        selected_sample = X_test.iloc[[sample_index]]

        st.write(
            "### Selected Sample"
        )

        st.dataframe(
            selected_sample,
            use_container_width=True
        )

        try:

            st.write(
                "### SHAP Waterfall Plot"
            )

            fig = plt.figure(
                figsize=(10, 6)
            )

            shap.plots.waterfall(
                shap_results["shap_values"][sample_index],
                show=False
            )

            st.pyplot(fig)

        except Exception:

            st.info(
                "Waterfall plot is not supported for this model."
            )
    # ---------------- PERMUTATION FEATURE IMPORTANCE ----------------

    st.subheader(
        "🔄 Permutation Feature Importance"
    )

    if is_regression:

        scoring = "r2"

    else:

        scoring = "accuracy"

    permutation_df = calculate_permutation_importance(

        best_model_object,

        X_test,

        y_test,

        scoring=scoring

    )

    if permutation_df is None:

        st.info(
            "Permutation Feature Importance is not available for this model."
        )

    else:

        st.dataframe(

            permutation_df,

            use_container_width=True

        )

        permutation_fig = plot_permutation_importance(

            permutation_df

        )

        st.pyplot(
            permutation_fig
        )

    # ---------------- EXPLAINABILITY REPORT ----------------

    st.subheader(
        "📝 Explainability Report"
    )

    try:

        report = generate_explainability_report(
            top_features,
            permutation_df,
            best_model_name
        )

        st.text_area(
            "Generated Report",
            report,
            height=250
        )

        st.download_button(
            label="📄 Download Explainability Report",
            data=report,
            file_name="explainability_report.txt",
            mime="text/plain"
        )

    except Exception:

        st.info(
            "Explainability report could not be generated."
        )

    # ---------------- FAIRNESS REPORT ----------------

    st.subheader(
        "⚖️ Fairness Analysis"
    )

    try:

        fairness_report = generate_fairness_report(
            X_test,
            models_dict[best_model_name]["predictions"]
        )

        st.text_area(
            "Fairness Report",
            fairness_report,
            height=250
        )

        st.download_button(
            label="📄 Download Fairness Report",
            data=fairness_report,
            file_name="fairness_report.txt",
            mime="text/plain"
        )

    except Exception:

        st.info(
            "Fairness analysis could not be generated."
        )

    # ---------------- PARTIAL DEPENDENCE PLOTS ----------------

    st.subheader(
        "📊 Partial Dependence Plots"
    )

    try:

        if hasattr(best_model_object, "predict"):

            feature_names = list(
                X_test.columns
            )

            top_features = feature_names[:2]

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            PartialDependenceDisplay.from_estimator(

                estimator=best_model_object,

                X=X_test,

                features=top_features,

                feature_names=feature_names,

                ax=ax

            )

            st.pyplot(fig)

        else:

            st.info(
                "Partial Dependence Plot is not available for this model."
            )

    except Exception:

        st.info(
            "Partial Dependence Plot is not supported for this model."
        )
    # ---------------- ACTUAL VS PREDICTED ----------------

    if is_regression:

        st.subheader(
            "📉 Actual vs Predicted"
        )

        best_predictions = models_dict[
            best_model_name
        ]["predictions"]

        plot_df = pd.DataFrame(
            {
                "Actual": y_test.values,
                "Predicted": best_predictions
            }
        )

        st.dataframe(
            plot_df.head(),
            use_container_width=True
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

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

        ax.set_xlabel(
            "Actual Values"
        )

        ax.set_ylabel(
            "Predicted Values"
        )

        ax.set_title(
            "Actual vs Predicted"
        )

        st.pyplot(
            fig
        )

        # ---------------- RESIDUAL ANALYSIS ----------------

        st.subheader(
            "📉 Residual Analysis"
        )

        plot_df["Residual"] = (
            plot_df["Actual"]
            - plot_df["Predicted"]
        )

        st.write(
            "### Residual Preview"
        )

        st.dataframe(
            plot_df[
                [
                    "Actual",
                    "Predicted",
                    "Residual"
                ]
            ].head(),
            use_container_width=True
        )

        fig2, ax2 = plt.subplots(
            figsize=(7, 5)
        )

        ax2.scatter(
            plot_df["Predicted"],
            plot_df["Residual"]
        )

        ax2.axhline(
            y=0,
            linestyle="--"
        )

        ax2.set_xlabel(
            "Predicted Values"
        )

        ax2.set_ylabel(
            "Residual"
        )

        ax2.set_title(
            "Residual Plot"
        )

        st.pyplot(
            fig2
        )

        fig3, ax3 = plt.subplots(
            figsize=(7, 5)
        )

        ax3.hist(
            plot_df["Residual"],
            bins=20
        )

        ax3.set_xlabel(
            "Residual"
        )

        ax3.set_ylabel(
            "Frequency"
        )

        ax3.set_title(
            "Residual Distribution"
        )

        st.pyplot(
            fig3
        )

    else:

        st.subheader(
            "📉 Confusion Matrix"
        )

        st.info(
            "Confusion Matrix visualization will be added for classification models."
        )

    # ---------------- EXPERIMENT HISTORY ----------------

    st.subheader(
        "🧪 Experiment History"
    )

    history_df = get_history()

    if history_df.empty:

        st.info(
            "No experiments have been recorded yet."
        )

    else:

        st.dataframe(
            history_df,
            use_container_width=True
        )

        # ---------------- MODEL MONITORING REPORT ----------------

        st.subheader(
            "📡 Model Monitoring Dashboard"
        )

        try:

            monitoring_report = generate_model_monitoring_report(
                history_df
            )

            st.text_area(
                "Monitoring Report",
                monitoring_report,
                height=250
            )

            st.download_button(
                label="📄 Download Monitoring Report",
                data=monitoring_report,
                file_name="model_monitoring_report.txt",
                mime="text/plain"
            )

        except Exception:

            st.info(
                "Monitoring report could not be generated."
            )

        # ---------------- DOWNLOAD HISTORY ----------------

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download History (CSV)",
            data=csv,
            file_name="experiment_history.csv",
            mime="text/csv"
        )

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            history_df.to_excel(
                writer,
                index=False,
                sheet_name="History"
            )

        st.download_button(
            label="📥 Download History (Excel)",
            data=excel_buffer.getvalue(),
            file_name="experiment_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Experiments",
                len(history_df)
            )

        with col2:

            st.metric(
                "Latest Best Model",
                history_df.iloc[-1]["Best Model"]
            )

        with col3:

            average_score = round(
                history_df["Score"].mean(),
                4
            )

            st.metric(
                "Average Score",
                average_score
            )

        if st.button(
            "🗑 Clear Experiment History"
        ):

            clear_history()

            st.success(
                "Experiment history cleared successfully."
            )

            st.rerun()

        # ---------------- LEADERBOARD ----------------

        st.subheader(
            "🏆 Experiment Leaderboard"
        )

        leaderboard_df = history_df.sort_values(
            by="Score",
            ascending=False,
            ignore_index=True
        )

        leaderboard_df.index += 1

        st.dataframe(
            leaderboard_df,
            use_container_width=True
        )

        st.caption(
            "Experiments are ranked according to their evaluation score."
        )