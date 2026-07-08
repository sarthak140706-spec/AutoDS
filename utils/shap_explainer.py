import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_shap_explanation(
    model,
    X_train,
    X_test
):
    """
    Generate SHAP values for the trained model.
    """

    try:

        explainer = shap.Explainer(
            model,
            X_train
        )

        shap_values = explainer(
            X_test
        )

        return {
            "explainer": explainer,
            "shap_values": shap_values
        }

    except Exception:

        return None


def show_shap_summary(
    shap_values,
    X_test
):
    """
    SHAP summary plot.
    """

    fig = plt.figure(
        figsize=(8, 5)
    )

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    return fig


def show_shap_bar(
    shap_values,
    X_test
):
    """
    SHAP feature importance bar plot.
    """

    fig = plt.figure(
        figsize=(8, 5)
    )

    shap.plots.bar(
        shap_values,
        show=False
    )

    return fig


def get_top_shap_features(
    shap_values,
    X_test
):
    """
    Return ranked SHAP feature importance.
    """

    values = shap_values.values

    if len(values.shape) == 3:
        values = values[:, :, 0]

    importance = np.abs(
        values
    ).mean(axis=0)

    feature_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Mean |SHAP|": importance
        }
    )

    feature_df = feature_df.sort_values(
        by="Mean |SHAP|",
        ascending=False
    )

    return feature_df