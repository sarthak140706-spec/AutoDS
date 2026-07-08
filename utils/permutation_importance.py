from sklearn.inspection import permutation_importance

import pandas as pd
import matplotlib.pyplot as plt


def calculate_permutation_importance(
    model,
    X_test,
    y_test,
    scoring=None
):
    """
    Calculate Permutation Feature Importance.
    """

    try:

        result = permutation_importance(

            estimator=model,

            X=X_test,

            y=y_test,

            scoring=scoring,

            n_repeats=10,

            random_state=42,

            n_jobs=-1

        )

        importance_df = pd.DataFrame(

            {

                "Feature": X_test.columns,

                "Importance": result.importances_mean,

                "Std": result.importances_std

            }

        )

        importance_df = importance_df.sort_values(

            by="Importance",

            ascending=False

        )

        return importance_df

    except Exception:

        return None


def plot_permutation_importance(
    importance_df
):
    """
    Plot Permutation Feature Importance.
    """

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    plot_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    ax.barh(

        plot_df["Feature"],

        plot_df["Importance"]

    )

    ax.set_xlabel(
        "Permutation Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Permutation Feature Importance"
    )

    return fig