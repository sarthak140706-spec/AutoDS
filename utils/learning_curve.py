from sklearn.model_selection import learning_curve

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_learning_curve(
    model,
    X,
    y,
    scoring
):
    """
    Generate learning curve statistics.
    """

    try:

        train_sizes, train_scores, validation_scores = learning_curve(

            estimator=model,

            X=X,

            y=y,

            cv=5,

            scoring=scoring,

            train_sizes=np.linspace(
                0.1,
                1.0,
                10
            ),

            n_jobs=-1,

            shuffle=True,

            random_state=42

        )

        results = {

            "train_sizes": train_sizes,

            "train_mean": train_scores.mean(axis=1),

            "train_std": train_scores.std(axis=1),

            "validation_mean": validation_scores.mean(axis=1),

            "validation_std": validation_scores.std(axis=1)

        }

        return results

    except Exception:

        return None


def plot_learning_curve(
    results
):
    """
    Plot learning curve.
    """

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(

        results["train_sizes"],

        results["train_mean"],

        marker="o",

        label="Training Score"

    )

    ax.fill_between(

        results["train_sizes"],

        results["train_mean"] - results["train_std"],

        results["train_mean"] + results["train_std"],

        alpha=0.2

    )

    ax.plot(

        results["train_sizes"],

        results["validation_mean"],

        marker="o",

        label="Cross Validation Score"

    )

    ax.fill_between(

        results["train_sizes"],

        results["validation_mean"] - results["validation_std"],

        results["validation_mean"] + results["validation_std"],

        alpha=0.2

    )

    ax.set_xlabel(
        "Training Samples"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Learning Curve"
    )

    ax.legend()

    return fig


def learning_curve_table(
    results
):
    """
    Return learning curve as dataframe.
    """

    return pd.DataFrame(

        {

            "Training Samples":
                results["train_sizes"],

            "Training Score":
                results["train_mean"],

            "Validation Score":
                results["validation_mean"]

        }

    )