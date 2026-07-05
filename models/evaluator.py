from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from utils.problem_type import detect_problem_type

import numpy as np


def evaluate_models(
    models_dict,
    y_test
):
    """
    Evaluate all trained models.
    """

    evaluation_results = {}

    # ---------------- Detect Problem Type ----------------

    problem_type = detect_problem_type(
        y_test
    )

    # ---------------- Evaluate ----------------

    for model_name, model_data in models_dict.items():

        predictions = model_data["predictions"]

        # -------- Regression --------

        if problem_type == "regression":

            evaluation_results[model_name] = {

                "MAE": mean_absolute_error(
                    y_test,
                    predictions
                ),

                "RMSE": np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                ),

                "R2 Score": r2_score(
                    y_test,
                    predictions
                )

            }

        # -------- Classification --------

        else:

            evaluation_results[model_name] = {

                "Accuracy": accuracy_score(
                    y_test,
                    predictions
                ),

                "Precision": precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ),

                "Recall": recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ),

                "F1 Score": f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

            }

    return evaluation_results