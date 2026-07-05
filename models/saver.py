import joblib


def save_model(
    models_dict,
    evaluation_results,
    encoders,
    scaler,
    feature_columns,
    save_model_enabled=True
):
    """
    Save the best-performing model along with preprocessing objects.

    If model saving is disabled, the best model is returned
    without writing a file to disk.
    """

    # ---------------- Determine Best Metric ----------------

    first_metrics = list(
        evaluation_results.values()
    )[0]

    if "R2 Score" in first_metrics:

        metric_name = "R2 Score"

    else:

        metric_name = "Accuracy"

    # ---------------- Select Best Model ----------------

    best_model_name = max(
        evaluation_results,
        key=lambda model:
        evaluation_results[model][metric_name]
    )

    best_model = models_dict[
        best_model_name
    ]["model"]

    file_path = None

    # ---------------- Save Model ----------------

    if save_model_enabled:

        file_path = "best_model.pkl"

        joblib.dump(
            {
                "model": best_model,
                "encoders": encoders,
                "scaler": scaler,
                "feature_columns": list(feature_columns)
            },
            file_path
        )

    return (
        best_model,
        best_model_name,
        file_path
    )