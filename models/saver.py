import os
import joblib


def save_model(
    models_dict,
    evaluation_results,
    encoders,
    scaler,
    feature_columns
):
    """
    Save the best model along with preprocessing objects
    and feature names.
    """

    first_metrics = list(evaluation_results.values())[0]

    if "R2 Score" in first_metrics:
        metric = "R2 Score"
    else:
        metric = "Accuracy"

    best_model_name = max(
        evaluation_results,
        key=lambda model: evaluation_results[model][metric]
    )

    best_model = models_dict[best_model_name]["model"]

    os.makedirs("saved_models", exist_ok=True)

    file_path = "saved_models/best_model.pkl"

    joblib.dump(
        {
            "model": best_model,
            "encoders": encoders,
            "scaler": scaler,
            "feature_columns": list(feature_columns)
        },
        file_path
    )

    return best_model, best_model_name, file_path