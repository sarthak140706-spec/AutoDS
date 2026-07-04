from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np


def evaluate_models(models_dict, y_test):

    evaluation_results = {}

    for model_name, model_data in models_dict.items():

        predictions = model_data["predictions"]
        model = model_data["model"]

        metrics = {}

        # ---------------- Regression ----------------
        if model.__class__.__name__ in ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor"]:

            metrics["R2 Score"] = r2_score(y_test, predictions)
            metrics["MAE"] = mean_absolute_error(y_test, predictions)
            metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, predictions))

        # ---------------- Classification ----------------
        else:

            metrics["Accuracy"] = accuracy_score(y_test, predictions)
            metrics["Precision"] = precision_score(y_test, predictions, average="weighted", zero_division=0)
            metrics["Recall"] = recall_score(y_test, predictions, average="weighted", zero_division=0)
            metrics["F1 Score"] = f1_score(y_test, predictions, average="weighted", zero_division=0)

        evaluation_results[model_name] = metrics

    return evaluation_results