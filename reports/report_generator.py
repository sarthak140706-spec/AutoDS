import pandas as pd


def generate_report(
    dataframe,
    target_column,
    evaluation_results
):
    """
    Generate a machine learning report.
    """

    report = {}

    # ---------------- Dataset Information ----------------
    report["Number of Rows"] = dataframe.shape[0]

    report["Number of Columns"] = dataframe.shape[1]

    report["Target Column"] = target_column

    # ---------------- Model Information ----------------
    report["Models Trained"] = list(
        evaluation_results.keys()
    )

    # ---------------- Evaluation Results ----------------
    report["Evaluation Results"] = evaluation_results

    # ---------------- Best Model ----------------
    first_metrics = list(
        evaluation_results.values()
    )[0]

    if "R2 Score" in first_metrics:

        metric_name = "R2 Score"

    else:

        metric_name = "Accuracy"

    best_model = max(
        evaluation_results,
        key=lambda model:
        evaluation_results[model][metric_name]
    )

    report["Best Model"] = best_model

    report["Best Score"] = (
        evaluation_results[best_model][metric_name]
    )

    # ---------------- Convert to DataFrame ----------------
    report_df = pd.DataFrame(
        {
            "Property": report.keys(),
            "Value": report.values()
        }
    )

    return report_df