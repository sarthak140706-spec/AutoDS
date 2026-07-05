import pandas as pd

from utils.problem_type import detect_problem_type


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

    # ---------------- Problem Type ----------------

    problem_type = detect_problem_type(
        dataframe[target_column]
    )

    report["Problem Type"] = problem_type.title()

    # ---------------- Models ----------------

    report["Models Trained"] = ", ".join(
        evaluation_results.keys()
    )

    # ---------------- Best Metric ----------------

    if problem_type == "regression":

        metric_name = "R2 Score"

    else:

        metric_name = "Accuracy"

    best_model = max(
        evaluation_results,
        key=lambda model:
        evaluation_results[model][metric_name]
    )

    report["Best Model"] = best_model

    report["Best Metric"] = metric_name

    report["Best Score"] = round(
        evaluation_results[best_model][metric_name],
        4
    )

    # ---------------- Evaluation Summary ----------------

    for model_name, metrics in evaluation_results.items():

        metric_text = ", ".join(
            [
                f"{metric}: {round(value, 4)}"
                for metric, value in metrics.items()
            ]
        )

        report[model_name] = metric_text

    # ---------------- Convert ----------------

    report_df = pd.DataFrame(
        {
            "Property": list(report.keys()),
            "Value": list(report.values())
        }
    )

    # ---------------- Arrow Compatibility ----------------

    report_df["Property"] = report_df["Property"].astype(str)

    report_df["Value"] = report_df["Value"].astype(str)

    return report_df