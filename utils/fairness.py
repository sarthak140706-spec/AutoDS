import pandas as pd


def generate_fairness_report(
    X_test,
    predictions
):
    """
    Generates a simple fairness report by comparing
    prediction averages across categorical features.
    """

    report = []

    categorical_columns = X_test.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_columns) == 0:

        report.append(
            "No categorical columns found. "
            "Fairness analysis skipped."
        )

        return "\n".join(report)

    prediction_series = pd.Series(
        predictions,
        index=X_test.index
    )

    for column in categorical_columns:

        report.append(
            f"\nFeature: {column}"
        )

        grouped = prediction_series.groupby(
            X_test[column]
        ).mean()

        for group, value in grouped.items():

            report.append(
                f"  {group}: {value:.4f}"
            )

    report.append(
        "\nLarge differences between groups may indicate "
        "potential prediction bias and should be investigated."
    )

    return "\n".join(report)