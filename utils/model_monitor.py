import pandas as pd


def generate_model_monitoring_report(
    history_df
):
    """
    Generates a simple monitoring report from
    experiment history.
    """

    if history_df.empty:

        return (
            "No experiments available."
        )

    report = []

    report.append(
        f"Total Experiments: {len(history_df)}"
    )

    report.append(
        f"Latest Best Model: "
        f"{history_df.iloc[-1]['Best Model']}"
    )

    report.append(
        f"Average Score: "
        f"{history_df['Score'].mean():.4f}"
    )

    report.append(
        f"Highest Score: "
        f"{history_df['Score'].max():.4f}"
    )

    report.append(
        f"Lowest Score: "
        f"{history_df['Score'].min():.4f}"
    )

    report.append(
        "\nPerformance Trend:"
    )

    for i, row in history_df.iterrows():

        report.append(
            f"Experiment {i+1}: "
            f"{row['Best Model']} "
            f"({row['Score']:.4f})"
        )

    return "\n".join(report)