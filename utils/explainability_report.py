import pandas as pd


def generate_explainability_report(
    shap_df,
    permutation_df,
    model_name
):
    """
    Generates a simple natural-language explanation
    of the model's behavior.
    """

    report = []

    report.append(
        f"Best Model Selected: {model_name}"
    )

    report.append("")

    if shap_df is not None and not shap_df.empty:

        report.append(
            "Top SHAP Features:"
        )

        for _, row in shap_df.head(5).iterrows():

            report.append(
                f"- {row['Feature']} "
                f"(importance={row['Mean |SHAP|']:.4f})"
            )

        report.append("")

    if permutation_df is not None and not permutation_df.empty:

        report.append(
            "Top Permutation Features:"
        )

        for _, row in permutation_df.head(5).iterrows():

            report.append(
                f"- {row['Feature']} "
                f"(importance={row['Importance']:.4f})"
            )

        report.append("")

    report.append(
        "Interpretation:"
    )

    report.append(
        "Features appearing consistently in both SHAP and "
        "Permutation Importance have the strongest influence "
        "on the model's predictions."
    )

    return "\n".join(report)