from pandas.api.types import (
    is_numeric_dtype
)


def detect_problem_type(y):
    """
    Detect whether the target column represents a
    regression or classification problem.
    """

    # ---------------- Non-numeric ----------------

    if not is_numeric_dtype(y):

        return "classification"

    # ---------------- Numeric ----------------

    unique_values = y.nunique()

    total_values = len(y)

    unique_ratio = unique_values / total_values

    # Few unique values → Classification

    if unique_values <= 20:

        return "classification"

    # Small ratio of unique values → Classification

    if unique_ratio < 0.05:

        return "classification"

    # Otherwise → Regression

    return "regression"