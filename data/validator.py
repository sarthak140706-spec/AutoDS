import pandas as pd


def validate_dataset(dataframe):
    """
    Validate the uploaded dataset before processing.
    """

    # ---------------- Dataset Exists ----------------

    if dataframe is None:

        return (
            False,
            "❌ Failed to load the dataset."
        )

    # ---------------- Empty Dataset ----------------

    if dataframe.empty:

        return (
            False,
            "❌ The uploaded dataset is empty."
        )

    # ---------------- Minimum Columns ----------------

    if dataframe.shape[1] < 2:

        return (
            False,
            "❌ Dataset must contain at least two columns."
        )

    # ---------------- Minimum Rows ----------------

    if dataframe.shape[0] < 10:

        return (
            False,
            "❌ Dataset must contain at least 10 rows for training."
        )

    # ---------------- Duplicate Column Names ----------------

    if dataframe.columns.duplicated().any():

        return (
            False,
            "❌ Dataset contains duplicate column names."
        )

    # ---------------- Completely Empty Columns ----------------

    empty_columns = dataframe.columns[
        dataframe.isnull().all()
    ]

    if len(empty_columns) > 0:

        return (
            False,
            "❌ Dataset contains completely empty column(s): "
            + ", ".join(empty_columns)
        )

    # ---------------- Completely Empty Rows ----------------

    if dataframe.isnull().all(axis=1).any():

        return (
            False,
            "❌ Dataset contains completely empty row(s)."
        )

    # ---------------- All Values Missing ----------------

    if dataframe.isna().sum().sum() == dataframe.size:

        return (
            False,
            "❌ Every value in the dataset is missing."
        )

    return (
        True,
        "Dataset validation successful."
    )