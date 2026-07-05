from sklearn.preprocessing import OrdinalEncoder


def encode_dataset(
    X_train,
    X_test,
    enable_encoding=True
):
    """
    Encode categorical columns using OrdinalEncoder.

    If encoding is disabled, the original datasets are returned.
    """

    X_train = X_train.copy()

    X_test = X_test.copy()

    # ---------------- Skip Encoding ----------------

    if not enable_encoding:

        return (
            X_train,
            X_test,
            {}
        )

    # ---------------- Find Categorical Columns ----------------

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    encoders = {}

    # ---------------- Encode ----------------

    for column in categorical_columns:

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        X_train[[column]] = encoder.fit_transform(
            X_train[[column]].astype(str)
        )

        X_test[[column]] = encoder.transform(
            X_test[[column]].astype(str)
        )

        encoders[column] = encoder

    return (
        X_train,
        X_test,
        encoders
    )