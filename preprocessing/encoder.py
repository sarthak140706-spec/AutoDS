from sklearn.preprocessing import OrdinalEncoder


def encode_dataset(X_train, X_test):
    """
    Encode categorical columns using OrdinalEncoder.
    Handles unseen categories during prediction.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    encoders = {}

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

    return X_train, X_test, encoders