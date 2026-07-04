from sklearn.preprocessing import LabelEncoder


def encode_dataset(X_train, X_test):
    """
    Encode categorical columns using LabelEncoder.
    Encoders are fitted only on the training data.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    encoders = {}

    for column in categorical_columns:

        encoder = LabelEncoder()

        X_train[column] = encoder.fit_transform(
            X_train[column].astype(str)
        )

        # Handle unseen categories in test data
        X_test[column] = X_test[column].astype(str)

        X_test[column] = X_test[column].apply(
            lambda value: value if value in encoder.classes_ else encoder.classes_[0]
        )

        X_test[column] = encoder.transform(X_test[column])

        encoders[column] = encoder

    return X_train, X_test, encoders