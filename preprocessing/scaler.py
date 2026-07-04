from sklearn.preprocessing import StandardScaler


def scale_dataset(X_train, X_test):
    """
    Scale numerical features using statistics from the training data only.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    numerical_columns = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    scaler = StandardScaler()

    if len(numerical_columns) > 0:

        X_train[numerical_columns] = scaler.fit_transform(
            X_train[numerical_columns]
        )

        X_test[numerical_columns] = scaler.transform(
            X_test[numerical_columns]
        )

    return X_train, X_test, scaler