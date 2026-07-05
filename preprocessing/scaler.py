from sklearn.preprocessing import StandardScaler


def scale_dataset(
    X_train,
    X_test,
    enable_scaling=True
):
    """
    Scale numerical features using StandardScaler.

    If scaling is disabled, the original datasets are returned.
    """

    X_train = X_train.copy()

    X_test = X_test.copy()

    # ---------------- Skip Scaling ----------------

    if not enable_scaling:

        return (
            X_train,
            X_test,
            None
        )

    # ---------------- Find Numerical Columns ----------------

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    scaler = StandardScaler()

    # ---------------- Scale ----------------

    X_train[numerical_columns] = scaler.fit_transform(
        X_train[numerical_columns]
    )

    X_test[numerical_columns] = scaler.transform(
        X_test[numerical_columns]
    )

    return (
        X_train,
        X_test,
        scaler
    )