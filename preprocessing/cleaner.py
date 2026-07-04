def preprocess_dataset(train_dataframe, test_dataframe):
    """
    Fill missing values in training and testing feature datasets.
    """

    X_train = train_dataframe.copy()
    X_test = test_dataframe.copy()

    numerical_columns = X_train.select_dtypes(include="number").columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # ---------------- Numerical Columns ----------------
    for col in numerical_columns:

        mean_value = X_train[col].mean()

        X_train[col] = X_train[col].fillna(mean_value)

        X_test[col] = X_test[col].fillna(mean_value)

    # ---------------- Categorical Columns ----------------
    for col in categorical_columns:

        mode_value = X_train[col].mode()[0]

        X_train[col] = X_train[col].fillna(mode_value)

        X_test[col] = X_test[col].fillna(mode_value)

    return X_train, X_test