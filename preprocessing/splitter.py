from sklearn.model_selection import train_test_split


def split_dataset(
    dataframe,
    target_column,
    test_size=0.20,
    random_state=42
):
    """
    Split the dataset into training and testing datasets.
    """

    X = dataframe.drop(
        columns=[target_column]
    )

    y = dataframe[target_column]

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    train_dataframe = X_train.copy()

    train_dataframe[target_column] = y_train.values

    test_dataframe = X_test.copy()

    test_dataframe[target_column] = y_test.values

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_dataframe,
        test_dataframe
    )