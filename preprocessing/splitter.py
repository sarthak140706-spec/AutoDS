from sklearn.model_selection import train_test_split


def split_dataset(dataframe, target_column):
    """
    Split dataset into train and test sets.
    """

    X = dataframe.drop(columns=[target_column])

    y = dataframe[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test