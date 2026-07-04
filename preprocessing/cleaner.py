def preprocess_dataset(dataframe):
    """Placeholder for preprocessing pipeline."""

    df = dataframe.copy()

    numerical_col = df.select_dtypes(include='number').columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in numerical_col:
        if df[col].isnull().any():
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)

    for col in categorical_columns:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    return df