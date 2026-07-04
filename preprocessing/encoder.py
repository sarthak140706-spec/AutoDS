from sklearn.preprocessing import LabelEncoder

def encode_dataset(dataframe):

    df = dataframe.copy()

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_columns:
        le = LabelEncoder()

        le.fit(df[col])

        df[col] = le.transform(df[col])

    return df