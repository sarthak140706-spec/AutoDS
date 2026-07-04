from sklearn.preprocessing import StandardScaler

def scale_dataset(dataframe):

    df = dataframe.copy()

    numerical_col = df.select_dtypes(include='number').columns.tolist()

    if len(numerical_col)>0:

        scaler = StandardScaler()

        scaled_values = scaler.fit_transform(df[numerical_col])

        df[numerical_col] = scaled_values

    return df