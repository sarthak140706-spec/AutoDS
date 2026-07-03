import pandas as pd


def load_dataset(uploaded_file):
    """
    Load the uploaded dataset into a Pandas DataFrame.
    """

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    return None