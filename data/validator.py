def validate_dataset(df):
    if df is None:
        return False, "Dataset does not exist."
    if df.shape[0] == 0:
        return False, "Dataset has no rows."
    if df.shape[1] == 0:
        return False, "Dataset has no columns."
    
    return True, ""