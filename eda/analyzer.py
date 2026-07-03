def analyze_dataset(dataframe):
    """Display the EDA section."""

    # ---------------- NUMERICAL ----------------
    numerical_columns = dataframe.select_dtypes(include='number').columns.tolist()
    num_count = len(numerical_columns)
    desc_stats = dataframe[numerical_columns].describe()

    # ---------------- CATEGORICAL ----------------
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()

    cat_unique_counts = {}
    cat_most_frequent = {}

    for col in categorical_columns:
        cat_unique_counts[col] = dataframe[col].nunique()
        cat_most_frequent[col] = dataframe[col].mode()[0]

    # ---------------- MISSING VALUES ----------------
    missing_counts = dataframe.isnull().sum()
    missing_percentage = (missing_counts / dataframe.shape[0]) * 100
    total_missing = missing_counts.sum()

    # ---------------- CORRELATION ----------------
    if len(numerical_columns) > 1:
        correlation_matrix = dataframe[numerical_columns].corr()
    else:
        correlation_matrix = None

    return {
        "num_count": num_count,
        "columns": numerical_columns,
        "stats": desc_stats,

        "categorical_columns": categorical_columns,
        "cat_unique_counts": cat_unique_counts,
        "cat_most_frequent": cat_most_frequent,

        "missing_counts": missing_counts,
        "missing_percentage": missing_percentage,
        "total_missing": total_missing,

        "correlation_matrix": correlation_matrix
    }