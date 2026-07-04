import joblib


def load_model(file_path):
    """
    Load the saved model and preprocessing objects.
    """

    saved_objects = joblib.load(file_path)

    model = saved_objects["model"]

    encoders = saved_objects["encoders"]

    scaler = saved_objects["scaler"]

    feature_columns = saved_objects["feature_columns"]

    return (
        model,
        encoders,
        scaler,
        feature_columns
    )