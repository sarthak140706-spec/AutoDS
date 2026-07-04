from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple machine learning models automatically
    based on the problem type.
    """

    trained_models = {}

    # ---------------- Detect Problem Type ----------------
    is_regression = (
        y_train.dtype.kind in "if"
        and y_train.nunique() > 10
    )

    # ---------------- Regression Models ----------------
    if is_regression:

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(random_state=42)
        }

    # ---------------- Classification Models ----------------
    else:

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            "Random Forest Classifier": RandomForestClassifier(random_state=42)
        }

    # ---------------- Train Every Model ----------------
    for model_name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        trained_models[model_name] = {
            "model": model,
            "predictions": predictions
        }

    return trained_models