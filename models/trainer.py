import streamlit as st

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.svm import (
    SVR,
    SVC
)

from utils.problem_type import detect_problem_type


def train_models(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train multiple machine learning models.
    """

    models_dict = {}

    # ---------------- Detect Problem Type ----------------

    problem_type = detect_problem_type(
        y_train
    )

    # ---------------- Classification Models ----------------

    if problem_type == "classification":

        models = {

            "Logistic Regression": LogisticRegression(
                max_iter=1000
            ),

            "Decision Tree": DecisionTreeClassifier(),

            "Random Forest": RandomForestClassifier(),

            "Support Vector Machine": SVC()

        }

    # ---------------- Regression Models ----------------

    else:

        models = {

            "Linear Regression": LinearRegression(),

            "Decision Tree": DecisionTreeRegressor(),

            "Random Forest": RandomForestRegressor(),

            "Support Vector Machine": SVR()

        }

    # ---------------- Train Models ----------------

    for model_name, model in models.items():

        try:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            models_dict[model_name] = {

                "model": model,

                "predictions": predictions,

                "problem_type": problem_type

            }

            st.success(
                f"✅ {model_name} trained successfully."
            )

        except Exception as error:

            st.warning(
                f"⚠️ {model_name} could not be trained.\n\nReason:\n{error}"
            )

            continue

    # ---------------- Check ----------------

    if len(models_dict) == 0:

        st.error(
            "❌ No models could be trained."
        )

        st.stop()

    return models_dict