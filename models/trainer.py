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

from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score
)

from utils.problem_type import (
    detect_problem_type
)

from utils.hyperparameter_grids import (
    REGRESSION_GRIDS,
    CLASSIFICATION_GRIDS
)


def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    enable_hyperparameter_tuning=False
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

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "Decision Tree":
                DecisionTreeClassifier(),

            "Random Forest":
                RandomForestClassifier(),

            "Support Vector Machine":
                SVC()

        }

        parameter_grids = CLASSIFICATION_GRIDS

        scoring = "accuracy"

    # ---------------- Regression Models ----------------

    else:

        models = {

            "Linear Regression":
                LinearRegression(),

            "Decision Tree":
                DecisionTreeRegressor(),

            "Random Forest":
                RandomForestRegressor(),

            "Support Vector Machine":
                SVR()

        }

        parameter_grids = REGRESSION_GRIDS

        scoring = "r2"

    # ---------------- Train Models ----------------

    for model_name, model in models.items():

        try:

            best_params = {}

            best_cv_score = None

            cv_scores = None

            cv_mean = None

            cv_std = None

            # -----------------------------------------
            # Hyperparameter Tuning
            # -----------------------------------------

            if enable_hyperparameter_tuning:

                param_grid = parameter_grids.get(
                    type(model),
                    {}
                )

                if param_grid:

                    st.info(
                        f"🔍 Tuning {model_name}..."
                    )

                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=5,
                        scoring=scoring,
                        n_jobs=-1
                    )

                    grid_search.fit(
                        X_train,
                        y_train
                    )

                    model = grid_search.best_estimator_

                    best_params = grid_search.best_params_

                    best_cv_score = round(
                        grid_search.best_score_,
                        4
                    )

                else:

                    model.fit(
                        X_train,
                        y_train
                    )

            else:

                model.fit(
                    X_train,
                    y_train
                )

            predictions = model.predict(
                X_test
            )

            # -----------------------------------------
            # Cross Validation Statistics
            # -----------------------------------------

            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring=scoring,
                n_jobs=-1
            )

            cv_mean = round(
                cv_scores.mean(),
                4
            )

            cv_std = round(
                cv_scores.std(),
                4
            )

            # -----------------------------------------
            # Store Results
            # -----------------------------------------

            models_dict[model_name] = {

                "model": model,

                "predictions": predictions,

                "problem_type": problem_type,

                "best_params": best_params,

                "best_cv_score": best_cv_score,

                "cv_scores": cv_scores,

                "cv_mean": cv_mean,

                "cv_std": cv_std

            }

            # -----------------------------------------
            # Success Message
            # -----------------------------------------

            if enable_hyperparameter_tuning:

                if best_cv_score is not None:

                    st.success(

                        f"✅ {model_name} trained successfully "
                        f"(GridSearch CV: {best_cv_score}, "
                        f"5-Fold CV Mean: {cv_mean})"

                    )

                else:

                    st.success(

                        f"✅ {model_name} trained successfully "
                        f"(5-Fold CV Mean: {cv_mean})"

                    )

            else:

                st.success(

                    f"✅ {model_name} trained successfully "
                    f"(5-Fold CV Mean: {cv_mean})"

                )
        except Exception as error:

            st.warning(

                f"⚠️ {model_name} could not be trained.\n\n"
                f"Reason:\n{error}"

            )

            continue

    # -----------------------------------------
    # Final Check
    # -----------------------------------------

    if len(models_dict) == 0:

        st.error(

            "❌ No models could be trained."

        )

        st.stop()

    return models_dict
            