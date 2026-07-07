"""
Hyperparameter grids used for automatic model tuning.
"""

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

# ==========================================================
# Regression Hyperparameter Grids
# ==========================================================

REGRESSION_GRIDS = {

    LinearRegression: {},

    DecisionTreeRegressor: {

        "max_depth": [None, 5, 10, 20],

        "min_samples_split": [2, 5, 10],

        "min_samples_leaf": [1, 2, 4]

    },

    RandomForestRegressor: {

        "n_estimators": [100, 200],

        "max_depth": [None, 10, 20],

        "min_samples_split": [2, 5],

        "min_samples_leaf": [1, 2]

    },

    SVR: {

        "C": [0.1, 1, 10],

        "kernel": [

            "linear",

            "rbf"

        ],

        "gamma": [

            "scale",

            "auto"

        ]

    }

}


# ==========================================================
# Classification Hyperparameter Grids
# ==========================================================

CLASSIFICATION_GRIDS = {

    LogisticRegression: {

        "C": [0.1, 1, 10],

        "solver": [

            "lbfgs"

        ],

        "max_iter": [

            1000

        ]

    },

    DecisionTreeClassifier: {

        "max_depth": [None, 5, 10, 20],

        "min_samples_split": [2, 5, 10],

        "min_samples_leaf": [1, 2, 4]

    },

    RandomForestClassifier: {

        "n_estimators": [100, 200],

        "max_depth": [None, 10, 20],

        "min_samples_split": [2, 5],

        "min_samples_leaf": [1, 2]

    },

    SVC: {

        "C": [0.1, 1, 10],

        "kernel": [

            "linear",

            "rbf"

        ],

        "gamma": [

            "scale",

            "auto"

        ]

    }

}