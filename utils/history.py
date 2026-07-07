import streamlit as st
import pandas as pd
from datetime import datetime


def initialize_history():
    """
    Initialize experiment history.
    """

    if "experiment_history" not in st.session_state:

        st.session_state["experiment_history"] = []


def add_experiment(
    best_model,
    score,
    metric,
    problem_type
):
    """
    Add a new experiment to history.
    """

    initialize_history()

    experiment = {

        "Timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Best Model": best_model,

        "Metric": metric,

        "Score": round(score, 4),

        "Problem Type": problem_type

    }

    st.session_state["experiment_history"].append(
        experiment
    )


def get_history():
    """
    Return experiment history as DataFrame.
    """

    initialize_history()

    return pd.DataFrame(
        st.session_state["experiment_history"]
    )


def clear_history():
    """
    Clear experiment history.
    """

    st.session_state["experiment_history"] = []