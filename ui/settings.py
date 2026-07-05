import streamlit as st


def show_settings_panel():
    """
    Display the project settings panel in the sidebar.
    Returns a dictionary containing all user-selected settings.
    """

    st.sidebar.header("⚙️ Project Settings")

    # ---------------- Train-Test Split ----------------
    split_option = st.sidebar.selectbox(
        "Train-Test Split",
        (
            "70 : 30",
            "80 : 20",
            "90 : 10"
        ),
        index=1
    )

    split_mapping = {
        "70 : 30": 0.30,
        "80 : 20": 0.20,
        "90 : 10": 0.10
    }

    test_size = split_mapping[split_option]

    # ---------------- Random State ----------------
    random_state = st.sidebar.number_input(
        "Random State",
        min_value=0,
        max_value=9999,
        value=42,
        step=1
    )

    # ---------------- Problem Type ----------------
    problem_type = st.sidebar.selectbox(
        "Problem Type",
        (
            "Auto Detect",
            "Regression",
            "Classification"
        ),
        index=0
    )

    # ---------------- Preprocessing ----------------
    st.sidebar.markdown("---")

    st.sidebar.subheader("Preprocessing")

    scaling = st.sidebar.checkbox(
        "Apply Feature Scaling",
        value=True
    )

    encoding = st.sidebar.checkbox(
        "Encode Categorical Features",
        value=True
    )

    # ---------------- Model Settings ----------------
    st.sidebar.markdown("---")

    st.sidebar.subheader("Model")

    save_model = st.sidebar.checkbox(
        "Save Best Model",
        value=True
    )

    # ---------------- Summary ----------------
    st.sidebar.markdown("---")

    st.sidebar.subheader("Current Configuration")

    st.sidebar.write(f"**Test Size:** {int(test_size * 100)}%")
    st.sidebar.write(f"**Random State:** {random_state}")
    st.sidebar.write(f"**Problem Type:** {problem_type}")

    settings = {
        "test_size": test_size,
        "random_state": random_state,
        "problem_type": problem_type,
        "scaling": scaling,
        "encoding": encoding,
        "save_model": save_model
    }

    return settings