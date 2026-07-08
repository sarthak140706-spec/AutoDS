import streamlit as st


def initialize_settings():
    """
    Initialize default application settings.
    """

    defaults = {
        "random_seed": 42,
        "test_size": 0.2,
        "cv_folds": 5,
        "enable_shap": True,
        "enable_permutation": True,
        "enable_pdp": True,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_settings_page():
    """
    Display application settings.
    """

    st.title("⚙️ Application Settings")

    st.markdown(
        "Customize the default behavior of AutoDS."
    )

    st.divider()

    st.subheader("📊 Model Training")

    st.session_state.random_seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=st.session_state.random_seed,
    )

    st.session_state.test_size = st.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.5,
        value=float(st.session_state.test_size),
        step=0.05,
    )

    st.session_state.cv_folds = st.slider(
        "Cross Validation Folds",
        min_value=2,
        max_value=10,
        value=st.session_state.cv_folds,
    )

    st.divider()

    st.subheader("🧠 Explainability")

    st.session_state.enable_shap = st.checkbox(
        "Enable SHAP Explainability",
        value=st.session_state.enable_shap,
    )

    st.session_state.enable_permutation = st.checkbox(
        "Enable Permutation Importance",
        value=st.session_state.enable_permutation,
    )

    st.session_state.enable_pdp = st.checkbox(
        "Enable Partial Dependence Plots",
        value=st.session_state.enable_pdp,
    )

    st.divider()

    if st.button("💾 Save Settings"):

        st.success(
            "Settings saved successfully!"
        )

    st.info(
        "These settings are stored only for the current Streamlit session."
    )