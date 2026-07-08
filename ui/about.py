import streamlit as st

from utils.system_info import get_system_info


def show_about_page():

    st.title("ℹ️ About AutoDS")

    st.markdown(
        """
        ## 🚀 AutoDS

        AutoDS is an Automated Machine Learning platform that simplifies the
        complete machine learning workflow from data preprocessing to model
        evaluation and explainability.

        It enables users to train multiple machine learning models with minimal
        coding while providing powerful visualizations and model interpretation.
        """
    )

    st.divider()

    st.subheader("✨ Features")

    features = [
        "📂 Dataset Upload",
        "📊 Automatic Exploratory Data Analysis (EDA)",
        "🧹 Data Cleaning & Preprocessing",
        "⚙️ Feature Engineering",
        "🤖 Multiple Machine Learning Models",
        "🔍 Hyperparameter Tuning",
        "📈 Cross Validation",
        "🧠 SHAP Explainability",
        "🔄 Permutation Feature Importance",
        "📊 Partial Dependence Plots",
        "⚖️ Fairness Analysis",
        "📡 Model Monitoring",
        "🧪 Experiment History",
    ]

    for feature in features:
        st.write(feature)

    st.divider()

    st.subheader("🛠️ Technology Stack")

    st.markdown(
        """
        - Python
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-Learn
        - SHAP
        - Matplotlib
        """
    )

    st.divider()

    st.subheader("💻 System Information")

    info = get_system_info()

    for key, value in info.items():
        st.write(f"**{key}:** {value}")

    st.divider()

    st.subheader("👨‍💻 Developer")

    st.write("**Developer:** Sarthak Jadhav")

    st.write(
        "AutoDS was developed as an end-to-end Automated Machine Learning project "
        "to simplify data preprocessing, model training, explainability, and "
        "performance analysis."
    )

    st.divider()

    st.success("Version 1.0.0")