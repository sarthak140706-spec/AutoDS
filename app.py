import streamlit as st

st.set_page_config(
    page_title="AutoDS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AutoDS – AI Data Science Copilot")

st.markdown("---")

st.success("✅ Sprint 1: Project Setup completed successfully!")

st.markdown(
    """
Welcome to **AutoDS**.

This application will automate the complete Machine Learning workflow, including:

- 📂 Dataset Upload
- 🔍 Data Validation
- 📊 Exploratory Data Analysis (EDA)
- ⚙️ Data Preprocessing
- 🤖 Model Training
- 📈 Model Comparison
- 🎯 Hyperparameter Optimization
- 🧠 Explainable AI (SHAP)
- 📄 PDF Report Generation
- 💾 Model Saving

---
🚧 **Current Status:** Sprint 1 – Project Setup
"""
)