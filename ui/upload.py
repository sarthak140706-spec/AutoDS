import streamlit as st

from data.loader import load_dataset
from data.validator import validate_dataset
from data.preview import show_preview

from eda.analyzer import analyze_dataset
from ui.eda_page import show_eda_page

from preprocessing.cleaner import preprocess_dataset
from preprocessing.encoder import encode_dataset
from preprocessing.scaler import scale_dataset

from ui.training_page import show_training_page
from ui.model_page import show_model_page

from models.trainer import train_models
from models.evaluator import evaluate_models


def show_upload_page():
    """Display the dataset upload page."""

    st.header("📂 Upload Dataset")

    st.write(
        "Upload a CSV or Excel dataset to begin the machine learning workflow."
    )

    uploaded_file = st.file_uploader(
        label="Choose a dataset",
        type=["csv", "xlsx"],
        help="Supported formats: CSV (.csv) and Excel (.xlsx)"
    )

    if uploaded_file is None:
        st.info("👆 Please upload a CSV or Excel file to continue.")
        return None

    # ---------------- Load Dataset ----------------
    dataframe = load_dataset(uploaded_file)

    is_valid, message = validate_dataset(dataframe)

    if not is_valid:
        st.error(message)
        return None

    st.success("✅ Dataset loaded successfully!")

    # ---------------- Preview ----------------
    show_preview(dataframe)

    # ---------------- EDA ----------------
    eda_results = analyze_dataset(dataframe)
    show_eda_page(dataframe, eda_results)

    # ---------------- Preprocessing ----------------
    processed_dataframe = preprocess_dataset(dataframe)
    encoded_dataframe = encode_dataset(processed_dataframe)
    scaled_dataframe = scale_dataset(encoded_dataframe)

    # ---------------- Train-Test Split ----------------
    X_train, X_test, y_train, y_test = show_training_page(scaled_dataframe)

    # ---------------- Train Multiple Models ----------------
    models_dict = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------- Evaluate All Models ----------------
    evaluation_results = evaluate_models(
        models_dict,
        y_test
    )

    # ---------------- Show Results ----------------
    show_model_page(
        models_dict,
        evaluation_results,
        y_test
    )

    return models_dict, evaluation_results