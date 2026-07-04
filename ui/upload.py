import streamlit as st

from data.loader import load_dataset
from data.validator import validate_dataset
from data.preview import show_preview
from eda.analyzer import analyze_dataset
from preprocessing.cleaner import preprocess_dataset
from preprocessing.encoder import encode_dataset
from preprocessing.scaler import scale_dataset
from ui.training_page import show_training_page
from ui.eda_page import show_eda_page


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

    dataframe = load_dataset(uploaded_file)

    is_valid,message = validate_dataset(dataframe)

    if not is_valid:
        st.error(message)
        return None

    st.success("✅ Dataset loaded successfully!")

    show_preview(dataframe)

    eda_results = analyze_dataset(dataframe)
    show_eda_page(dataframe,eda_results)

    processed_dataframe = preprocess_dataset(dataframe)
    encoded_dataframe = encode_dataset(processed_dataframe)
    scaled_dataframe = scale_dataset(encoded_dataframe)
    show_training_page(scaled_dataframe)

    return scaled_dataframe