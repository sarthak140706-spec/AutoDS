import streamlit as st

from data.loader import load_dataset
from data.validator import validate_dataset
from data.preview import show_preview

from eda.analyzer import analyze_dataset
from ui.eda_page import show_eda_page

from ui.training_page import show_training_page

from preprocessing.splitter import split_dataset
from preprocessing.cleaner import preprocess_dataset
from preprocessing.encoder import encode_dataset
from preprocessing.scaler import scale_dataset

from models.trainer import train_models
from models.evaluator import evaluate_models
from models.saver import save_model

from ui.model_page import show_model_page
from ui.prediction_page import show_prediction_page

from reports.report_generator import generate_report


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

    show_eda_page(
        dataframe,
        eda_results
    )

    # ---------------- Training Configuration ----------------
    target_column, test_size = show_training_page(
        dataframe
    )

    # ---------------- Train-Test Split ----------------
    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_dataframe,
        test_dataframe
    ) = split_dataset(
        dataframe,
        target_column,
        test_size
    )

    # ---------------- Download Split Datasets ----------------
    st.subheader("📥 Download Split Datasets")

    train_csv = train_dataframe.to_csv(
        index=False
    ).encode("utf-8")

    test_csv = test_dataframe.to_csv(
        index=False
    ).encode("utf-8")

    train_filename = (
        uploaded_file.name.rsplit(".", 1)[0]
        + "_train.csv"
    )

    test_filename = (
        uploaded_file.name.rsplit(".", 1)[0]
        + "_test.csv"
    )

    st.download_button(
        label="📥 Download Training Dataset",
        data=train_csv,
        file_name=train_filename,
        mime="text/csv"
    )

    st.download_button(
        label="📥 Download Testing Dataset",
        data=test_csv,
        file_name=test_filename,
        mime="text/csv"
    )

    # ---------------- Missing Value Handling ----------------
    X_train, X_test = preprocess_dataset(
        X_train,
        X_test
    )

    # ---------------- Encoding ----------------
    X_train, X_test, encoders = encode_dataset(
        X_train,
        X_test
    )

    # ---------------- Scaling ----------------
    X_train, X_test, scaler = scale_dataset(
        X_train,
        X_test
    )

        # ---------------- Dataset Information ----------------
    st.subheader("Processed Dataset")

    st.write(f"**Training samples:** {X_train.shape[0]}")

    st.write(f"**Testing samples:** {X_test.shape[0]}")

    st.write("### Processed Feature Dataset")

    st.dataframe(X_train.head())

    st.write("### Target Preview")

    st.dataframe(y_train.head())

    # ---------------- Train Models ----------------
    models_dict = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------- Evaluate Models ----------------
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

    # ---------------- Generate Report ----------------
    report_df = generate_report(
        dataframe,
        target_column,
        evaluation_results
    )

    st.subheader("📄 Machine Learning Report")

    st.dataframe(report_df)

    # ---------------- Download Report ----------------
    report_csv = report_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download ML Report",
        data=report_csv,
        file_name="ml_report.csv",
        mime="text/csv"
    )

    # ---------------- Save Best Model ----------------
    best_model, best_model_name, file_path = save_model(
        models_dict,
        evaluation_results,
        encoders,
        scaler,
        X_train.columns
    )

    # ---------------- Prediction ----------------
    prediction_dataframe = show_prediction_page(
        best_model,
        file_path,
        target_column
    )

    return prediction_dataframe, report_df