import streamlit as st

from data.loader import load_dataset
from data.validator import validate_dataset
from data.preview import show_preview

from eda.analyzer import analyze_dataset
from ui.eda_page import show_eda_page

from ui.training_page import show_training_page
from ui.settings import show_settings_panel

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
from utils.history import add_experiment


def show_upload_page():
    """Display the dataset upload page."""

    try:

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

            st.info(
                "👆 Please upload a CSV or Excel file to continue."
            )

            return None

        # ---------------- Load Dataset ----------------

        dataframe = load_dataset(
            uploaded_file
        )

        is_valid, message = validate_dataset(
            dataframe
        )

        if not is_valid:

            st.error(message)

            return None

        st.success(
            "✅ Dataset loaded successfully!"
        )

        # ---------------- Project Settings ----------------

        settings = show_settings_panel()

        # ---------------- Preview ----------------

        show_preview(
            dataframe
        )

        # ---------------- EDA ----------------

        eda_results = analyze_dataset(
            dataframe
        )

        show_eda_page(
            dataframe,
            eda_results
        )

        # ---------------- Training Configuration ----------------

        target_column, _ = show_training_page(
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
            test_size=settings["test_size"],
            random_state=settings["random_state"]
        )

        # ---------------- Download Split Datasets ----------------

        st.subheader(
            "📥 Download Split Datasets"
        )

        train_csv = train_dataframe.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        test_csv = test_dataframe.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

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
            X_test,
            enable_encoding=settings["encoding"]
        )

        # ---------------- Scaling ----------------

        X_train, X_test, scaler = scale_dataset(
            X_train,
            X_test,
            enable_scaling=settings["scaling"]
        )

        # ---------------- Dataset Information ----------------

        st.subheader("Processed Dataset")

        st.write(
            f"**Training samples:** {X_train.shape[0]}"
        )

        st.write(
            f"**Testing samples:** {X_test.shape[0]}"
        )

        st.write("### Processed Feature Dataset")

        st.dataframe(
            X_train.head()
        )

        st.write("### Target Preview")

        st.dataframe(
            y_train.head()
        )

        # ---------------- Train Models ----------------

        models_dict = train_models(
            X_train,
            X_test,
            y_train,
            y_test,
            enable_hyperparameter_tuning=settings[
                "hyperparameter_tuning"
            ]
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
            X_train,
            X_test,
            y_test
        )

        # ---------------- Save Experiment ----------------

        first_metrics = list(
            evaluation_results.values()
        )[0]

        if "R2 Score" in first_metrics:

            metric_name = "R2 Score"

        else:

            metric_name = "Accuracy"

        best_model_name = max(

            evaluation_results,

            key=lambda model:
            evaluation_results[model][metric_name]

        )

        problem_type = models_dict[
            best_model_name
        ]["problem_type"]

        add_experiment(

            best_model=best_model_name,

            score=evaluation_results[
                best_model_name
            ][metric_name],

            metric=metric_name,

            problem_type=problem_type

        )
        
        # ---------------- Generate Report ----------------

        report_df = generate_report(
            dataframe,
            target_column,
            evaluation_results
        )

        st.subheader(
            "📄 Machine Learning Report"
        )

        st.dataframe(
            report_df
        )

        # ---------------- Download Report ----------------

        report_csv = report_df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

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
            X_train.columns,
            save_model_enabled=settings["save_model"]
        )

        # ---------------- Prediction ----------------

        if settings["save_model"]:

            prediction_dataframe = show_prediction_page(
                best_model,
                file_path,
                target_column
            )

        else:

            prediction_dataframe = None

            st.info(
                "Prediction module is disabled because "
                "'Save Best Model' is turned OFF."
            )

        return (
            prediction_dataframe,
            report_df
        )

    except FileNotFoundError:

        st.error(
            "❌ Required file was not found."
        )

        return None

    except ValueError as error:

        st.error(
            f"❌ Invalid input:\n\n{error}"
        )

        return None

    except KeyError as error:

        st.error(
            f"❌ Missing required column:\n\n{error}"
        )

        return None

    except Exception as error:

        st.error(
            "❌ An unexpected error occurred while processing the dataset."
        )

        with st.expander(
            "🔍 Show Technical Details"
        ):

            st.exception(error)

        return None