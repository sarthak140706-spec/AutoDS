import streamlit as st
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)

from models.loader import load_model


def show_prediction_page(
    model,
    file_path,
    target_column
):
    """
    Prediction page.
    Supports:
    - Prediction only
    - Actual vs Predicted comparison
    - Download prediction results
    """

    st.subheader("🔮 Predictions")

    st.success("✅ Best model saved successfully.")

    # ----------------------------------------------------
    # Download Best Model
    # ----------------------------------------------------

    with open(file_path, "rb") as model_file:

        st.download_button(
            label="📥 Download Best Model",
            data=model_file,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )

    st.markdown("---")

    # ----------------------------------------------------
    # Upload Prediction Dataset
    # ----------------------------------------------------

    st.subheader("📂 Upload Dataset for Prediction")

    prediction_file = st.file_uploader(
        "Choose CSV or Excel File",
        type=["csv", "xlsx"],
        key="prediction_dataset"
    )

    if prediction_file is None:

        st.info(
            "Upload a dataset to generate predictions."
        )

        return None

    # ----------------------------------------------------
    # Read Dataset
    # ----------------------------------------------------

    if prediction_file.name.endswith(".csv"):

        uploaded_df = pd.read_csv(
            prediction_file
        )

    else:

        uploaded_df = pd.read_excel(
            prediction_file
        )

    st.success(
        "Prediction dataset loaded successfully."
    )

    st.write("### Uploaded Dataset")

    st.dataframe(
        uploaded_df.head()
    )

    # ----------------------------------------------------
    # Preserve Original Dataset
    # ----------------------------------------------------

    display_df = uploaded_df.copy()

    prediction_df = uploaded_df.copy()

    actual_values = None

    if target_column in prediction_df.columns:

        actual_values = prediction_df[
            target_column
        ].copy()

        prediction_df.drop(
            columns=[target_column],
            inplace=True
        )

    # ----------------------------------------------------
    # Load Saved Objects
    # ----------------------------------------------------

    (
        loaded_model,
        encoders,
        scaler,
        feature_columns
    ) = load_model(file_path)

        # ----------------------------------------------------
    # Encode Categorical Features
    # ----------------------------------------------------

    for column, encoder in encoders.items():

        if column in prediction_df.columns:

            prediction_df[[column]] = encoder.transform(
                prediction_df[[column]].astype(str)
            )

    # ----------------------------------------------------
    # Scale Numerical Features
    # ----------------------------------------------------

    numerical_columns = prediction_df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numerical_columns) > 0:

        prediction_df[numerical_columns] = scaler.transform(
            prediction_df[numerical_columns]
        )

    # ----------------------------------------------------
    # Match Training Feature Order
    # ----------------------------------------------------

    prediction_df = prediction_df[
        feature_columns
    ]

    # ----------------------------------------------------
    # Generate Predictions
    # ----------------------------------------------------

    predictions = loaded_model.predict(
        prediction_df
    )

    # ----------------------------------------------------
    # Create Final Output Table
    # ----------------------------------------------------

    result_df = display_df.copy()

    result_df["Predicted Value"] = predictions

    # ----------------------------------------------------
    # Evaluation Mode
    # ----------------------------------------------------

    if actual_values is not None:

        result_df["Prediction Error"] = (
            actual_values.values -
            predictions
        )

        st.subheader(
            "📊 Actual vs Predicted"
        )

        st.dataframe(result_df)

        st.subheader(
            "📈 Evaluation Metrics"
        )

        try:

            mae = mean_absolute_error(
                actual_values,
                predictions
            )

            mse = mean_squared_error(
                actual_values,
                predictions
            )

            rmse = mse ** 0.5

            r2 = r2_score(
                actual_values,
                predictions
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "MAE",
                    f"{mae:.4f}"
                )

            with col2:

                st.metric(
                    "RMSE",
                    f"{rmse:.4f}"
                )

            with col3:

                st.metric(
                    "R² Score",
                    f"{r2:.4f}"
                )

        except Exception:

            accuracy = accuracy_score(
                actual_values,
                predictions
            )

            st.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

        # ----------------------------------------------------
    # Prediction Only Mode
    # ----------------------------------------------------

    else:

        st.subheader(
            "📊 Prediction Results"
        )

        st.dataframe(result_df)

    # ----------------------------------------------------
    # Download Prediction Results
    # ----------------------------------------------------

    prediction_csv = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Results",
        data=prediction_csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

    # ----------------------------------------------------
    # Prediction Summary
    # ----------------------------------------------------

    st.subheader("📋 Prediction Summary")

    st.write(
        f"**Total Samples:** {len(result_df)}"
    )

    st.write(
        f"**Predictions Generated:** {len(predictions)}"
    )

    if actual_values is not None:

        st.success(
            "Evaluation completed successfully."
        )

    else:

        st.info(
            "Actual target column not found. Showing predictions only."
        )

    return result_df