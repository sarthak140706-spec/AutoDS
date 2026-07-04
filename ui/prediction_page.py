import streamlit as st
import pandas as pd

from models.loader import load_model


def show_prediction_page(model, file_path, target_column):
    """Display prediction page."""

    st.subheader("🔮 Predictions")

    st.success("✅ Best model saved successfully.")

    # ---------------- Download Model ----------------
    with open(file_path, "rb") as model_file:

        st.download_button(
            label="📥 Download Best Model",
            data=model_file,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )

    st.markdown("---")

    # ---------------- Upload Prediction Dataset ----------------
    st.subheader("📂 Upload Dataset for Prediction")

    prediction_file = st.file_uploader(
        "Choose a CSV or Excel dataset",
        type=["csv", "xlsx"],
        key="prediction_dataset"
    )

    if prediction_file is None:
        st.info("👆 Upload a dataset to generate predictions.")
        return None

    # ---------------- Load Dataset ----------------
    if prediction_file.name.endswith(".csv"):
        prediction_df = pd.read_csv(prediction_file)
    else:
        prediction_df = pd.read_excel(prediction_file)

    st.success("✅ Prediction dataset loaded successfully.")

    st.write("### Dataset Preview")
    st.dataframe(prediction_df.head())

    # ---------------- Remove Target Column ----------------
    if target_column in prediction_df.columns:
        prediction_df = prediction_df.drop(columns=[target_column])

    # ---------------- Load Saved Objects ----------------
    loaded_model, encoders, scaler, feature_columns = load_model(
        file_path
    )

    # ---------------- Validate Feature Columns ----------------
    missing_columns = [
        column for column in feature_columns
        if column not in prediction_df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing feature columns: {', '.join(missing_columns)}"
        )
        st.stop()

    # Remove extra columns
    prediction_df = prediction_df[feature_columns]

    # ---------------- Encode Categorical Columns ----------------
    for column, encoder in encoders.items():

        if column in prediction_df.columns:

            prediction_df[column] = prediction_df[column].astype(str)

            prediction_df[column] = prediction_df[column].apply(
                lambda value: value
                if value in encoder.classes_
                else encoder.classes_[0]
            )

            prediction_df[column] = encoder.transform(
                prediction_df[column]
            )

    # ---------------- Scale Numerical Columns ----------------
    numerical_columns = prediction_df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numerical_columns) > 0:

        prediction_df[numerical_columns] = scaler.transform(
            prediction_df[numerical_columns]
        )

    # ---------------- Generate Predictions ----------------
    predictions = loaded_model.predict(prediction_df)

    # ---------------- Prepare Results ----------------
    result_df = prediction_df.copy()

    result_df["Prediction"] = predictions

    # ---------------- Display Results ----------------
    st.subheader("📊 Prediction Results")

    st.dataframe(result_df)

    # ---------------- Prediction Summary ----------------
    st.subheader("📈 Prediction Summary")

    st.write(f"**Total Records:** {len(result_df)}")

    st.write("**Prediction Column:** Prediction")

    st.write(
        f"**Prediction Data Type:** {result_df['Prediction'].dtype}"
    )

    unique_predictions = result_df["Prediction"].nunique()

    st.write(f"**Unique Predictions:** {unique_predictions}")

    if unique_predictions <= 20:

        st.write("### Prediction Distribution")

        distribution = (
            result_df["Prediction"]
            .value_counts()
            .reset_index()
        )

        distribution.columns = [
            "Prediction",
            "Count"
        ]

        st.dataframe(distribution)

    # ---------------- Download Predictions ----------------
    csv = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

    return result_df