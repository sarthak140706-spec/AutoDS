import streamlit as st


def show_training_page(dataframe):
    """
    Display training configuration page.
    """

    st.header("⚙️ Training Configuration")

    # ---------------- Target Column ----------------
    target_column = st.selectbox(
        "Select the Target Column",
        options=dataframe.columns,
        index=None,
        placeholder="Choose the target column"
    )

    if target_column is None:
        st.info("👆 Please select the target column to continue.")
        st.stop()

    # ---------------- Train-Test Split ----------------
    split_option = st.selectbox(
        "Select Train-Test Split",
        options=[
            "80 : 20",
            "75 : 25",
            "70 : 30",
            "60 : 40"
        ],
        index=0
    )

    split_mapping = {
        "80 : 20": 0.20,
        "75 : 25": 0.25,
        "70 : 30": 0.30,
        "60 : 40": 0.40
    }

    test_size = split_mapping[split_option]

    st.success(
        f"Selected Target Column: {target_column}"
    )

    st.success(
        f"Train-Test Split: {split_option}"
    )

    return target_column, test_size