import streamlit as st


def show_training_page(dataframe):
    """
    Display the training page and let the user select the target column.
    """

    st.header("Training Page")

    target_column = st.selectbox(
        "Select the target column",
        options=dataframe.columns,
        index=None,
        placeholder="Choose the target column"
    )

    if target_column is None:
        st.info("👆 Please select the target column to continue.")
        st.stop()

    return target_column