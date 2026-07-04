import streamlit as st
from preprocessing.splitter import split_dataset

def show_training_page(dataframe):
    # Display heading
    st.header("Training Page")

    # Display dropdown containing all column names
    # User selects target column
    target_column = st.selectbox("Select the target column", dataframe.columns)

    # Call split_dataset()
    X_train, X_test, y_train, y_test = split_dataset(dataframe, target_column)

    # Display: Training samples / Testing samples
    st.write(f"Training samples: {X_train.shape[0]}")
    st.write(f"Testing samples: {X_test.shape[0]}")

    st.write("Processed feature dataset preview")
    st.dataframe(X_train.head(5))

    st.write("Processed target preview")
    st.dataframe(y_train.head(5))
    
    return X_train, X_test, y_train, y_test