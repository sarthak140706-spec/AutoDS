import streamlit as st

from ui.upload import show_upload_page

st.set_page_config(
    page_title="AutoDS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AutoDS – AI Data Science Copilot")

st.markdown("---")

st.sidebar.title("Navigation")
st.sidebar.success("Sprint 2")

df=show_upload_page()