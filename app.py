import streamlit as st

from ui.upload import show_upload_page
from ui.settings import show_settings_panel
from ui.about import show_about_page

st.set_page_config(
    page_title="AutoDS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🤖 AutoDS")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 AutoDS",
        "⚙️ Settings",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.title("Progress")
st.sidebar.success(
    "🚀 Sprint 10 - Final Release"
)

# ---------------- PAGES ----------------

if page == "🏠 AutoDS":

    st.title("🤖 AutoDS – AI Data Science Copilot")

    st.markdown("---")

    show_upload_page()

elif page == "⚙️ Settings":

    show_settings_panel()

elif page == "ℹ️ About":

    show_about_page()