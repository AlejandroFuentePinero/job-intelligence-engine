import streamlit as st

from src.job_intel.app import home, landscape, recommender, upskilling_macro


st.set_page_config(page_title="Job Intelligence Engine", layout="wide")

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

st.sidebar.radio(
    "Navigation",
    ["Home", "Landscape", "Recommender", "Upskilling"],
    key="page",
)

page = st.session_state["page"]

if page == "Home":
    home.render()
elif page == "Landscape":
    landscape.render()
elif page == "Recommender":
    recommender.render()
else:
    upskilling_macro.render()
