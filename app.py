import streamlit as st

from src.job_intel.app import home, landscape, recommender, upskilling_macro
from src.job_intel.app.engine import get_build_info


def _render_build_info_sidebar() -> None:
    info = get_build_info()

    with st.sidebar.expander("Build info", expanded=False):
        st.write(f"Commit: {info.get('git_commit') or 'unknown'}")
        st.write(
            f"Assets updated (UTC): {info.get('assets_updated_at_utc') or 'unknown'}"
        )
        st.write(f"Python: {info.get('python_version') or 'unknown'}")

        missing = info.get("missing_required_assets") or []
        if missing:
            st.warning("Missing required assets:\n- " + "\n- ".join(missing))


st.set_page_config(page_title="Job Intelligence Engine", layout="wide")

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Sidebar: navigation first
st.sidebar.radio(
    "Navigation",
    ["Home", "Landscape", "Recommender", "Upskilling"],
    key="page",
)

# Sidebar: build info at the bottom
st.sidebar.markdown("---")
_render_build_info_sidebar()

page = st.session_state["page"]

if page == "Home":
    home.render()
elif page == "Landscape":
    landscape.render()
elif page == "Recommender":
    recommender.render()
else:
    upskilling_macro.render()
