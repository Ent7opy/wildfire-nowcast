import streamlit as st

st.set_page_config(page_title="Wildfire Nowcast & Forecast", layout="wide")

st.title("Wildfire Nowcast & Forecast")
st.write(
    "This Streamlit placeholder demonstrates the start of the map-based interface. "
    "Use this page to wire up the real dynamic layers once the API and data are ready."
)
st.info(
    "- Hook this page up to live map tiles once the API is ready.\n"
    "- Add controls for filters / layer toggles before enabling the export tools.\n"
    "- Integrate the `/health` check once the backend is available."
)

