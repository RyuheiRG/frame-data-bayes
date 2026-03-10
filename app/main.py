import streamlit as st

from ingestion.data_loader import load_and_validate_data, detect_column_types, DataLoadError

from ui.eda_panel import render_eda
from ui.bayes_panel import render_bayes_engine
from ui.insights_panel import render_insights

st.set_page_config(page_title="Bayesian Engine | Zero Trust",
                   layout="wide", page_icon="🛡️")


@st.cache_data(show_spinner="Desencriptando e ingiriendo payload...", ttl=3600)
def cached_load_data(file_buffer):
    return load_and_validate_data(file_buffer)


@st.cache_data(ttl=3600)
def cached_detect_column_types(df):
    return detect_column_types(df)


st.title("🛡️ Motor de Inferencia Bayesiana")
st.markdown("---")

with st.sidebar:
    st.header("1. Ingesta de Datos")
    uploaded_file = st.file_uploader("Cargar dataset (CSV)", type=["csv"])
    if st.button("Limpiar Caché de Sesión"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

if uploaded_file is not None:
    try:

        df = cached_load_data(uploaded_file)
        st.sidebar.success(
            f"CSV montado en memoria: {df.shape[0]} filas, {df.shape[1]} dims.")

        col_types = cached_detect_column_types(df)

        tab1, tab2, tab3 = st.tabs(
            ["📊 Exploración (EDA)", "🧠 Motor Bayes", "💡 Insights Estratégicos"])

        with tab1:
            render_eda(df, col_types)

        with tab2:
            render_bayes_engine(df, col_types)

        with tab3:
            render_insights(df, col_types)

    except DataLoadError as e:
        st.error(f"Falla de Ingesta: {str(e)}")
else:
    st.info("Arquitectura en Standby. Esperando inyección de payload (CSV) en el panel lateral.")
