import streamlit as st
import io
import pandas as pd

from ingestion.data_loader import load_and_validate_data, detect_column_types, DataLoadError
from ui.eda_panel import render_eda
from ui.bayes_panel import render_bayes_engine
from ui.insights_panel import render_insights

st.set_page_config(page_title="Bayesian Engine | Zero Trust",
                   layout="wide", page_icon="🛡️")


@st.cache_data(show_spinner="Desencriptando e ingiriendo payload...", ttl=3600)
def cached_load_data(raw_bytes: bytes) -> pd.DataFrame:
    return load_and_validate_data(io.BytesIO(raw_bytes))


def nuke_session_state():
    """Callback de ejecución inmediata cuando cambia el input file."""
    st.session_state.clear()


def main():

    st.title("🛡️ Motor de Inferencia Bayesiana")
    st.markdown("---")

    with st.sidebar:
        st.header("1. Ingesta de Datos")

        uploaded_file = st.file_uploader(
            "Cargar dataset (CSV)",
            type=["csv"],
            on_change=nuke_session_state
        )

    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue()
            df = cached_load_data(raw_bytes)

            st.sidebar.success(
                f"CSV montado en memoria: {df.shape[0]} filas, {df.shape[1]} dims.")
            col_types = detect_column_types(df)

            tab1, tab2, tab3 = st.tabs(
                ["📊 Exploración (EDA)", "🧠 Motor Bayes", "💡 Insights Estratégicos"])

            with tab1:
                render_eda(df, col_types)
            with tab2:
                render_bayes_engine(df, col_types)
            with tab3:
                render_insights(df, col_types)

        except DataLoadError as e:
            st.error(f"Falla de Ingesta (Capa de Datos): {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"Excepción Crítica No Controlada: {str(e)}")
            st.stop()
    else:
        st.info(
            "Arquitectura en Standby. Esperando inyección de payload (CSV) en el panel lateral.")


if __name__ == "__main__":
    main()
