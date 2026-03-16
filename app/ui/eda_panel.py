import streamlit as st
import pandas as pd
from typing import Dict, List
import logging

from visualization.visualizer import plot_histogram, plot_time_series


def render_eda(df: pd.DataFrame, col_types: Dict[str, List[str]]):
    st.header("2. Escaneo de Superficie y EDA")

    with st.expander("Ver mapa de tipos de datos detectados", expanded=False):
        st.json(col_types)

    numericas = col_types.get("numericas", [])
    fechas = col_types.get("fechas", [])

    if numericas:
        col_eda1, col_eda2 = st.columns(2)

        with col_eda1:
            hist_col = st.selectbox(
                "Selecciona variable para Histograma", numericas)
            try:
                with st.spinner("Renderizando densidad..."):
                    fig_hist = plot_histogram(df, hist_col)

                    st.pyplot(fig_hist, clear_figure=True)
            except Exception as e:
                st.error(f"Falla de renderizado (Histograma): {str(e)}")
                logging.error(f"EDA Panel Error - Histograma: {str(e)}")

        with col_eda2:
            if fechas:
                date_col = st.selectbox("Selecciona Vector de Tiempo", fechas)
                try:
                    with st.spinner("Mapeando línea temporal..."):
                        fig_time = plot_time_series(df, date_col, hist_col)

                        if fig_time:
                            st.pyplot(fig_time, clear_figure=True)
                        else:
                            st.warning(
                                "Vector de tiempo incompatible o vacío.")
                except Exception as e:
                    st.error(
                        f"Falla de renderizado (Serie Temporal): {str(e)}")
                    logging.error(f"EDA Panel Error - Time Series: {str(e)}")
            else:
                st.info(
                    "No se detectaron vectores de tiempo (datetime) para el escaneo cronológico.")
