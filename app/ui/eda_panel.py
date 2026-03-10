import streamlit as st
import matplotlib.pyplot as plt
from visualization.visualizer import plot_histogram, plot_time_series


def render_eda(df, col_types):
    st.header("2. Escaneo de Superficie y EDA")

    with st.expander("Ver mapa de tipos de datos detectados", expanded=False):
        st.json(col_types)

    if col_types["numericas"]:
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            hist_col = st.selectbox(
                "Selecciona variable para Histograma", col_types["numericas"])
            fig_hist = plot_histogram(df, hist_col)
            st.pyplot(fig_hist)
            plt.close(fig_hist)

        with col_eda2:
            if col_types["fechas"]:
                date_col = st.selectbox(
                    "Selecciona Vector de Tiempo", col_types["fechas"])
                fig_time = plot_time_series(df, date_col, hist_col)
                if fig_time:
                    st.pyplot(fig_time)
                    plt.close(fig_time)
            else:
                st.info("No se detectaron vectores de tiempo (datetime).")
