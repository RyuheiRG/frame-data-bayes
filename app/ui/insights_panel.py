import streamlit as st
import matplotlib.pyplot as plt

from analytics.insights import (
    best_variable_by_mutual_info, event_rarity,
    independence_by_correlation, model_reliability_estimate
)

from visualization.visualizer import plot_correlation_heatmap, plot_missingness_bar


def render_insights(df, col_types):
    st.header("6. Insights Estadísticos de Negocio")

    target_col = st.session_state.get('target_col')
    target_value = st.session_state.get('target_value')

    if not target_col or not target_value:
        st.info(
            "Configura la Variable Objetivo en la pestaña 'Motor Bayes' para habilitar los insights.")
        return

    with st.container(border=True):
        st.subheader("📝 Diagnóstico Automático")

        prior, rarity = event_rarity(df, target_col, target_value)
        st.markdown(
            f"**1. Perfil del Evento:** La clase `{target_value}` se clasifica como **{rarity}** (Probabilidad Base: {prior:.2%}).")

        if col_types["numericas"]:
            with st.spinner("Midiendo entropía..."):
                best_var, best_score = best_variable_by_mutual_info(
                    df, col_types["numericas"], target_col)
            if best_var:
                st.markdown(
                    f"**2. Principal Predictor:** La variable que más información comparte con el objetivo es `{best_var}` (Score: {best_score:.4f}).")
            else:
                st.markdown(
                    "**2. Principal Predictor:** Datos insuficientes para calcular Información Mutua.")

        ind = independence_by_correlation(df, col_types.get("numericas", []))
        st.markdown(
            f"**3. Análisis de Independencia:** {ind['conclusion']} (Correlación absoluta media: {ind.get('mean_abs_corr', 0):.3f}).")

        with st.spinner("Validando modelo (Cross-Validation)..."):
            reliability = model_reliability_estimate(
                df, col_types.get("numericas", []), target_col)
        baseline = reliability.get('baseline_accuracy')
        nb_acc = reliability.get('nb_cv_accuracy')

        if baseline and nb_acc:
            st.markdown(
                f"**4. Rendimiento Esperado (CV):** El modelo alcanza un **{nb_acc:.2%}** de precisión estimada frente a un baseline del **{baseline:.2%}**.")
        else:
            st.markdown(
                f"**4. Rendimiento Esperado:** {reliability.get('note', 'No aplicable')}")

    st.subheader("Exploración Avanzada (Advanced EDA)")
    if col_types["numericas"]:
        with st.expander("Mapa de correlaciones térmico", expanded=False):
            fig_corr = plot_correlation_heatmap(df, col_types["numericas"])
            if fig_corr:
                st.pyplot(fig_corr)
                plt.close(fig_corr)

        with st.expander("Integridad del Dataset (Valores Faltantes)", expanded=False):
            if df.isna().sum().sum() == 0:
                st.success(
                    "✅ **Dataset 100% íntegro.** No se detectaron valores nulos (NaNs) en ninguna dimensión. Blindaje perfecto para el motor inferencial.")
            else:
                fig_miss = plot_missingness_bar(df)
                if fig_miss:
                    st.pyplot(fig_miss)
                    plt.close(fig_miss)
