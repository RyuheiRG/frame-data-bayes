import pandas as pd
import streamlit as st
from typing import Dict, List

from analytics.insights import (
    best_variable_by_mutual_info, event_rarity,
    independence_by_correlation, model_reliability_estimate
)
from visualization.visualizer import plot_correlation_heatmap, plot_missingness_bar


def render_insights(
    df: pd.DataFrame,
    col_types: Dict[str, List[str]],
    target_col: str = None,
    target_value: str = None
):
    st.header("6. Insights Estadísticos de Negocio")

    target_col = target_col or st.session_state.get('target_col')
    target_value = target_value or st.session_state.get('target_value')

    if not target_col or not target_value:
        st.info(
            "⚠️ Configura la Variable Objetivo en la pestaña 'Motor Bayes' para habilitar la telemetría.")
        return

    if target_col not in df.columns:
        st.error(
            f"Error de Sincronización: La columna '{target_col}' no existe en el payload actual.")
        return

    with st.container(border=True):
        st.subheader("📝 Diagnóstico Automático")

        prior, rarity = event_rarity(df, target_col, target_value)
        st.markdown(
            f"**1. Perfil del Evento:** La clase `{target_value}` se clasifica como **{rarity}** (Probabilidad Base: {prior:.2%})."
        )

        numeric_cols = col_types.get("numericas", [])

        if numeric_cols:
            with st.spinner("Midiendo entropía e información mutua..."):
                best_var, best_score = best_variable_by_mutual_info(
                    df, numeric_cols, target_col)

            if best_var and isinstance(best_score, (int, float)):
                st.markdown(
                    f"**2. Principal Predictor:** La variable que más información comparte con el objetivo es `{best_var}` (Score: {best_score:.4f})."
                )
            else:
                st.markdown(
                    "**2. Principal Predictor:** Datos insuficientes o error al calcular Información Mutua.")
        else:
            st.markdown(
                "**2. Principal Predictor:** N/A (Se requieren vectores numéricos).")

        ind = independence_by_correlation(df, numeric_cols)
        mean_corr = ind.get('mean_abs_corr', 0)
        safe_mean_corr = float(mean_corr) if isinstance(
            mean_corr, (int, float)) else 0.0

        st.markdown(
            f"**3. Análisis de Independencia:** {ind.get('conclusion', 'N/A')} "
            f"(Correlación absoluta media: {safe_mean_corr:.3f})."
        )

        if numeric_cols:
            with st.spinner("Validando robustez del modelo (Cross-Validation)..."):
                subset_df = df[numeric_cols + [target_col]]
                reliability = model_reliability_estimate(
                    subset_df, numeric_cols, target_col)

            baseline = reliability.get('baseline_accuracy')
            nb_acc = reliability.get('nb_cv_accuracy')

            if isinstance(baseline, float) and isinstance(nb_acc, float):
                delta = nb_acc - baseline
                trend = "🚀 Supera" if delta > 0 else "⚠️ Inferior"
                st.markdown(
                    f"**4. Rendimiento Esperado (CV):** El modelo alcanza un **{nb_acc:.2%}** de precisión estimada "
                    f"frente a un baseline trivial del **{baseline:.2%}**. ({trend} al azar por {delta:+.2%})"
                )
            else:
                note = reliability.get('note', 'Cálculo no convergente.')
                st.markdown(f"**4. Rendimiento Esperado:** {note}")

    st.subheader("Exploración Avanzada (Advanced EDA)")
    if numeric_cols:
        with st.expander("Mapa de correlaciones térmico", expanded=False):
            fig_corr = plot_correlation_heatmap(df, numeric_cols)
            if fig_corr:
                st.pyplot(fig_corr, clear_figure=True)

        with st.expander("Integridad del Dataset (Valores Faltantes)", expanded=False):
            total_nans = df.isna().sum().sum()
            if total_nans == 0:
                st.success(
                    "✅ **Dataset 100% íntegro.** No se detectaron valores nulos (NaNs). Blindaje perfecto para el motor inferencial."
                )
            else:
                st.warning(
                    f"⚠️ Se detectaron {total_nans} valores nulos en el dataset total.")
                fig_miss = plot_missingness_bar(df)
                if fig_miss:
                    st.pyplot(fig_miss, clear_figure=True)
