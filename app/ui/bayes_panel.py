import pandas as pd
import streamlit as st
import numpy as np
from typing import Dict, List

from models.bayes_engine import BayesianAnalyzer
from visualization.visualizer import plot_probability_comparison, plot_confusion_matrix


def _purge_dependent_state():
    keys_to_purge = ['target_value', 'model_metrics']
    for key in keys_to_purge:
        if key in st.session_state:
            del st.session_state[key]


def render_bayes_engine(df: pd.DataFrame, col_types: Dict[str, List[str]]):
    st.header("3. Configuración del Objetivo (Target)")

    possible_targets = list(
        set(col_types.get("binarias", []) + col_types.get("categoricas", [])))

    if not possible_targets:
        st.error(
            "Vector de falla: No se detectaron columnas categóricas/binarias válidas para usar como objetivo.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox(
            "Selecciona la Variable Objetivo (A)",
            possible_targets,
            key="target_col",
            on_change=_purge_dependent_state
        )
    with col2:
        classes_sorted = sorted(
            df[target_col].dropna().unique(), key=lambda x: str(x))
        target_value = st.selectbox(
            "Selecciona la clase positiva (Evento Anómalo)",
            classes_sorted,
            key="target_value"
        )

    engine = BayesianAnalyzer(df, target_col)

    st.markdown("---")
    st.header("4. Motor Probabilístico (Teorema de Bayes)")
    prior_a = engine.calculate_prior(target_value)
    st.metric(
        label=f"P(A) -> Probabilidad Base de '{target_value}'", value=f"{prior_a:.4f}")

    st.subheader("Evaluación de Evidencia Condicional")

    if not col_types.get("numericas"):
        st.warning(
            "No hay vectores numéricos disponibles para la evaluación de evidencia condicional.")
        return

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        evidence_col = st.selectbox(
            "Selecciona Variable de Evidencia (B)", col_types["numericas"])
    with col_ev2:
        raw_mean = df[evidence_col].mean()
        safe_mean = float(raw_mean) if not np.isnan(raw_mean) else 0.0
        umbral = st.number_input(
            f"Umbral: {evidence_col} > X", value=safe_mean)

    condition_mask = df[evidence_col] > umbral
    prob_b_given_a = engine.calculate_conditional(
        evidence_col, condition_mask, target_value)

    total_b_subset = df[condition_mask]
    prob_b = len(total_b_subset) / len(df) if not df.empty else 0.0

    posterior = (prob_b_given_a * prior_a) / prob_b if prob_b > 0 else 0.0

    st.latex(r"P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}")

    res1, res2, res3 = st.columns(3)
    res1.metric(label="P(B | A) [Sensibilidad Evidencia]",
                value=f"{prob_b_given_a:.4f}")
    res2.metric(label="P(B) [Probabilidad Evidencia]", value=f"{prob_b:.4f}")
    res3.metric(label="P(A | B) [Probabilidad Posterior]",
                value=f"{posterior:.4f}", delta=f"{(posterior - prior_a):.4f} vs Prior")

    fig_bayes = plot_probability_comparison(
        prior_a, posterior, str(target_value))
    st.pyplot(fig_bayes, clear_figure=True)

    st.markdown("---")
    st.header("5. Clasificador Naïve Bayes (GaussianNB)")

    vectores_seguros = col_types.get("numericas", [])

    if not vectores_seguros:
        st.warning(
            "El clasificador Gaussiano requiere vectores numéricos continuos. No hay datos compatibles.")
    else:
        default_features = vectores_seguros[:3] if len(
            vectores_seguros) >= 3 else vectores_seguros
        features = st.multiselect(
            "Selecciona variables predictoras (Vectores Numéricos Continuos)",
            options=vectores_seguros,
            default=default_features
        )

        if st.button("Ejecutar Entrenamiento y Extraer Métricas", type="primary"):
            if not features:
                st.warning(
                    "Hitbox Fallido: Debes seleccionar al menos un vector predictivo.")
            else:
                try:
                    with st.spinner("Entrenando clasificador Gaussiano..."):
                        metrics = engine.train_naive_bayes(features)
                        st.session_state['model_metrics'] = metrics
                except ValueError as e:
                    st.error(f"Error de Integridad (Model Layer): {e}")
                except Exception as e:
                    st.error(f"Excepción No Controlada (System Halt): {e}")

        if 'model_metrics' in st.session_state:
            metrics = st.session_state['model_metrics']

            st.success("Modelo entrenado y métricas retenidas en memoria.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy global", f"{metrics['accuracy']:.4f}")
            m2.metric("Sensibilidad (Recall)", f"{metrics['sensibilidad']}")
            m3.metric("Especificidad", f"{metrics['especificidad']}")

            st.subheader("Matriz de Confusión")
            classes_labels = [str(c) for c in metrics['clases_detectadas']]

            fig_cm = plot_confusion_matrix(
                metrics['matriz_confusion'], classes=classes_labels)
            st.pyplot(fig_cm, clear_figure=True)
