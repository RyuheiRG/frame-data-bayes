import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data.data_loader import load_and_validate_data, detect_column_types, DataLoadError
from models.bayes_engine import BayesianAnalyzer
from visualization.visualizer import plot_histogram, plot_probability_comparison, plot_confusion_matrix, plot_time_series
from analytics.insights import (
    plot_correlation_heatmap, plot_missingness_bar,
    best_variable_by_mutual_info, event_rarity,
    independence_by_correlation, model_reliability_estimate
)

st.set_page_config(page_title="Bayesian Engine | Zero Trust", layout="wide")


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
        st.rerun()

if uploaded_file is not None:
    try:
        df = cached_load_data(uploaded_file)
        st.sidebar.success(
            f"CSV montado en memoria: {df.shape[0]} filas, {df.shape[1]} dims.")

        st.header("2. Escaneo de Superficie y EDA")
        col_types = cached_detect_column_types(df)

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

        st.header("3. Configuración del Objetivo (Target)")
        possible_targets = list(
            set(col_types["binarias"] + col_types["categoricas"]))

        if not possible_targets:
            st.error(
                "Vector de falla: No se detectaron columnas válidas para usar como objetivo.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox(
                "Selecciona la Variable Objetivo (A)", possible_targets)
        with col2:
            classes_sorted = sorted(
                df[target_col].dropna().unique(), key=lambda x: str(x))
            target_value = st.selectbox(
                "Selecciona la clase positiva (Evento Anómalo)", classes_sorted)

        engine = BayesianAnalyzer(df, target_col)

        st.markdown("---")
        st.header("4. Motor Probabilístico (Teorema de Bayes)")
        prior_a = engine.calculate_prior(target_value)
        st.metric(
            label=f"P(A) -> Probabilidad Base de '{target_value}'", value=f"{prior_a:.4f}")

        st.subheader("Evaluación de Evidencia Condicional")
        col_ev1, col_ev2 = st.columns(2)
        with col_ev1:
            evidence_col = st.selectbox(
                "Selecciona Variable de Evidencia (B)", col_types["numericas"])
        with col_ev2:
            umbral = st.number_input(
                f"Umbral: {evidence_col} > X", value=float(df[evidence_col].mean()))

        def condition(x): return x > umbral

        prob_b_given_a = engine.calculate_conditional(
            evidence_col, condition, target_value)
        total_b_subset = df[df[evidence_col] > umbral]
        prob_b = len(total_b_subset) / len(df) if len(df) > 0 else 0

        posterior = engine.apply_bayes_theorem(prior_a, prob_b_given_a, prob_b)
        st.latex(r"P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}")

        res1, res2, res3 = st.columns(3)
        res1.metric(
            label="P(B | A) [Sensibilidad de Evidencia]", value=f"{prob_b_given_a:.4f}")
        res2.metric(
            label="P(B) [Probabilidad de Evidencia]", value=f"{prob_b:.4f}")
        res3.metric(label="P(A | B) [Probabilidad Posterior]", value=f"{posterior:.4f}",
                    delta=f"{(posterior - prior_a):.4f} respecto a Prior")

        fig_bayes = plot_probability_comparison(
            prior_a, posterior, str(target_value))
        st.pyplot(fig_bayes)
        plt.close(fig_bayes)

        st.markdown("---")
        st.header("5. Clasificador Naïve Bayes (GaussianNB)")
        st.write(
            "Generación de un modelo inferencial multivariable asumiendo independencia entre features.")

        features = st.multiselect("Selecciona variables predictoras (Vectores Numéricos)",
                                  col_types["numericas"],
                                  default=col_types["numericas"][:3] if len(col_types["numericas"]) >= 3 else col_types["numericas"])

        if st.button("Ejecutar Entrenamiento y Extraer Métricas", type="primary"):
            if not features:
                st.warning("Debes seleccionar al menos un vector predictivo.")
            else:
                try:
                    metrics = engine.train_naive_bayes(features)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Accuracy global", f"{metrics['accuracy']:.4f}")
                    m2.metric("Sensibilidad (Recall)",
                              f"{metrics['sensibilidad']}")
                    m3.metric("Especificidad", f"{metrics['especificidad']}")

                    st.subheader("Matriz de Confusión")
                    classes_labels = [str(c)
                                      for c in metrics['clases_detectadas']]
                    fig_cm = plot_confusion_matrix(
                        metrics['matriz_confusion'], classes=classes_labels)
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)

                except ValueError as e:
                    st.error(f"Error en validación de Model Layer: {e}")
                except Exception as e:
                    st.error(f"Falla crítica en el entrenamiento: {e}")

        st.markdown("---")
        st.header("6. Insights Estadísticos de Negocio")

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

            ind = independence_by_correlation(
                df, col_types.get("numericas", []))
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
                fig_miss = plot_missingness_bar(df)
                st.pyplot(fig_miss)
                plt.close(fig_miss)

    except DataLoadError as e:
        st.error(str(e))
else:
    st.info("Arquitectura en Standby. Esperando inyección de payload (CSV) en el panel lateral.")
