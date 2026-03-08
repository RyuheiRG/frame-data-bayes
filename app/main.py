import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.data_loader import load_and_validate_data, detect_column_types, DataLoadError
from modules.bayes_engine import BayesianAnalyzer
from modules.visualizer import plot_histogram, plot_probability_comparison, plot_confusion_matrix, plot_time_series

st.set_page_config(page_title="Bayesian Engine | Zero Trust", layout="wide")

# --- OPTIMIZACIÓN: CACHING PARA EVITAR RE-CÁLCULOS ---


@st.cache_data(ttl=600)
def cached_detect_column_types(df):
    return detect_column_types(df)


st.title("🛡️ Motor de Inferencia Bayesiana")
st.markdown("---")

with st.sidebar:
    st.header("1. Ingesta de Datos")
    uploaded_file = st.file_uploader("Cargar dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = load_and_validate_data(uploaded_file)
        st.success("CSV montado en memoria exitosamente.")

        st.header("2. Escaneo de Superficie")

        col_types = cached_detect_column_types(df)

        with st.expander("Ver mapa de tipos de datos detectados", expanded=False):
            st.json(col_types)

        st.subheader("Análisis Exploratorio (EDA)")
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
        possible_targets = col_types["binarias"] + col_types["categoricas"]

        if not possible_targets:
            st.error(
                "Vector de falla: No se detectaron columnas válidas para usar como objetivo.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                target_col = st.selectbox(
                    "Selecciona la Variable Objetivo (A)", possible_targets)
            with col2:
                classes_sorted = sorted(
                    df[target_col].dropna().unique(), key=lambda x: str(x))
                target_value = st.selectbox(
                    "Selecciona la clase positiva", classes_sorted)

            engine = BayesianAnalyzer(df, target_col)

            st.markdown("---")
            st.header("4. Motor Probabilístico")
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

            posterior = engine.apply_bayes_theorem(
                prior_a, prob_b_given_a, prob_b)
            st.latex(r"P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}")

            res1, res2, res3 = st.columns(3)
            res1.metric(
                label="P(B | A) [Sensibilidad de Evidencia]", value=f"{prob_b_given_a:.4f}")
            res2.metric(
                label="P(B) [Probabilidad de Evidencia]", value=f"{prob_b:.4f}")
            res3.metric(
                label="P(A | B) [Probabilidad Posterior]", value=f"{posterior:.4f}")

            fig_bayes = plot_probability_comparison(
                prior_a, posterior, target_value)
            st.pyplot(fig_bayes)
            plt.close(fig_bayes)

            st.markdown("---")
            st.header("5. Clasificador Naïve Bayes (GaussianNB)")

            features = st.multiselect("Selecciona variables predictoras (Vectores Numéricos)", col_types["numericas"], default=col_types["numericas"][:3] if len(
                col_types["numericas"]) >= 3 else col_types["numericas"])

            if st.button("Ejecutar Entrenamiento", type="primary"):
                if not features:
                    st.warning(
                        "Debes seleccionar al menos un vector predictivo.")
                else:
                    try:
                        metrics = engine.train_naive_bayes(features)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Accuracy global",
                                  f"{metrics['accuracy']:.4f}")
                        m2.metric("Sensibilidad (Recall)",
                                  f"{metrics['sensibilidad']}")
                        m3.metric("Especificidad",
                                  f"{metrics['especificidad']}")

                        st.subheader("Matriz de Confusión")
                        classes_labels = [str(c)
                                          for c in metrics['clases_detectadas']]
                        fig_cm = plot_confusion_matrix(
                            metrics['matriz_confusion'], classes=classes_labels)
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)
                        st.session_state['last_metrics'] = metrics

                    except ValueError as e:
                        st.error(
                            f"Error de validación en los datos (Model Layer): {e}")
                    except Exception as e:
                        st.error(f"Falla crítica en el entrenamiento: {e}")

    except DataLoadError as e:
        st.error(str(e))

else:
    st.info(
        "Arquitectura en Standby. Esperando carga de payload (CSV) en el panel lateral.")
