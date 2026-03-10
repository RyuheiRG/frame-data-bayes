import streamlit as st
import matplotlib.pyplot as plt
from models.bayes_engine import BayesianAnalyzer
from visualization.visualizer import plot_probability_comparison, plot_confusion_matrix


def render_bayes_engine(df, col_types):
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

    # --- STATE MANAGEMENT: Guardamos en memoria global ---
    st.session_state['target_col'] = target_col
    st.session_state['target_value'] = target_value

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
    res1.metric(label="P(B | A) [Sensibilidad Evidencia]",
                value=f"{prob_b_given_a:.4f}")
    res2.metric(label="P(B) [Probabilidad Evidencia]", value=f"{prob_b:.4f}")
    res3.metric(label="P(A | B) [Probabilidad Posterior]", value=f"{posterior:.4f}",
                delta=f"{(posterior - prior_a):.4f} respecto a Prior")

    fig_bayes = plot_probability_comparison(
        prior_a, posterior, str(target_value))
    st.pyplot(fig_bayes)
    plt.close(fig_bayes)

    st.markdown("---")
    st.header("5. Clasificador Naïve Bayes (GaussianNB)")

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
                classes_labels = [str(c) for c in metrics['clases_detectadas']]
                fig_cm = plot_confusion_matrix(
                    metrics['matriz_confusion'], classes=classes_labels)
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            except ValueError as e:
                st.error(f"Error en validación de Model Layer: {e}")
            except Exception as e:
                st.error(f"Falla crítica en el entrenamiento: {e}")
