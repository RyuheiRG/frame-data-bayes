import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap


def plot_histogram(df: pd.DataFrame, column: str):
    """Renderiza el histograma de una variable numérica con curva de densidad."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=column, kde=True, ax=ax, color="#4CAF50")
    ax.set_title(f"Distribución de Densidad: {column}")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    return fig


def plot_probability_comparison(prior: float, posterior: float, target_name: str):
    """
    Compara P(A) vs P(A|B) en un diagrama de barras determinístico.
    Mapea el impacto de la evidencia en el axioma de probabilidad.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f'P({target_name}) Base', f'P({target_name} | Evidencia)']
    values = [prior, posterior]

    bars = ax.bar(labels, values, color=['#757575', '#1976D2'])
    # Rango estricto [0, 1] de probabilidad + margen visual
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probabilidad')
    ax.set_title('Desplazamiento Bayesiano (Prior vs Posterior)')

    # Inyección de valores exactos sobre las barras para reducir ambigüedad
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02,
                f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes: list = None):
    """Genera un heatmap estricto mitigando el overflow de texto en los labels."""
    # Ampliamos la superficie de renderizado
    fig, ax = plt.subplots(figsize=(8, 6))

    if classes is None:
        classes = ["Negativo (0)", "Positivo (1)"]
    else:
        # Envolvemos el texto largo para que salte de línea dinámicamente cada 20 caracteres
        classes = [textwrap.fill(str(c), width=20) for c in classes]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                # Escalamos la barra de color para balancear la UI
                cbar_kws={'shrink': .8})

    ax.set_ylabel('Valor Real (Ground Truth)', fontweight='bold')
    ax.set_xlabel('Predicción del Modelo', fontweight='bold')
    ax.set_title('Matriz de Confusión (Naïve Bayes)', pad=20)

    # Rotación táctica a 45 grados y alineación a la derecha para evitar colisiones
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    return fig


def plot_time_series(df: pd.DataFrame, date_col: str, feature_col: str):
    """
    Traza la evolución temporal. 
    Aplica Zero Trust verificando que la columna sea verdaderamente temporal.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        # Fallback silencioso si la serie no es de tiempo
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    # Ordenar por fecha para evitar renderizados de líneas caóticas (spaghetti plot)
    df_sorted = df.sort_values(by=date_col)

    sns.lineplot(data=df_sorted, x=date_col,
                 y=feature_col, ax=ax, color="#FF9800")
    ax.set_title(f"Evolución Temporal: {feature_col} a lo largo de {date_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
