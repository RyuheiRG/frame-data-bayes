import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap


def plot_histogram(df: pd.DataFrame, column: str):
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Vulnerabilidad: La columna {column} no es numérica.")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=column, kde=True, ax=ax, color="#4CAF50")
    ax.set_title(f"Distribución de Densidad: {column}")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    return fig


def plot_probability_comparison(prior: float, posterior: float, target_name: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f'P({target_name}) Base', f'P({target_name} | Evidencia)']
    values = [prior, posterior]

    bars = ax.bar(labels, values, color=['#757575', '#1976D2'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probabilidad')
    ax.set_title('Desplazamiento Bayesiano (Prior vs Posterior)')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02,
                f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes: list = None):
    fig, ax = plt.subplots(figsize=(8, 6))

    if classes is None:
        classes = ["Negativo (0)", "Positivo (1)"]
    else:
        classes = [textwrap.fill(str(c), width=20) for c in classes]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                cbar_kws={'shrink': .8})

    ax.set_ylabel('Valor Real (Ground Truth)', fontweight='bold')
    ax.set_xlabel('Predicción del Modelo', fontweight='bold')
    ax.set_title('Matriz de Confusión (Naïve Bayes)', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_time_series(df: pd.DataFrame, date_col: str, feature_col: str):
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return None
    if not pd.api.types.is_numeric_dtype(df[feature_col]):
        raise ValueError(
            f"La variable a graficar en el tiempo ({feature_col}) debe ser numérica.")

    fig, ax = plt.subplots(figsize=(10, 4))
    df_sorted = df.sort_values(by=date_col)

    sns.lineplot(data=df_sorted, x=date_col,
                 y=feature_col, ax=ax, color="#FF9800")
    ax.set_title(f"Evolución Temporal: {feature_col} a lo largo de {date_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
