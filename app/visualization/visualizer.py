import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap
import seaborn as sns
from analytics.insights import correlation_matrix, missingness_summary


def plot_histogram(df: pd.DataFrame, column: str):
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Vulnerabilidad: La columna {column} no es numérica.")

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_kde = len(df[column].dropna()) < 5000

    sns.histplot(data=df, x=column, kde=plot_kde,
                 ax=ax, color="#4CAF50", bins='auto')
    ax.set_title(f"Distribución de Densidad: {column}")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    return fig


def plot_probability_comparison(prior: float, posterior: float, target_name: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f'Prior Base', f'Posterior (Con Evidencia)']
    values = [prior, posterior]

    bars = ax.bar(labels, values, color=['#757575', '#1976D2'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probabilidad')
    ax.set_title(f'Desplazamiento Bayesiano para {target_name}')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02,
                f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes: list = None):
    fig, ax = plt.subplots(figsize=(8, 6))

    if classes is None:
        classes = ["Negativo", "Positivo"]
    else:
        classes = [textwrap.fill(str(c), width=15) for c in classes]

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
            f"La variable a graficar ({feature_col}) debe ser numérica.")

    fig, ax = plt.subplots(figsize=(10, 4))
    df_sorted = df.dropna(
        subset=[date_col, feature_col]).sort_values(by=date_col)

    MAX_POINTS_UI = 2000
    if len(df_sorted) > MAX_POINTS_UI:
        step = len(df_sorted) // MAX_POINTS_UI
        df_sorted = df_sorted.iloc[::step]
        ax.set_title(
            f"Evolución: {feature_col} (Submuestreo estricto a {MAX_POINTS_UI} pts)")
    else:
        ax.set_title(f"Evolución Temporal: {feature_col} vs {date_col}")

    sns.lineplot(data=df_sorted, x=date_col, y=feature_col,
                 ax=ax, color="#FF9800", linewidth=1.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list, method: str = 'pearson'):
    if not numeric_cols:
        return None

    corr = correlation_matrix(df, numeric_cols, method)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap='coolwarm',
                ax=ax, cbar_kws={'shrink': .6})
    ax.set_title(f"Mapa de Correlaciones ({method})")
    plt.tight_layout()
    return fig


def plot_missingness_bar(df: pd.DataFrame):
    miss = missingness_summary(df)

    fig, ax = plt.subplots(figsize=(8, 4))
    miss_sorted = miss.sort_values('missing_pct', ascending=False).head(30)
    ax.bar(miss_sorted.index.astype(str),
           miss_sorted['missing_pct'], color='#E53935')
    ax.set_ylabel('Porcentaje faltante')
    ax.set_xticklabels(miss_sorted.index.astype(str), rotation=45, ha='right')
    ax.set_title('Valores faltantes por columna (top 30)')
    plt.tight_layout()
    return fig
