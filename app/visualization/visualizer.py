from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap
from typing import Optional, List

from analytics.insights import correlation_matrix, missingness_summary


def plot_histogram(df: pd.DataFrame, column: str) -> Figure:
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Type Fault: La columna '{column}' no es un escalar numérico.")

    fig = Figure(figsize=(8, 4))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    clean_series = df[column].dropna()
    plot_kde = (len(clean_series) < 5000) and (clean_series.nunique() > 1)

    sns.histplot(data=clean_series, kde=plot_kde,
                 ax=ax, color="#4CAF50", bins='auto')

    ax.set_title(f"Distribución de Densidad: {column}")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()

    return fig


def plot_probability_comparison(prior: float, posterior: float, target_name: str) -> Figure:
    fig = Figure(figsize=(6, 4))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    labels = ['Prior Base', 'Posterior (Con Evidencia)']
    values = [prior, posterior]

    bars = ax.bar(labels, values, color=['#757575', '#1976D2'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probabilidad')
    ax.set_title(f'Desplazamiento Bayesiano para {target_name}')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02,
                f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes: Optional[List[str]] = None) -> Figure:
    fig = Figure(figsize=(8, 6))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

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

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')

    for tick in ax.get_yticklabels():
        tick.set_rotation(0)

    fig.tight_layout()
    return fig


def plot_time_series(df: pd.DataFrame, date_col: str, feature_col: str) -> Optional[Figure]:
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return None

    if not pd.api.types.is_numeric_dtype(df[feature_col]):
        raise ValueError(
            f"Type Fault: La variable '{feature_col}' debe ser numérica.")

    fig = Figure(figsize=(10, 4))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    df_clean = df.dropna(subset=[date_col, feature_col])

    MAX_POINTS_UI = 2000
    if len(df_clean) > MAX_POINTS_UI:
        df_clean = df_clean.sort_values(by=date_col)
        df_clean['__time_bin'] = pd.cut(df_clean[date_col], bins=MAX_POINTS_UI)
        df_sorted = df_clean.groupby('__time_bin')[
            feature_col].max().reset_index()
        df_sorted[date_col] = df_sorted['__time_bin'].apply(lambda x: x.mid)
        ax.set_title(
            f"Evolución Máxima: {feature_col} (Agregado a {MAX_POINTS_UI} pts)")
    else:
        df_sorted = df_clean.sort_values(by=date_col)
        ax.set_title(f"Evolución Temporal: {feature_col} vs {date_col}")

    sns.lineplot(data=df_sorted, x=date_col, y=feature_col,
                 ax=ax, color="#FF9800", linewidth=1.5)

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], method: str = 'pearson') -> Optional[Figure]:
    if not numeric_cols:
        return None

    corr = correlation_matrix(df, numeric_cols, method)

    fig = Figure(figsize=(8, 6))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    sns.heatmap(corr, annot=False, cmap='coolwarm',
                ax=ax, cbar_kws={'shrink': .6})
    ax.set_title(f"Mapa de Correlaciones ({method})")

    fig.tight_layout()
    return fig


def plot_missingness_bar(df: pd.DataFrame) -> Figure:
    miss = missingness_summary(df)

    fig = Figure(figsize=(8, 4))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    miss_sorted = miss.sort_values('missing_pct', ascending=False).head(30)

    x_pos = np.arange(len(miss_sorted))
    ax.bar(x_pos, miss_sorted['missing_pct'], color="#D70400")

    ax.set_ylabel('Porcentaje faltante')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(miss_sorted.index.astype(str), rotation=45, ha='right')
    ax.set_title('Valores faltantes por columna (top 30)')

    fig.tight_layout()
    return fig
