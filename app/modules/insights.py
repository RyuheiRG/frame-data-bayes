import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


def summary_statistics(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    desc = df[numeric_cols].describe().T
    desc['missing_count'] = df[numeric_cols].isna().sum().values
    desc['missing_pct'] = desc['missing_count'] / len(df)
    return desc


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    s = df.isna().sum()
    pct = s / len(df)
    return pd.DataFrame({'missing_count': s, 'missing_pct': pct})


def correlation_matrix(df: pd.DataFrame, numeric_cols: list, method: str = 'pearson') -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    return df[numeric_cols].corr(method=method)


def top_correlated_pairs(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(columns=['feature_a', 'feature_b', 'abs_corr'])
    corr_abs = corr.abs()
    pairs = []
    cols = corr_abs.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr_abs.iloc[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pd.DataFrame(pairs[:n], columns=['feature_a', 'feature_b', 'abs_corr'])


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list, method: str = 'pearson'):
    if not numeric_cols:
        return None
    corr = correlation_matrix(df, numeric_cols, method)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax,
                cbar_kws={'shrink': .6})
    ax.set_title(f"Mapa de Correlaciones ({method})")
    plt.tight_layout()
    return fig


def plot_missingness_bar(df: pd.DataFrame):
    miss = missingness_summary(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    miss_sorted = miss.sort_values('missing_pct', ascending=False).head(30)
    ax.bar(miss_sorted.index.astype(str), miss_sorted['missing_pct'], color='#E53935')
    ax.set_ylabel('Porcentaje faltante')
    ax.set_xticklabels(miss_sorted.index.astype(str), rotation=45, ha='right')
    ax.set_title('Porcentaje de valores faltantes por columna (top 30)')
    plt.tight_layout()
    return fig


def target_group_stats(df: pd.DataFrame, numeric_cols: list, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns or not numeric_cols:
        return pd.DataFrame()
    grouped = df.groupby(target_col)[numeric_cols].agg(['mean', 'std', 'count'])
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    return grouped


def mutual_info_scores(df: pd.DataFrame, numeric_cols: list, target_col: str) -> pd.Series:
    clean = df.dropna(subset=numeric_cols + [target_col])
    if clean.empty or not numeric_cols:
        return pd.Series(dtype=float)
    X = clean[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = clean[target_col]
    try:
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
        return pd.Series(mi, index=numeric_cols).sort_values(ascending=False)
    except Exception:
        return pd.Series(dtype=float)
