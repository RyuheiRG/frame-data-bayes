import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


@st.cache_data(show_spinner=False)
def summary_statistics(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    desc = df[numeric_cols].describe().T
    desc['missing_count'] = df[numeric_cols].isna().sum().values
    desc['missing_pct'] = desc['missing_count'] / len(df)
    return desc


@st.cache_data(show_spinner=False)
def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    s = df.isna().sum()
    pct = s / len(df)
    return pd.DataFrame({'missing_count': s, 'missing_pct': pct})


@st.cache_data(show_spinner=False)
def correlation_matrix(df: pd.DataFrame, numeric_cols: list, method: str = 'pearson') -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    return df[numeric_cols].corr(method=method)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def target_group_stats(df: pd.DataFrame, numeric_cols: list, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns or not numeric_cols:
        return pd.DataFrame()
    grouped = df.groupby(target_col)[numeric_cols].agg(
        ['mean', 'std', 'count'])
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    return grouped


@st.cache_data(show_spinner=False)
def mutual_info_scores(df: pd.DataFrame, numeric_cols: list, target_col: str) -> pd.Series:
    clean = df.dropna(subset=numeric_cols + [target_col])
    if clean.empty or not numeric_cols:
        return pd.Series(dtype=float)
    X = clean[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = clean[target_col]

    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y.astype(str))

    try:
        mi = mutual_info_classif(
            X, y, discrete_features=False, random_state=42)
        return pd.Series(mi, index=numeric_cols).sort_values(ascending=False)
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(show_spinner=False)
def best_variable_by_mutual_info(df: pd.DataFrame, numeric_cols: list, target_col: str):
    mi = mutual_info_scores(df, numeric_cols, target_col)
    if mi.empty:
        return None, None
    top = mi.idxmax()
    return top, float(mi.loc[top])


@st.cache_data(show_spinner=False)
def event_rarity(df: pd.DataFrame, target_col: str, target_value) -> tuple:
    clean = df.dropna(subset=[target_col])
    if clean.empty:
        return 0.0, "Sin datos suficientes"
    prior = len(clean[clean[target_col] == target_value]) / len(clean)
    if prior <= 0.01:
        label = "Extremadamente raro (<1%)"
    elif prior <= 0.05:
        label = "Raro (<5%)"
    elif prior <= 0.2:
        label = "Poco frecuente (<20%)"
    else:
        label = "Común (>20%)"
    return prior, label


@st.cache_data(show_spinner=False)
def independence_by_correlation(df: pd.DataFrame, numeric_cols: list) -> dict:
    if not numeric_cols or len(numeric_cols) < 2:
        return {"mean_abs_corr": None, "max_abs_corr": None, "conclusion": "N/A (Faltan variables numéricas)"}
    corr = correlation_matrix(df, numeric_cols).abs()
    vals = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().values
    if len(vals) == 0:
        return {"mean_abs_corr": 0.0, "max_abs_corr": 0.0, "conclusion": "Sin correlaciones calculables"}

    mean_abs = float(np.nanmean(vals))
    max_abs = float(np.nanmax(vals))

    if mean_abs < 0.1:
        conclusion = "Variables mayormente independientes (Correlación media muy baja)."
    elif mean_abs < 0.3:
        conclusion = "Dependencia moderada entre variables."
    else:
        conclusion = "Dependencia notable (Alto riesgo de multicolinealidad para Naive Bayes)."

    return {"mean_abs_corr": mean_abs, "max_abs_corr": max_abs, "conclusion": conclusion}


@st.cache_data(show_spinner=False)
def model_reliability_estimate(df: pd.DataFrame, numeric_cols: list, target_col: str) -> dict:
    res = {"baseline_accuracy": None, "nb_cv_accuracy": None, "note": None}
    clean = df.dropna(subset=numeric_cols + [target_col])
    if clean.empty or not numeric_cols:
        res["note"] = "Datos insuficientes para validar."
        return res

    y = clean[target_col]
    try:
        counts = y.value_counts(normalize=True)
        res["baseline_accuracy"] = float(counts.iloc[0])
    except Exception:
        res["note"] = "Fallo al calcular Baseline."

    X = clean[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    if y.dtype == 'object' or y.dtype.name == 'category':
        y_enc = LabelEncoder().fit_transform(y.astype(str))
    else:
        y_enc = y.values

    if len(clean) < 10 or len(np.unique(y_enc)) < 2:
        res["note"] = "Target mono-clase o varianza insuficiente para Cross-Validation."
        return res

    try:
        nb = GaussianNB()
        scores = cross_val_score(
            nb, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
        res["nb_cv_accuracy"] = float(np.mean(scores))
    except Exception as e:
        res["note"] = f"Error CV: {str(e)[:50]}..."

    return res
