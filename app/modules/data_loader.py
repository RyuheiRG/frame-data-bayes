import pandas as pd
import re


class DataLoadError(Exception):
    pass


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zero Trust UX-Friendly: Elimina prefijos de inyección CSV (=, +, -, @)
    pero preserva espacios, tildes y puntuación para no destruir el HUD.
    """

    df.columns = [re.sub(r'^[=+\-@]+', '', str(c)).strip() for c in df.columns]
    return df


def is_likely_datetime(series: pd.Series, threshold=0.7) -> bool:
    """Heurística: Convierte a fecha si el 70% de la columna es parseable."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.notna().mean() >= threshold


def is_binary_column(series: pd.Series) -> bool:
    """Detección estricta de binarios reales."""
    vals = set([str(x).strip().lower() for x in series.dropna().unique()])
    allowed = [{"0", "1"}, {"true", "false"}, {
        "yes", "no"}, {"y", "n"}, {"sí", "no"}]

    if any(vals <= a for a in allowed):
        return True

    return series.dropna().nunique() == 2 and pd.api.types.is_numeric_dtype(series)


def load_and_validate_data(filepath, max_rows=200_000, encoding=("utf-8", "latin-1")) -> pd.DataFrame:
    """Ingesta Zero Trust con coerción de estado global (Fix de Desincronización)."""
    last_exc = None
    df = None

    for enc in encoding:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except Exception as e:
            last_exc = e
            continue

    if df is None:
        raise DataLoadError(
            f"Falla crítica de lectura CSV (Encoding/IO): {last_exc}")

    if df.empty:
        raise DataLoadError("Vector de falla: El archivo CSV está vacío.")

    if len(df) > max_rows:
        raise DataLoadError(
            f"Ataque DoS mitigado: Dataset supera el límite de {max_rows} filas.")

    df = sanitize_column_names(df)

    for col in df.columns:
        if df[col].dtype == 'object' and is_likely_datetime(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def detect_column_types(df: pd.DataFrame) -> dict:
    """Escaneo de superficie. Como el DF ya está curado, solo lee los tipos."""
    col_types = {
        "numericas": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categoricas": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "fechas": df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist(),
        "binarias": [col for col in df.columns if is_binary_column(df[col])]
    }

    return col_types
