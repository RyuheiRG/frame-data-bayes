import pandas as pd
import re


class DataLoadError(Exception):
    pass


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zero Trust UX-Friendly: Sanitiza nombres de columnas sin destruir el array.
    """
    safe_cols = []
    for c in df.columns:
        # Limpia caracteres de inyección CSV clásicos al inicio
        cleaned = re.sub(r'^[=+\-@\t\r\n]+', '', str(c)).strip()
        # Si la columna queda vacía por ser puro símbolo, asignamos nombre genérico
        safe_cols.append(cleaned if cleaned else "col_anonima")

    df.columns = safe_cols
    return df


def is_likely_datetime(series: pd.Series, threshold=0.7) -> bool:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.notna().mean() >= threshold


def is_binary_column(series: pd.Series) -> bool:
    """Detección estricta de binarios reales. Corrige el bug de subsets constantes."""
    clean_series = series.dropna()
    vals = set([str(x).strip().lower() for x in clean_series.unique()])

    # Regla de Oro: Un vector binario TIENE que tener exactamente 2 estados únicos
    if len(vals) != 2:
        return False

    allowed_sets = [{"0", "1"}, {"true", "false"},
                    {"yes", "no"}, {"y", "n"}, {"sí", "no"}]

    if any(vals.issubset(a) for a in allowed_sets):
        return True

    return pd.api.types.is_numeric_dtype(series)


def load_and_validate_data(filepath, max_rows=200_000, encoding=("utf-8", "latin-1")) -> pd.DataFrame:
    """
    Ingesta Zero Trust. Corta de raíz intentos de DoS limitando la lectura en I/O,
    no en memoria.
    """
    last_exc = None
    df = None

    for enc in encoding:
        try:
            # OPTIMIZACIÓN: Leemos max_rows + 1. Si llega al +1, sabemos que excedió el límite.
            df = pd.read_csv(filepath, encoding=enc, nrows=max_rows + 1)
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
            f"Ataque DoS / OutOfMemory mitigado: El dataset supera el límite de {max_rows} filas.")

    df = sanitize_column_names(df)

    for col in df.columns:
        if df[col].dtype == 'object' and is_likely_datetime(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def detect_column_types(df: pd.DataFrame) -> dict:
    return {
        "numericas": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categoricas": df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist(),
        "fechas": df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist(),
        "binarias": [col for col in df.columns if is_binary_column(df[col])]
    }
