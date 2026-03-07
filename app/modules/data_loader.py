import pandas as pd
import re

# Excepción personalizada (Desacoplamiento de UI)


class DataLoadError(Exception):
    pass


# Regex para detectar intentos de inyección de macros CSV
_INJECTION_PREFIX = re.compile(r'^[=+\-@]')


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Evita colisiones y bugs de indexación limpiando caracteres especiales."""
    df.columns = [
        re.sub(r'[^A-Za-z0-9_]', '_', str(c)).strip().lower()
        for c in df.columns
    ]
    return df


def is_likely_datetime(series: pd.Series, threshold=0.7) -> bool:
    """Heurística: Solo convierte a fecha si al menos el 70% de la columna es parseable."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.notna().mean() >= threshold


def is_binary_column(series: pd.Series) -> bool:
    """Detección estricta de binarios reales, evitando falsos positivos (ej. Género)."""
    vals = set([str(x).strip().lower() for x in series.dropna().unique()])
    allowed = [{"0", "1"}, {"true", "false"}, {
        "yes", "no"}, {"y", "n"}, {"sí", "no"}]

    if any(vals <= a for a in allowed):
        return True

    # Fallback numérico estricto
    return series.dropna().nunique() == 2 and pd.api.types.is_numeric_dtype(series)


def load_and_validate_data(filepath, max_rows=200_000, encoding=("utf-8", "latin-1")) -> pd.DataFrame:
    """
    Ingesta Zero Trust. No asume encoding correcto y prohíbe datasets masivos.
    Ya NO depende de Streamlit.
    """
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

    # Sanitización de nombres de columnas
    df = sanitize_column_names(df)
    return df


def detect_column_types(df: pd.DataFrame) -> dict:
    """Escaneo de superficie con heurísticas avanzadas."""
    df_scan = df.copy()

    for col in df_scan.columns:
        if df_scan[col].dtype == 'object' and is_likely_datetime(df_scan[col]):
            df_scan[col] = pd.to_datetime(df_scan[col], errors="coerce")

    col_types = {
        "numericas": df_scan.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categoricas": df_scan.select_dtypes(include=['object', 'category']).columns.tolist(),
        "fechas": df_scan.select_dtypes(include=['datetime64']).columns.tolist(),
        "binarias": [col for col in df_scan.columns if is_binary_column(df_scan[col])]
    }

    return col_types
