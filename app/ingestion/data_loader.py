import pandas as pd
import re


class DataLoadError(Exception):
    pass


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Zero Trust UX-Friendly: Sanitiza nombres de columnas."""
    safe_cols = [re.sub(r'^[=+\-@\t\r\n]+', '', str(c)).strip()
                 or "col_anonima" for c in df.columns]
    df.columns = safe_cols
    return df


def is_binary_column(series: pd.Series) -> bool:
    """Detección con fast-fail (early exit) para ahorrar CPU."""
    clean_series = series.dropna()

    if clean_series.nunique() != 2:
        return False

    if pd.api.types.is_numeric_dtype(series):
        return True

    vals = set(clean_series.astype(str).str.strip().str.lower().unique())
    allowed_sets = [{"0", "1"}, {"true", "false"},
                    {"yes", "no"}, {"y", "n"}, {"sí", "no"}]

    return any(vals.issubset(a) for a in allowed_sets)


def load_and_validate_data(filepath_or_buffer, max_rows=200_000, encoding=("utf-8", "latin-1")) -> pd.DataFrame:
    df = None
    last_exc = None

    for enc in encoding:
        try:
            df = pd.read_csv(filepath_or_buffer,
                             encoding=enc, nrows=max_rows + 1)
            break
        except Exception as e:
            last_exc = e
            if hasattr(filepath_or_buffer, 'seek'):
                filepath_or_buffer.seek(0)
            continue

    if df is None:
        raise DataLoadError(
            f"Falla crítica de lectura CSV (Encoding/IO): {last_exc}")
    if df.empty:
        raise DataLoadError("Vector de falla: El archivo CSV está vacío.")
    if len(df) > max_rows:
        raise DataLoadError(
            f"Ataque DoS mitigado: dataset supera el límite de {max_rows} filas.")

    df = sanitize_column_names(df)

    for col in df.columns:
        if df[col].dtype == 'object':
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                df[col] = parsed

    return df


def detect_column_types(df: pd.DataFrame) -> dict:
    """Clasificación disjunta estricta. Una columna no puede existir en dos estados simultáneos."""
    tipos = {"numericas": [], "categoricas": [], "fechas": [], "binarias": []}

    for col in df.columns:
        series = df[col]

        if is_binary_column(series):
            tipos["binarias"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            tipos["fechas"].append(col)
        elif pd.api.types.is_numeric_dtype(series):
            tipos["numericas"].append(col)
        else:
            tipos["categoricas"].append(col)

    return tipos
