import pandas as pd
import streamlit as st


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Carga un CSV y valida su integridad básica.
    Retorna el DataFrame o lanza una excepción detallada.
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("El archivo CSV está vacío.")
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo en la ruta {filepath}")
        return None
    except Exception as e:
        st.error(f"Error crítico al leer el archivo: {str(e)}")
        return None


def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Escanea el DataFrame y clasifica las columnas por tipo de dato.
    """
    # Intentar forzar la conversión de fechas primero para una mejor detección
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass

    col_types = {
        "numericas": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categoricas": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "fechas": df.select_dtypes(include=['datetime64']).columns.tolist(),
        "binarias": [col for col in df.columns if df[col].nunique() == 2]
    }

    return col_types
