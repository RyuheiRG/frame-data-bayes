import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


class BayesianAnalyzer:
    def __init__(self, df: pd.DataFrame, target_col: str):
        """
        Inicializa el motor. 
        Zero Trust: Verifica que la columna objetivo exista antes de instanciar.
        """
        if target_col not in df.columns:
            raise ValueError(
                f"Vulnerabilidad detectada: La columna '{target_col}' no existe en el dataset.")

        self.df = df.copy()  # Trabajamos con una copia aislada
        self.target_col = target_col
        self.model = GaussianNB()

    def calculate_prior(self, target_value) -> float:
        """
        Calcula P(A) -> Ejemplo: P(Fallo)
        """
        total_records = len(self.df)
        if total_records == 0:
            return 0.0

        target_counts = len(self.df[self.df[self.target_col] == target_value])
        return target_counts / total_records

    def calculate_conditional(self, feature_col: str, feature_condition: callable, target_value) -> float:
        """
        Calcula P(B|A) -> P(Evidencia | Fallo)
        Acepta un callable (función lambda) para evaluar umbrales lógicos dinámicos.
        Ejemplo de condition: lambda x: x > 80 (Temperatura > umbral)
        """
        # Filtrar el universo donde A es verdadero (Ej: Donde SÍ hay fallo)
        subset_target = self.df[self.df[self.target_col] == target_value]
        total_target = len(subset_target)

        if total_target == 0:
            return 0.0

        # Contar cuántos de ese subconjunto cumplen la condición de la evidencia
        evidence_matches = len(
            subset_target[subset_target[feature_col].apply(feature_condition)])

        return evidence_matches / total_target

    def apply_bayes_theorem(self, prior_a: float, prob_b_given_a: float, prob_b: float) -> float:
        """
        Aplica la fórmula estricta: P(A|B) = (P(B|A) * P(A)) / P(B)
        """
        if prob_b == 0:
            return 0.0  # Mitigación contra ZeroDivisionError
        return (prob_b_given_a * prior_a) / prob_b

    def train_naive_bayes(self, feature_cols: list):
        """
        Entrena el clasificador y retorna las métricas exactas solicitadas.
        Trade-off: GaussianNB asume distribución normal continua. 
        Las variables categóricas deberían estar codificadas previamente.
        """
        # Limpieza estricta: Solo tomamos filas sin NaNs en las features seleccionadas
        df_clean = self.df.dropna(subset=feature_cols + [self.target_col])
        X = df_clean[feature_cols]
        y = df_clean[self.target_col]

        # Validar si hay suficientes clases
        if len(y.unique()) < 2:
            raise ValueError(
                "El dataset filtrado no tiene suficientes clases (mínimo 2) en la variable objetivo para entrenar.")

        # Split determinístico (random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Extracción de métricas
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        # Cálculo de Sensibilidad y Especificidad (Asumiendo matriz 2x2 para simplificar)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensibilidad = "Multiclase (Requiere promedio)"
            especificidad = "Multiclase (Requiere promedio)"

        return {
            "accuracy": acc,
            "matriz_confusion": cm,
            "sensibilidad": sensibilidad,
            "especificidad": especificidad,
            "y_test": y_test,
            "y_pred": y_pred
        }
