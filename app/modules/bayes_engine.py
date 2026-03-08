import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


class BayesianAnalyzer:
    def __init__(self, df: pd.DataFrame, target_col: str, random_state: int = 42):
        if target_col not in df.columns:
            raise ValueError(
                f"Vulnerabilidad detectada: La columna target '{target_col}' no existe en el DataFrame.")

        self.df = df.copy()
        self.target_col = target_col
        self.random_state = random_state
        # var_smoothing ayuda al problema de Laplace en GaussianNB (estabilidad matemática)
        self.model = GaussianNB(var_smoothing=1e-9)

    def calculate_prior(self, target_value) -> float:
        total_records = len(self.df.dropna(subset=[self.target_col]))
        if total_records == 0:
            return 0.0
        target_counts = len(self.df[self.df[self.target_col] == target_value])
        return target_counts / total_records

    def calculate_conditional(self, feature_col: str, feature_condition: callable, target_value) -> float:
        subset_target = self.df[self.df[self.target_col]
                                == target_value].dropna(subset=[feature_col])
        total_target = len(subset_target)
        if total_target == 0:
            return 0.0

        evidence_matches = len(
            subset_target[subset_target[feature_col].apply(feature_condition)])
        return evidence_matches / total_target

    def apply_bayes_theorem(self, prior_a: float, prob_b_given_a: float, prob_b: float) -> float:
        if prob_b <= 0.0:  # Prevención de división por cero y probabilidades negativas
            return 0.0
        return (prob_b_given_a * prior_a) / prob_b

    def _validate_and_prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp = X.copy()
        for col in Xp.columns:
            if not pd.api.types.is_numeric_dtype(Xp[col]):
                try:
                    Xp[col] = pd.to_numeric(Xp[col], errors='coerce')
                except Exception:
                    raise ValueError(
                        f"Feature '{col}' debe ser estrictamente numérica.")

        # Si la coerción generó NaNs, no podemos pasarlos a GaussianNB
        if Xp.isna().any().any():
            raise ValueError(
                "Datos numéricos corruptos (NaNs) detectados después de forzar el tipado.")
        return Xp

    def train_naive_bayes(self, feature_cols: list):
        # Limpieza estricta de variables implicadas
        df_clean = self.df.dropna(subset=feature_cols + [self.target_col])

        if len(df_clean) < 10:
            raise ValueError(
                "El dataset colapsó a <10 filas al purgar valores nulos en las features seleccionadas.")

        # FIX: Evitar el crasheo de train_test_split si el target filtrado solo tiene 1 clase
        if df_clean[self.target_col].nunique() < 2:
            raise ValueError(
                "Tras la limpieza, el target solo tiene una clase. No se puede aplicar clasificación binaria/multiclase (Varianza Cero).")

        X_raw = df_clean[feature_cols]
        y = df_clean[self.target_col]

        X = self._validate_and_prepare_X(X_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        # Cálculo seguro de métricas (soporte para binario real)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensibilidad = round(sensibilidad, 4)
            especificidad = round(especificidad, 4)
        else:
            sensibilidad = "Multiclase (N/A binario)"
            especificidad = "Multiclase (N/A binario)"

        return {
            "accuracy": acc,
            "matriz_confusion": cm,
            "sensibilidad": sensibilidad,
            "especificidad": especificidad,
            "clases_detectadas": self.model.classes_.tolist()
        }
