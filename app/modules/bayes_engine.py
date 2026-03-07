import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


class BayesianAnalyzer:
    def __init__(self, df: pd.DataFrame, target_col: str, random_state: int = 42):
        if target_col not in df.columns:
            raise ValueError(
                f"Vulnerabilidad detectada: La columna '{target_col}' no existe.")

        self.df = df.copy()
        self.target_col = target_col
        self.random_state = random_state
        self.model = GaussianNB()

    def calculate_prior(self, target_value) -> float:
        total_records = len(self.df)
        if total_records == 0:
            return 0.0
        target_counts = len(self.df[self.df[self.target_col] == target_value])
        return target_counts / total_records

    def calculate_conditional(self, feature_col: str, feature_condition: callable, target_value) -> float:
        subset_target = self.df[self.df[self.target_col] == target_value]
        total_target = len(subset_target)
        if total_target == 0:
            return 0.0

        evidence_matches = len(
            subset_target[subset_target[feature_col].apply(feature_condition)])
        return evidence_matches / total_target

    def apply_bayes_theorem(self, prior_a: float, prob_b_given_a: float, prob_b: float) -> float:
        if prob_b == 0:
            return 0.0
        return (prob_b_given_a * prior_a) / prob_b

    def _validate_and_prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sanitización estricta: GaussianNB colapsa con strings."""
        Xp = X.copy()
        for col in Xp.columns:
            if not pd.api.types.is_numeric_dtype(Xp[col]):
                try:
                    Xp[col] = pd.to_numeric(Xp[col], errors='raise')
                except Exception:
                    raise ValueError(
                        f"Vector de falla: Feature '{col}' debe ser numérica para GaussianNB.")
        return Xp

    def train_naive_bayes(self, feature_cols: list):
        # 1. Limpieza
        df_clean = self.df.dropna(subset=feature_cols + [self.target_col])
        if len(df_clean) < 10:
            raise ValueError(
                "Dataset resultante post-limpieza es demasiado pequeño (<10 filas).")

        X_raw = df_clean[feature_cols]
        y = df_clean[self.target_col]

        # 2. Validación de tipos (Evita crashes de Scikit-Learn)
        X = self._validate_and_prepare_X(X_raw)

        # 3. Split con estratificación para mantener balance de clases
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Exposición de probabilidades (Útil para calibración futura o IA)
        y_prob = self.model.predict_proba(X_test)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensibilidad = "Multiclase (Requiere métrica macro)"
            especificidad = "Multiclase (Requiere métrica macro)"

        return {
            "accuracy": acc,
            "matriz_confusion": cm,
            "sensibilidad": sensibilidad,
            "especificidad": especificidad,
            # Guardamos una muestra para el JSON de la IA
            "y_prob_sample": y_prob[:5].tolist(),
            "clases_detectadas": self.model.classes_.tolist()
        }
