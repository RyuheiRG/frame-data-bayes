import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer


class BayesianAnalyzer:
    def __init__(self, df: pd.DataFrame, target_col: str, random_state: int = 42):
        """
        Inicializa el motor probabilístico bajo arquitectura Zero Trust.
        No muta ni duplica el dataframe original.
        """
        if target_col not in df.columns:
            raise ValueError(
                f"Vulnerabilidad Crítica: La columna target '{target_col}' no existe en el payload.")

        self.df = df
        self.target_col = target_col
        self.random_state = random_state

        self.model = GaussianNB(var_smoothing=1e-9)

    def calculate_prior(self, target_value) -> float:
        """
        Cálculo vectorizado de P(A). Complejidad O(1) a nivel de memoria.
        """
        target_series = self.df[self.target_col].dropna()
        if target_series.empty:
            return 0.0

        return float((target_series == target_value).mean())

    def calculate_conditional(self, feature_col: str, condition_mask: pd.Series, target_value) -> float:
        """
        Cálculo de P(B|A).
        REFACTOR: Recibe una máscara booleana (Hitbox superpuesto) en lugar de un callable.
        Se ejecuta a nivel de C, evadiendo el cuello de botella del GIL de Python.
        """
        target_mask = self.df[self.target_col] == target_value
        subset_target = self.df[target_mask]

        if subset_target.empty:
            return 0.0

        evidence_matches = (target_mask & condition_mask).sum()
        return float(evidence_matches / len(subset_target))

    def _validate_and_prepare_X(self, X_raw: pd.DataFrame) -> np.ndarray:
        """
        Saneamiento del tensor de entrenamiento. Reemplaza el peligroso dropna/fillna(0)
        con imputación estratégica para preservar la campana de Gauss.
        """
        X_numeric = X_raw.apply(pd.to_numeric, errors='coerce')

        imputer = SimpleImputer(strategy='median')
        X_clean = imputer.fit_transform(X_numeric)

        return X_clean

    def train_naive_bayes(self, feature_cols: list) -> dict:
        """
        Pipeline seguro de entrenamiento y extracción de métricas.
        Genera el clasificador P(A|x1, x2, ..., xn).
        """
        df_clean = self.df.dropna(subset=[self.target_col])

        if len(df_clean) < 10:
            raise ValueError(
                "Inanición de Datos: El dataset colapsó a <10 filas al purgar el target nulo.")

        if df_clean[self.target_col].nunique() < 2:
            raise ValueError(
                "Target mono-clase: No se puede clasificar sin varianza. (División lógica por cero).")

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
