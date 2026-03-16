import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class BayesianAnalyzer:
    def __init__(self, df: pd.DataFrame, target_col: str, random_state: int = 42):

        if target_col not in df.columns:
            raise ValueError(
                f"Vulnerabilidad Crítica: La columna target '{target_col}' no existe en el payload."
            )

        self.df = df
        self.target_col = target_col
        self.random_state = random_state

        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', GaussianNB(var_smoothing=1e-9))
        ])

    def calculate_prior(self, target_value) -> float:

        target_series = self.df[self.target_col].dropna()
        if target_series.empty:
            return 0.0
        return float((target_series == target_value).mean())

    def calculate_conditional(self, feature_col: str, condition_mask: pd.Series, target_value) -> float:
        target_mask = self.df[self.target_col] == target_value
        subset_target = self.df[target_mask]

        if subset_target.empty:
            return 0.0

        evidence_matches = (target_mask & condition_mask).sum()
        return float(evidence_matches / len(subset_target))

    def _filter_numeric_features(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = X_raw.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError(
                "Data Type Fault: No se detectaron features numéricas válidas para GaussianNB.")

        dropped = set(X_raw.columns) - set(numeric_cols)
        if dropped:
            pass

        return X_raw[numeric_cols]

    def train_naive_bayes(self, feature_cols: list) -> dict:
        df_clean = self.df.dropna(subset=[self.target_col])

        if len(df_clean) < 10:
            raise ValueError(
                "Inanición de Datos: El dataset colapsó a <10 filas al purgar el target nulo.")

        if df_clean[self.target_col].nunique() < 2:
            raise ValueError(
                "Target mono-clase: No se puede clasificar sin varianza. (División lógica por cero).")

        X_raw = df_clean[feature_cols]
        y = df_clean[self.target_col]

        X = self._filter_numeric_features(X_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        metrics = {
            "accuracy": round(acc, 4),
            "matriz_confusion": cm,
            "sensibilidad": "Multiclase (N/A binario)",
            "especificidad": "Multiclase (N/A binario)",
            "clases_detectadas": self.pipeline.named_steps['classifier'].classes_.tolist()
        }

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["sensibilidad"] = round(
                tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
            metrics["especificidad"] = round(
                tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0

        return metrics
