"""
YADEM Logistic Regression Model (Model A)
=============================================================================
The most interpretable of the three ensemble models. Calculates the
probability of default by fitting a linear relationship between the
features and the outcome. Highly auditable and serves as the baseline.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from typing import Dict, Any, Optional
from loguru import logger
import joblib
import os

from src.config.model_config import LogisticRegressionConfig


class YademLogisticRegression:
    """
    Model A: Logistic Regression
    
    Purpose:
    - Produces a probability score easy to explain to regulators
    - Serves as interpretable baseline for benchmarking complex models
    - Works best with well-structured numerical data (bank statements, bureau scores)
    """

    def __init__(self, config: Optional[LogisticRegressionConfig] = None):
        self.config = config or LogisticRegressionConfig()
        self.model = LogisticRegression(**self.config.to_dict())
        self.is_fitted = False
        self.feature_names = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train the logistic regression model.
        
        Returns:
            Dictionary with training metrics.
        """
        logger.info("Training Logistic Regression (Model A)...")
        self.feature_names = list(X_train.columns)

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Training metrics
        train_proba = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)

        metrics = {
            "model": "logistic_regression",
            "train_auc_roc": round(train_auc, 4),
            "n_features": len(self.feature_names),
            "n_train_samples": len(X_train),
        }

        if X_val is not None and y_val is not None:
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            metrics["val_auc_roc"] = round(val_auc, 4)
            logger.info(f"  Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
        else:
            logger.info(f"  Train AUC: {train_auc:.4f}")

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of default (class 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_coefficients(self) -> pd.DataFrame:
        """Return feature coefficients for interpretability."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_[0],
            "abs_coefficient": np.abs(self.model.coef_[0]),
        }).sort_values("abs_coefficient", ascending=False)

    def save(self, path: str) -> str:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, path)
        logger.info(f"Logistic Regression saved to {path}")
        return path

    def load(self, path: str) -> None:
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.is_fitted = True
        logger.info(f"Logistic Regression loaded from {path}")
