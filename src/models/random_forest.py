"""
YADEM Random Forest Model (Model B)
=============================================================================
Captures non-linear relationships by building hundreds of decision trees on
random subsets of data and features. Robust against noise and missing values,
which is critical in the African SME context where data completeness varies.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, Optional
from loguru import logger
import joblib
import os

from src.config.model_config import RandomForestConfig


class YademRandomForest:
    """
    Model B: Random Forest
    
    Purpose:
    - Handles mixed data types (numerical + categorical) exceptionally well
    - Robust to noise and missing values prevalent in African SME data
    - Captures non-linear relationships that Logistic Regression cannot
    - Provides built-in feature importance rankings
    """

    def __init__(self, config: Optional[RandomForestConfig] = None):
        self.config = config or RandomForestConfig()
        self.model = RandomForestClassifier(**self.config.to_dict())
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
        Train the random forest model.
        
        Returns:
            Dictionary with training metrics.
        """
        logger.info("Training Random Forest (Model B)...")
        self.feature_names = list(X_train.columns)

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        train_proba = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)

        metrics = {
            "model": "random_forest",
            "train_auc_roc": round(train_auc, 4),
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
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
        """Return probability of default."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances ranked by importance."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

    def save(self, path: str) -> str:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, path)
        logger.info(f"Random Forest saved to {path}")
        return path

    def load(self, path: str) -> None:
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.is_fitted = True
        logger.info(f"Random Forest loaded from {path}")
