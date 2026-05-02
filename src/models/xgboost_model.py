"""
YADEM XGBoost Model (Model C)
=============================================================================
The most powerful of the three ensemble models. Uses sequential gradient
boosting where each tree learns from the errors of the previous one.
Captures complex, higher-order interactions between features.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, Optional
from loguru import logger
import joblib
import os

from src.config.model_config import XGBoostConfig


class YademXGBoost:
    """
    Model C: XGBoost (Extreme Gradient Boosting)
    
    Purpose:
    - Captures complex higher-order interactions that other models miss
    - E.g., can learn that low bureau score + high alt-data + strong
      psychometric = actually lower risk than bureau alone suggests
    - Consistently top performer in global credit scoring competitions
    """

    def __init__(self, config: Optional[XGBoostConfig] = None):
        self.config = config or XGBoostConfig()
        self._init_params = self.config.to_dict()
        # Remove early_stopping_rounds from init (used in fit)
        self._early_stopping = self._init_params.pop("early_stopping_rounds", 50)
        self.model = xgb.XGBClassifier(**self._init_params)
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
        Train the XGBoost model with optional early stopping.
        
        Returns:
            Dictionary with training metrics.
        """
        logger.info("Training XGBoost (Model C)...")
        self.feature_names = list(X_train.columns)

        # Compute scale_pos_weight from data for class imbalance
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        self.model.set_params(scale_pos_weight=n_neg / max(n_pos, 1))

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True

        train_proba = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)

        metrics = {
            "model": "xgboost",
            "train_auc_roc": round(train_auc, 4),
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "n_features": len(self.feature_names),
            "n_train_samples": len(X_train),
            "best_iteration": getattr(self.model, "best_iteration", self.config.n_estimators),
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

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return feature importances ranked by importance."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

    def save(self, path: str) -> str:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, path)
        logger.info(f"XGBoost saved to {path}")
        return path

    def load(self, path: str) -> None:
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.is_fitted = True
        logger.info(f"XGBoost loaded from {path}")
