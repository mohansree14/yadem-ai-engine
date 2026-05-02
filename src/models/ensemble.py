"""
YADEM Ensemble Scorer
=============================================================================
Orchestrates the weighted consensus of three ML models (LR, RF, XGBoost).
This is Stage 2 of the YADEM AI Engine pipeline.

Rather than relying on a single algorithm, the ensemble combines outputs
from all three models via a weighted average, improving accuracy and
reducing the risk of any one model's blind spots.

Default weights: LR=0.25, RF=0.35, XGB=0.40
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from loguru import logger
import joblib
import os

from src.models.logistic import YademLogisticRegression
from src.models.random_forest import YademRandomForest
from src.models.xgboost_model import YademXGBoost
from src.config.model_config import ModelConfig


class EnsembleScorer:
    """
    Weighted ensemble of Logistic Regression, Random Forest, and XGBoost.
    Produces a combined probability of default used for score generation.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.lr = YademLogisticRegression(self.config.logistic_regression)
        self.rf = YademRandomForest(self.config.random_forest)
        self.xgb = YademXGBoost(self.config.xgboost)
        self.weights = self.config.ensemble_weights
        self.is_fitted = False
        self.training_metrics: Dict[str, Any] = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train all three ensemble models.
        
        Returns:
            Combined training metrics from all models.
        """
        logger.info("=" * 60)
        logger.info("YADEM ENSEMBLE TRAINING")
        logger.info(f"Weights: LR={self.weights['logistic_regression']}, "
                     f"RF={self.weights['random_forest']}, "
                     f"XGB={self.weights['xgboost']}")
        logger.info("=" * 60)

        # Train each model
        lr_metrics = self.lr.train(X_train, y_train, X_val, y_val)
        rf_metrics = self.rf.train(X_train, y_train, X_val, y_val)
        xgb_metrics = self.xgb.train(X_train, y_train, X_val, y_val)

        self.is_fitted = True

        # Compute ensemble metrics
        from sklearn.metrics import roc_auc_score
        ensemble_proba_train = self.predict_proba(X_train)
        ensemble_auc_train = roc_auc_score(y_train, ensemble_proba_train)

        self.training_metrics = {
            "logistic_regression": lr_metrics,
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics,
            "ensemble": {
                "train_auc_roc": round(ensemble_auc_train, 4),
                "weights": self.weights,
            },
        }

        if X_val is not None and y_val is not None:
            ensemble_proba_val = self.predict_proba(X_val)
            ensemble_auc_val = roc_auc_score(y_val, ensemble_proba_val)
            self.training_metrics["ensemble"]["val_auc_roc"] = round(ensemble_auc_val, 4)

            logger.info("=" * 60)
            logger.info("ENSEMBLE RESULTS")
            logger.info(f"  LR  Val AUC: {lr_metrics.get('val_auc_roc', 'N/A')}")
            logger.info(f"  RF  Val AUC: {rf_metrics.get('val_auc_roc', 'N/A')}")
            logger.info(f"  XGB Val AUC: {xgb_metrics.get('val_auc_roc', 'N/A')}")
            logger.info(f"  ENS Val AUC: {ensemble_auc_val:.4f}")
            logger.info("=" * 60)

        return self.training_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return weighted ensemble probability of default.
        
        Combines: LR*0.25 + RF*0.35 + XGB*0.40
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call train() first.")

        lr_proba = self.lr.predict_proba(X)
        rf_proba = self.rf.predict_proba(X)
        xgb_proba = self.xgb.predict_proba(X)

        ensemble_proba = (
            self.weights["logistic_regression"] * lr_proba
            + self.weights["random_forest"] * rf_proba
            + self.weights["xgboost"] * xgb_proba
        )

        return ensemble_proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions from ensemble."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_individual_predictions(
        self, X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Return predictions from each individual model."""
        return {
            "logistic_regression": self.lr.predict_proba(X),
            "random_forest": self.rf.predict_proba(X),
            "xgboost": self.xgb.predict_proba(X),
        }

    def save(self, directory: str = "./models") -> Dict[str, str]:
        """Save all three models to disk."""
        os.makedirs(directory, exist_ok=True)
        paths = {
            "logistic_regression": self.lr.save(
                os.path.join(directory, "logistic_regression.joblib")
            ),
            "random_forest": self.rf.save(
                os.path.join(directory, "random_forest.joblib")
            ),
            "xgboost": self.xgb.save(
                os.path.join(directory, "xgboost.joblib")
            ),
        }
        # Save ensemble metadata
        meta_path = os.path.join(directory, "ensemble_meta.joblib")
        joblib.dump({
            "weights": self.weights,
            "training_metrics": self.training_metrics,
        }, meta_path)
        paths["meta"] = meta_path
        logger.info(f"Ensemble saved to {directory}")
        return paths

    def load(self, directory: str = "./models") -> None:
        """Load all three models from disk."""
        self.lr.load(os.path.join(directory, "logistic_regression.joblib"))
        self.rf.load(os.path.join(directory, "random_forest.joblib"))
        self.xgb.load(os.path.join(directory, "xgboost.joblib"))

        meta_path = os.path.join(directory, "ensemble_meta.joblib")
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.weights = meta.get("weights", self.weights)
            self.training_metrics = meta.get("training_metrics", {})

        self.is_fitted = True
        logger.info(f"Ensemble loaded from {directory}")
