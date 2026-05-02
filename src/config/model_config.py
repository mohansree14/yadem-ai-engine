"""
YADEM Model Configuration
Hyperparameters for all three ensemble models and training settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class LogisticRegressionConfig:
    """Hyperparameters for Logistic Regression (Model A)."""
    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: str = "balanced"
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }


@dataclass
class RandomForestConfig:
    """Hyperparameters for Random Forest (Model B)."""
    n_estimators: int = 500
    max_depth: int = 15
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass
class XGBoostConfig:
    """Hyperparameters for XGBoost (Model C)."""
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0  # Will be computed from data
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 50
    random_state: int = 42
    n_jobs: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "eval_metric": self.eval_metric,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass
class ModelConfig:
    """Master model configuration."""
    logistic_regression: LogisticRegressionConfig = field(
        default_factory=LogisticRegressionConfig
    )
    random_forest: RandomForestConfig = field(
        default_factory=RandomForestConfig
    )
    xgboost: XGBoostConfig = field(
        default_factory=XGBoostConfig
    )

    # Ensemble weights (must sum to 1.0)
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "logistic_regression": 0.25,
        "random_forest": 0.35,
        "xgboost": 0.40,
    })

    # Training settings
    test_size: float = 0.20
    validation_size: float = 0.15
    cv_folds: int = 5
    random_state: int = 42

    # Performance thresholds
    min_auc_roc: float = 0.80
    min_gini: float = 0.60
    min_ks_statistic: float = 0.30
    max_psi: float = 0.25  # Population Stability Index drift threshold

    # Target decision time
    max_decision_time_seconds: float = 48.0
