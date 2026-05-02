"""
YADEM Model Training Pipeline
=============================================================================
End-to-end training script that:
  1. Generates synthetic data (or loads existing data)
  2. Engineers 100+ features
  3. Cleans and preprocesses data
  4. Trains the 3-model ensemble (LR + RF + XGBoost)
  5. Evaluates model performance (AUC, Gini, KS)
  6. Fits SHAP explainer
  7. Saves all artifacts

Usage:
    python train.py
    python train.py --samples 10000
    python train.py --data-path data/my_data.csv
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve,
)
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.synthetic.generator import SyntheticDataGenerator
from src.features.engine import FeatureEngine
from src.data.processing.cleaner import DataCleaner
from src.models.ensemble import EnsembleScorer
from src.scoring.scorer import CreditScorer
from src.explainability.shap_explainer import SHAPExplainer
from src.config.model_config import ModelConfig
from src.config.risk_config import RiskConfig


def compute_ks_statistic(y_true, y_prob):
    """Compute Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    return ks


def compute_gini(auc):
    """Compute Gini coefficient from AUC."""
    return 2 * auc - 1


def train(
    n_samples: int = 5000,
    data_path: str = None,
    model_dir: str = "./models",
    data_dir: str = "./data/synthetic",
):
    """Run the complete training pipeline."""
    total_start = time.time()
    logger.info("=" * 70)
    logger.info("  YADEM AI CREDIT DECISIONING ENGINE — TRAINING PIPELINE")
    logger.info("=" * 70)

    # ================================================================
    # STEP 1: Generate or load data
    # ================================================================
    logger.info("\n📊 STEP 1: Data Generation / Loading")
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info(f"Generating {n_samples} synthetic SME records...")
        generator = SyntheticDataGenerator(seed=42)
        df, filepath = generator.generate_and_save(n_samples, data_dir)
        logger.info(f"Data saved to {filepath}")

    logger.info(f"Dataset: {df.shape[0]} records, {df.shape[1]} columns")
    logger.info(f"Default rate: {df['default'].mean():.2%}")

    # ================================================================
    # STEP 2: Feature Engineering
    # ================================================================
    logger.info("\n🔧 STEP 2: Feature Engineering")
    fe = FeatureEngine()
    df = fe.compute_features(df)
    logger.info(f"Total features after engineering: {df.shape[1]}")

    # ================================================================
    # STEP 3: Data Preprocessing
    # ================================================================
    logger.info("\n🧹 STEP 3: Data Preprocessing")
    cleaner = DataCleaner()
    X, y, feature_columns = cleaner.fit_transform(df, target_col="default")
    logger.info(f"Final feature matrix: {X.shape}")
    logger.info(f"Feature count: {len(feature_columns)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ================================================================
    # STEP 4: Train Ensemble
    # ================================================================
    logger.info("\n🤖 STEP 4: Training 3-Model Ensemble")
    config = ModelConfig()
    ensemble = EnsembleScorer(config)
    metrics = ensemble.train(X_train, y_train, X_val, y_val)

    # ================================================================
    # STEP 5: Evaluate on Test Set
    # ================================================================
    logger.info("\n📈 STEP 5: Test Set Evaluation")
    test_proba = ensemble.predict_proba(X_test)
    test_pred = (test_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, test_proba)
    gini = compute_gini(auc)
    ks = compute_ks_statistic(y_test, test_proba)

    logger.info(f"  Test AUC-ROC:  {auc:.4f} {'✓' if auc >= 0.80 else '✗ (target: 0.80)'}")
    logger.info(f"  Test Gini:     {gini:.4f} {'✓' if gini >= 0.60 else '✗ (target: 0.60)'}")
    logger.info(f"  Test KS:       {ks:.4f} {'✓' if ks >= 0.30 else '✗ (target: 0.30)'}")

    logger.info("\n  Classification Report:")
    report = classification_report(y_test, test_pred)
    for line in report.split("\n"):
        logger.info(f"    {line}")

    cm = confusion_matrix(y_test, test_pred)
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    logger.info(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # ================================================================
    # STEP 6: Score Distribution
    # ================================================================
    logger.info("\n📊 STEP 6: Score Distribution")
    scorer = CreditScorer(RiskConfig())
    test_scores = []
    for prob in test_proba:
        score = scorer._probability_to_score(prob)
        test_scores.append(score)

    test_scores = np.array(test_scores)
    band_counts = {}
    risk_config = RiskConfig()
    for band_name, band_cfg in risk_config.bands.items():
        count = ((test_scores >= band_cfg.min_score) & (test_scores <= band_cfg.max_score)).sum()
        pct = count / len(test_scores) * 100
        band_counts[band_name.value] = count
        logger.info(f"  Band {band_name.value} ({band_cfg.meaning:10s}): "
                     f"{count:5d} ({pct:5.1f}%)")

    # ================================================================
    # STEP 7: SHAP Explainer
    # ================================================================
    logger.info("\n🔍 STEP 7: Fitting SHAP Explainer")
    explainer = SHAPExplainer()
    try:
        # Use XGBoost model for TreeExplainer (most efficient)
        explainer.fit(ensemble.xgb.model, X_train)

        # Demo explanation on first test record
        demo_explanation = explainer.explain(
            X_test.iloc[[0]], "DEMO-APPLICANT", top_n=3
        )
        logger.info(f"\n  Demo explanation:")
        logger.info(f"  {demo_explanation.explanation_text}")
        logger.info("  ✓ SHAP explainer fitted successfully")
    except Exception as e:
        logger.warning(f"  SHAP fitting failed (non-critical): {e}")

    # ================================================================
    # STEP 8: Save Artifacts
    # ================================================================
    logger.info("\n💾 STEP 8: Saving Model Artifacts")
    os.makedirs(model_dir, exist_ok=True)

    # Save ensemble models
    ensemble.save(model_dir)

    # Save data cleaner
    cleaner_path = os.path.join(model_dir, "data_cleaner.joblib")
    joblib.dump(cleaner, cleaner_path)
    logger.info(f"  Data cleaner → {cleaner_path}")

    # Save feature engine metadata
    fe_path = os.path.join(model_dir, "feature_engine.joblib")
    joblib.dump({"feature_names": fe.computed_feature_names}, fe_path)

    # Save training metrics
    metrics_path = os.path.join(model_dir, "training_metrics.joblib")
    all_metrics = {
        **metrics,
        "test_metrics": {
            "auc_roc": round(auc, 4),
            "gini": round(gini, 4),
            "ks_statistic": round(ks, 4),
        },
        "data_info": {
            "n_samples": len(df),
            "n_features": len(feature_columns),
            "default_rate": round(y.mean(), 4),
        },
    }
    joblib.dump(all_metrics, metrics_path)

    # Save explainer
    try:
        explainer_path = os.path.join(model_dir, "shap_explainer.joblib")
        joblib.dump(explainer, explainer_path)
        logger.info(f"  SHAP explainer → {explainer_path}")
    except Exception:
        logger.warning("  Could not save SHAP explainer (non-critical)")

    # ================================================================
    # SUMMARY
    # ================================================================
    total_time = time.time() - total_start
    logger.info("\n" + "=" * 70)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time:     {total_time:.1f}s")
    logger.info(f"  Test AUC-ROC:   {auc:.4f}")
    logger.info(f"  Test Gini:      {gini:.4f}")
    logger.info(f"  Test KS:        {ks:.4f}")
    logger.info(f"  Model dir:      {os.path.abspath(model_dir)}")
    logger.info(f"  Features used:  {len(feature_columns)}")
    logger.info(f"\n  Run the API with:")
    logger.info(f"    uvicorn src.api.main:app --reload --port 8000")
    logger.info("=" * 70)

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YADEM AI Engine Training Pipeline"
    )
    parser.add_argument(
        "--samples", type=int, default=5000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to existing CSV data file"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./models",
        help="Directory to save trained models"
    )

    args = parser.parse_args()
    train(
        n_samples=args.samples,
        data_path=args.data_path,
        model_dir=args.model_dir,
    )
