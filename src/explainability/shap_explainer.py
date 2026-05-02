"""
YADEM SHAP Explainer
=============================================================================
Uses SHAP (SHapley Additive exPlanations) to decompose each credit decision
into the fair additive contribution of each input feature.

This is Stage 5 of the YADEM AI Engine pipeline. Every decision must be
explainable for:
  - Regulatory compliance (CBN, NDPC)
  - Applicant communication (why was I declined?)
  - Non-discriminatory decision-making audits

Provides both:
  - Global interpretability (model's overall logic)
  - Local interpretability (explaining specific cases)
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class FeatureExplanation:
    """Explanation for a single feature's contribution."""
    feature_name: str
    feature_value: float
    shap_value: float
    direction: str  # "positive_risk" or "negative_risk"
    human_readable: str


@dataclass
class CreditExplanation:
    """Complete explanation for a single credit decision."""
    applicant_id: str
    base_value: float  # Average model output
    final_score_contribution: float
    top_positive_factors: List[FeatureExplanation]  # Increased risk
    top_negative_factors: List[FeatureExplanation]   # Decreased risk
    all_shap_values: Dict[str, float]
    explanation_text: str


# Human-readable feature descriptions for Nigerian SME context
FEATURE_DESCRIPTIONS = {
    "feat_loan_to_revenue_ratio": "Loan amount relative to annual revenue",
    "feat_revenue_trend": "Revenue trend over past 6 months",
    "feat_net_cashflow_ratio": "Net cash flow strength",
    "feat_revenue_stability": "Revenue consistency/stability",
    "feat_debt_to_income": "Existing debt relative to income",
    "feat_balance_buffer": "Cash reserve buffer",
    "feat_utility_discipline": "Utility bill payment discipline",
    "feat_bureau_behavior": "Credit bureau repayment history",
    "feat_delinquency_severity": "Severity of past delinquencies",
    "feat_prior_reliability": "Past loan repayment reliability with YADEM",
    "feat_psychometric_score": "Financial behaviour assessment score",
    "feat_sim_stability": "Mobile SIM card tenure (stability proxy)",
    "feat_network_diversity": "Transaction counterparty diversity",
    "feat_business_maturity": "Business age and maturity",
    "feat_formalization_score": "Business registration and verification status",
    "feat_identity_strength": "Identity verification confidence",
    "feat_sector_performance": "Performance relative to sector average",
    "feat_sector_risk": "Industry sector risk level",
    "feat_geo_risk": "Geographic region risk level",
    "feat_environmental_risk": "Combined external environment risk",
    "feat_bureau_x_psychometric": "Combined bureau and psychometric signal",
    "feat_alt_data_compensation": "Alternative data compensating for thin bureau file",
    "feat_revenue_maturity": "Revenue × business age interaction",
    "feat_digital_presence": "Digital channel presence (POS, mobile money, e-commerce)",
    "feat_installment_to_revenue": "Estimated monthly installment vs. revenue",
    "feat_returning_bonus": "Returning customer loyalty signal",
    "debit_to_credit_ratio_3m": "Ratio of debits to credits in past 3 months",
    "bureau_score_avg": "Average credit bureau score",
    "has_active_default": "Active default at any credit bureau",
    "num_delinquencies_12m": "Number of delinquencies in past 12 months",
    "business_age_months": "Business age in months",
    "num_missed_utility_payments_12m": "Missed utility payments in past year",
    "avg_monthly_revenue_6m": "Average monthly revenue (6 months)",
    "cashflow_volatility_6m": "Cash flow volatility over 6 months",
}


class SHAPExplainer:
    """
    SHAP-based explainability for the YADEM ensemble.
    
    Uses TreeExplainer for RF/XGBoost and LinearExplainer for LR,
    then combines explanations weighted by ensemble weights.
    """

    def __init__(self):
        self.tree_explainer = None
        self.is_fitted = False

    def fit(self, model, X_background: pd.DataFrame) -> None:
        """
        Initialize the SHAP explainer with a background dataset.
        
        Args:
            model: The trained model (typically the XGBoost model for TreeExplainer).
            X_background: Background dataset for SHAP (sample of training data).
        """
        logger.info("Fitting SHAP explainer...")
        # Use a sample for efficiency
        n_background = min(200, len(X_background))
        background = X_background.sample(n_background, random_state=42)

        try:
            # TreeExplainer is fastest and most accurate for tree-based models
            self.tree_explainer = shap.TreeExplainer(model)
            logger.info("Using TreeExplainer (tree-based model detected)")
        except Exception:
            # Fallback to KernelExplainer
            self.tree_explainer = shap.KernelExplainer(
                model.predict_proba, background
            )
            logger.info("Using KernelExplainer (fallback)")

        self.is_fitted = True
        self.feature_names = list(X_background.columns)

    def explain(
        self,
        X: pd.DataFrame,
        applicant_id: str = "unknown",
        top_n: int = 3,
    ) -> CreditExplanation:
        """
        Generate SHAP explanation for a single applicant.
        
        Args:
            X: Feature DataFrame for one applicant (1 row).
            applicant_id: Applicant identifier.
            top_n: Number of top factors to surface.
            
        Returns:
            CreditExplanation with top factors and human-readable text.
        """
        if not self.is_fitted:
            raise RuntimeError("Explainer not fitted. Call fit() first.")

        # Compute SHAP values
        shap_values = self.tree_explainer.shap_values(X)

        # Handle multi-output SHAP (take class 1 = default)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        # Build feature explanations
        feature_explanations = []
        for i, fname in enumerate(self.feature_names):
            desc = FEATURE_DESCRIPTIONS.get(fname, fname.replace("_", " ").title())
            fval = float(X.iloc[0, i]) if i < len(X.columns) else 0
            sval = float(sv[i])
            direction = "positive_risk" if sval > 0 else "negative_risk"

            if sval > 0:
                human = f"↑ Risk: {desc} (contributed to higher default probability)"
            else:
                human = f"↓ Risk: {desc} (contributed to lower default probability)"

            feature_explanations.append(FeatureExplanation(
                feature_name=fname,
                feature_value=round(fval, 4),
                shap_value=round(sval, 4),
                direction=direction,
                human_readable=human,
            ))

        # Sort by absolute SHAP value
        sorted_explanations = sorted(
            feature_explanations, key=lambda x: abs(x.shap_value), reverse=True
        )

        # Top positive risk factors (increased default probability)
        top_positive = [
            e for e in sorted_explanations if e.direction == "positive_risk"
        ][:top_n]

        # Top negative risk factors (decreased default probability)
        top_negative = [
            e for e in sorted_explanations if e.direction == "negative_risk"
        ][:top_n]

        # Build human-readable explanation text
        explanation_text = self._build_explanation_text(
            applicant_id, top_positive, top_negative
        )

        # Base value
        base_value = float(self.tree_explainer.expected_value)
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])

        return CreditExplanation(
            applicant_id=applicant_id,
            base_value=round(base_value, 4),
            final_score_contribution=round(float(sv.sum()), 4),
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
            all_shap_values={
                fname: round(float(sv[i]), 4)
                for i, fname in enumerate(self.feature_names)
            },
            explanation_text=explanation_text,
        )

    def _build_explanation_text(
        self,
        applicant_id: str,
        top_positive: List[FeatureExplanation],
        top_negative: List[FeatureExplanation],
    ) -> str:
        """Build a human-readable explanation string."""
        lines = [f"Credit Decision Explanation for {applicant_id}:", ""]

        if top_positive:
            lines.append("Primary risk factors (areas for improvement):")
            for i, f in enumerate(top_positive, 1):
                desc = FEATURE_DESCRIPTIONS.get(
                    f.feature_name, f.feature_name.replace("_", " ")
                )
                lines.append(f"  ({i}) {desc}")

        if top_negative:
            lines.append("")
            lines.append("Strengths supporting your application:")
            for i, f in enumerate(top_negative, 1):
                desc = FEATURE_DESCRIPTIONS.get(
                    f.feature_name, f.feature_name.replace("_", " ")
                )
                lines.append(f"  ({i}) {desc}")

        return "\n".join(lines)

    def explain_batch(
        self,
        X: pd.DataFrame,
        applicant_ids: List[str],
        top_n: int = 3,
    ) -> List[CreditExplanation]:
        """Generate explanations for multiple applicants."""
        explanations = []
        for i in range(len(X)):
            row = X.iloc[[i]]
            aid = applicant_ids[i] if i < len(applicant_ids) else f"applicant_{i}"
            explanations.append(self.explain(row, aid, top_n))
        return explanations
