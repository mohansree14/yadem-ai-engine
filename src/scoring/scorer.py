"""
YADEM Credit Scorer
=============================================================================
Converts ensemble probability output into a YADEM credit score (0-1,000)
and maps it to risk bands (A-E) with routing decisions.
This is Stage 3 of the YADEM AI Engine pipeline.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

from src.config.risk_config import RiskConfig, RiskBand, DecisionAction


@dataclass
class CreditDecision:
    """Complete credit decision output from the YADEM engine."""
    applicant_id: str
    yadem_score: int                    # 0-1000
    probability_of_default: float       # 0.0-1.0
    risk_band: str                      # A, B, C, D, E
    risk_band_meaning: str              # Excellent, Good, etc.
    decision: str                       # AUTO_APPROVE, MANUAL_REVIEW, etc.
    max_tenure_months: int
    rate_multiplier: float
    individual_scores: Dict[str, float]  # Per-model probabilities
    decision_rules_applied: List[str]   # Rules that were triggered
    is_fraud_flagged: bool


class CreditScorer:
    """
    Converts probability of default to YADEM score (0-1000)
    and applies decision routing based on risk bands.
    """

    def __init__(self, risk_config: Optional[RiskConfig] = None):
        self.risk_config = risk_config or RiskConfig()

    def score(
        self,
        applicant_id: str,
        prob_default: float,
        individual_probs: Dict[str, float],
        applicant_data: Optional[Dict[str, Any]] = None,
        fraud_result: Optional[Dict[str, Any]] = None,
    ) -> CreditDecision:
        """
        Generate a complete credit decision.
        
        Args:
            applicant_id: Unique applicant identifier.
            prob_default: Ensemble probability of default (0-1).
            individual_probs: Per-model probabilities.
            applicant_data: Raw applicant data for decision rules.
            fraud_result: Fraud screening result.
            
        Returns:
            CreditDecision with score, band, and routing.
        """
        # Convert probability to 0-1000 score (inverse — lower default = higher score)
        yadem_score = self._probability_to_score(prob_default)

        # Check fraud override
        is_fraud = False
        if fraud_result and not fraud_result.get("passed", True):
            yadem_score = 0  # Override to E-band
            is_fraud = True

        # Get risk band
        band_config = self.risk_config.get_band(yadem_score)

        # Apply decision rules
        rules_applied = []
        if applicant_data:
            yadem_score, rules_applied = self._apply_decision_rules(
                yadem_score, applicant_data
            )
            # Re-evaluate band after rules
            band_config = self.risk_config.get_band(yadem_score)

        return CreditDecision(
            applicant_id=applicant_id,
            yadem_score=yadem_score,
            probability_of_default=round(prob_default, 4),
            risk_band=band_config.band.value,
            risk_band_meaning=band_config.meaning,
            decision=band_config.action.value,
            max_tenure_months=band_config.max_tenure_months,
            rate_multiplier=band_config.rate_multiplier,
            individual_scores={
                k: round(v, 4) for k, v in individual_probs.items()
            },
            decision_rules_applied=rules_applied,
            is_fraud_flagged=is_fraud,
        )

    def _probability_to_score(self, prob_default: float) -> int:
        """
        Convert probability of default to YADEM score (0-1000).
        Uses an inverse sigmoid-like mapping:
        - prob=0.0 → score=1000 (no risk)
        - prob=0.5 → score=~500
        - prob=1.0 → score=0 (certain default)
        """
        # Clamp probability
        prob = max(0.001, min(0.999, prob_default))

        # Non-linear mapping that spreads scores across the range
        # Using a logit-based transformation for better discrimination
        raw_score = (1 - prob) * 1000

        # Apply slight non-linearity to spread middle range
        if raw_score > 500:
            adjusted = 500 + (raw_score - 500) ** 0.95
        else:
            adjusted = 500 - (500 - raw_score) ** 0.95

        return int(np.clip(adjusted, 0, 1000))

    def _apply_decision_rules(
        self,
        score: int,
        data: Dict[str, Any],
    ) -> tuple:
        """
        Apply business rules on top of the ML score.
        Stage 4 of the YADEM Engine pipeline.
        
        Rules encode:
        - Regulatory requirements (CBN affordability test)
        - Product constraints (min business age)
        - Risk appetite (hard blocks on active defaults)
        - Override rules (strategic sectors)
        """
        rules_applied = []

        # Rule 1: Hard block on active defaults at any bureau
        if data.get("has_active_default", 0) == 1:
            score = min(score, 250)  # Force to D/E band
            rules_applied.append("HARD_BLOCK: Active default at credit bureau")

        # Rule 2: Affordability test (CBN requirement)
        loan_amount = data.get("requested_loan_amount_ngn", 0)
        monthly_revenue = data.get("avg_monthly_revenue_6m", 0)
        if monthly_revenue > 0:
            loan_to_revenue = loan_amount / (monthly_revenue * 12)
            if loan_to_revenue > self.risk_config.max_loan_to_revenue_ratio:
                score = min(score, 600)  # Cap at C-band max
                rules_applied.append(
                    f"AFFORDABILITY: Loan-to-revenue ratio {loan_to_revenue:.1%} "
                    f"exceeds {self.risk_config.max_loan_to_revenue_ratio:.0%} limit"
                )

        # Rule 3: Minimum business age
        biz_age = data.get("business_age_months", 0)
        if biz_age < self.risk_config.min_business_age_months:
            score = min(score, 550)
            rules_applied.append(
                f"BUSINESS_AGE: {biz_age} months < minimum "
                f"{self.risk_config.min_business_age_months} months"
            )

        # Rule 4: KYC verification requirement
        if data.get("bvn_verified", 0) == 0:
            score = min(score, 300)
            rules_applied.append("KYC_FAIL: BVN not verified")

        # Rule 5: Positive override for agriculture during planting season
        if (
            data.get("business_sector") == "agriculture"
            and data.get("market_cycle_indicator", 0) > 0.7
            and score >= 450
        ):
            score = min(score + 50, 1000)
            rules_applied.append("SECTOR_OVERRIDE: Agriculture planting season bonus +50")

        # Rule 6: Returning customer bonus
        if (
            data.get("is_returning_applicant", 0) == 1
            and data.get("prior_repayment_rate", 0) > 0.95
            and data.get("prior_defaults", 0) == 0
        ):
            score = min(score + 30, 1000)
            rules_applied.append("LOYALTY_BONUS: Excellent returning customer +30")

        return score, rules_applied

    def batch_score(
        self,
        applicant_ids: List[str],
        probs_default: np.ndarray,
        individual_probs_list: List[Dict[str, float]],
    ) -> List[CreditDecision]:
        """Score a batch of applicants."""
        decisions = []
        for i in range(len(applicant_ids)):
            decision = self.score(
                applicant_id=applicant_ids[i],
                prob_default=probs_default[i],
                individual_probs=individual_probs_list[i],
            )
            decisions.append(decision)
        return decisions
