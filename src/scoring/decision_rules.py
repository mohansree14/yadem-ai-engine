"""
YADEM Decision Rules Engine
=============================================================================
Business rules that override or modify ML-based credit decisions.
Implements 6 core rules from the YADEM architecture:
  1. CBN Affordability Test (DTI < 33%)
  2. Business Age Minimum (>= 12 months)
  3. Active Default Hard Block
  4. Sector Risk Override
  5. Loan Amount Cap by Risk Band
  6. Repeat Borrower Bonus
"""

from typing import Dict, List, Tuple
from loguru import logger


class DecisionRulesEngine:
    """Applies business rules post-ML scoring to ensure regulatory compliance."""

    # CBN max debt-to-income ratio
    MAX_DTI_RATIO = 0.33

    # Minimum business age (months)
    MIN_BUSINESS_AGE_MONTHS = 12

    # Maximum loan amounts by risk band (NGN)
    MAX_LOAN_BY_BAND = {
        "A": 50_000_000,   # ₦50M
        "B": 20_000_000,   # ₦20M
        "C": 5_000_000,    # ₦5M
        "D": 0,            # Declined
        "E": 0,            # Hard decline
    }

    # High-risk sectors with tighter limits
    HIGH_RISK_SECTORS = {
        "cryptocurrency", "gambling", "forex_trading",
        "ponzi", "mlm", "unregistered_fintech"
    }

    # Sector risk multipliers (applied to max loan)
    SECTOR_MULTIPLIERS = {
        "agriculture": 0.8,
        "retail": 1.0,
        "manufacturing": 1.1,
        "technology": 1.2,
        "healthcare": 1.1,
        "construction": 0.7,
        "transportation": 0.9,
        "hospitality": 0.85,
    }

    def __init__(self):
        logger.info("Decision Rules Engine initialized")

    def apply_rules(self, scoring_result: Dict, applicant_data: Dict) -> Dict:
        """
        Apply all business rules to a scoring result.

        Args:
            scoring_result: Output from CreditScorer (score, band, decision)
            applicant_data: Raw applicant data for rule evaluation

        Returns:
            Modified scoring result with rule overrides applied
        """
        rules_applied = []
        original_decision = scoring_result.get("decision", "UNKNOWN")
        decision = original_decision
        overrides = []

        # Rule 1: CBN Affordability Test
        r1 = self._check_affordability(applicant_data)
        rules_applied.append(r1)
        if not r1["passed"]:
            decision = "DECLINED"
            overrides.append("FAILED_AFFORDABILITY")

        # Rule 2: Business Age Minimum
        r2 = self._check_business_age(applicant_data)
        rules_applied.append(r2)
        if not r2["passed"]:
            decision = "MANUAL_REVIEW"
            overrides.append("INSUFFICIENT_BUSINESS_AGE")

        # Rule 3: Active Default Hard Block
        r3 = self._check_active_defaults(applicant_data)
        rules_applied.append(r3)
        if not r3["passed"]:
            decision = "DECLINED"
            overrides.append("ACTIVE_DEFAULT")

        # Rule 4: Sector Risk Override
        r4 = self._check_sector_risk(applicant_data)
        rules_applied.append(r4)
        if not r4["passed"]:
            decision = "DECLINED"
            overrides.append("PROHIBITED_SECTOR")

        # Rule 5: Loan Amount Cap
        r5 = self._check_loan_cap(
            scoring_result.get("risk_band", "E"),
            applicant_data.get("loan_amount_requested", 0),
            applicant_data.get("business_sector", "retail"),
        )
        rules_applied.append(r5)
        if not r5["passed"]:
            overrides.append("EXCEEDS_BAND_LIMIT")
            if decision not in ("DECLINED",):
                decision = "MANUAL_REVIEW"

        # Rule 6: Repeat Borrower Bonus
        r6 = self._check_repeat_borrower(applicant_data)
        rules_applied.append(r6)

        result = {
            **scoring_result,
            "final_decision": decision,
            "decision_overridden": decision != original_decision,
            "original_decision": original_decision,
            "rules_applied": rules_applied,
            "overrides": overrides,
            "max_approved_amount": r5.get("max_amount", 0),
        }

        if decision != original_decision:
            logger.warning(
                f"Decision overridden: {original_decision} → {decision} | "
                f"Reasons: {overrides}"
            )

        return result

    def _check_affordability(self, data: Dict) -> Dict:
        """Rule 1: CBN debt-to-income ratio check."""
        monthly_income = data.get("avg_monthly_revenue", 0)
        existing_debt = data.get("existing_monthly_obligations", 0)
        new_repayment = data.get("estimated_monthly_repayment", 0)

        total_obligations = existing_debt + new_repayment
        dti = total_obligations / max(1, monthly_income)

        return {
            "rule": "cbn_affordability",
            "passed": dti <= self.MAX_DTI_RATIO,
            "dti_ratio": round(dti, 4),
            "threshold": self.MAX_DTI_RATIO,
            "detail": f"DTI {dti:.1%} vs max {self.MAX_DTI_RATIO:.0%}",
        }

    def _check_business_age(self, data: Dict) -> Dict:
        """Rule 2: Minimum business operating history."""
        age_months = data.get("years_in_business", 0) * 12
        if "business_age_months" in data:
            age_months = data["business_age_months"]

        return {
            "rule": "business_age_minimum",
            "passed": age_months >= self.MIN_BUSINESS_AGE_MONTHS,
            "business_age_months": age_months,
            "minimum_months": self.MIN_BUSINESS_AGE_MONTHS,
        }

    def _check_active_defaults(self, data: Dict) -> Dict:
        """Rule 3: Hard block if applicant has active defaults."""
        has_default = data.get("has_active_default", False)
        if not has_default:
            has_default = data.get("delinquent_accounts", 0) > 0

        return {
            "rule": "active_default_block",
            "passed": not has_default,
            "has_active_default": has_default,
        }

    def _check_sector_risk(self, data: Dict) -> Dict:
        """Rule 4: Block prohibited sectors."""
        sector = data.get("business_sector", "").lower()
        is_prohibited = sector in self.HIGH_RISK_SECTORS

        return {
            "rule": "sector_risk",
            "passed": not is_prohibited,
            "sector": sector,
            "prohibited": is_prohibited,
        }

    def _check_loan_cap(self, risk_band: str, requested_amount: float,
                        sector: str) -> Dict:
        """Rule 5: Enforce loan amount caps by risk band."""
        base_max = self.MAX_LOAN_BY_BAND.get(risk_band, 0)
        multiplier = self.SECTOR_MULTIPLIERS.get(sector.lower(), 1.0)
        max_amount = base_max * multiplier

        return {
            "rule": "loan_amount_cap",
            "passed": requested_amount <= max_amount,
            "requested": requested_amount,
            "max_amount": max_amount,
            "risk_band": risk_band,
            "sector_multiplier": multiplier,
        }

    def _check_repeat_borrower(self, data: Dict) -> Dict:
        """Rule 6: Apply bonus for repeat borrowers with good history."""
        is_repeat = data.get("previous_loans_count", 0) > 0
        repayment_rate = data.get("previous_repayment_rate", 0)

        bonus_eligible = is_repeat and repayment_rate >= 0.95

        return {
            "rule": "repeat_borrower_bonus",
            "passed": True,  # This rule never blocks, only enhances
            "is_repeat_borrower": is_repeat,
            "repayment_rate": repayment_rate,
            "bonus_eligible": bonus_eligible,
            "score_bonus": 25 if bonus_eligible else 0,
        }
