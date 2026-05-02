"""
YADEM Risk Configuration
Defines risk bands, score ranges, and decision routing rules.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
from enum import Enum


class RiskBand(str, Enum):
    """YADEM Risk Band Classification (A-E)."""
    A = "A"  # Excellent
    B = "B"  # Good
    C = "C"  # Borderline
    D = "D"  # High Risk
    E = "E"  # Blacklist


class DecisionAction(str, Enum):
    """Routing action for each risk band."""
    AUTO_APPROVE = "AUTO_APPROVE"
    AUTO_APPROVE_STANDARD = "AUTO_APPROVE_STANDARD"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    DECLINE_COACHING = "DECLINE_COACHING"
    HARD_DECLINE = "HARD_DECLINE"


@dataclass
class RiskBandConfig:
    """Configuration for a single risk band."""
    band: RiskBand
    min_score: int
    max_score: int
    meaning: str
    action: DecisionAction
    max_tenure_months: int
    rate_multiplier: float  # 1.0 = base rate


@dataclass
class RiskConfig:
    """Complete risk configuration for the YADEM scoring engine."""

    # Score range: 0-1000
    max_score: int = 1000
    min_score: int = 0

    # Risk band definitions
    bands: Dict[RiskBand, RiskBandConfig] = field(default_factory=lambda: {
        RiskBand.A: RiskBandConfig(
            band=RiskBand.A,
            min_score=800, max_score=1000,
            meaning="Excellent",
            action=DecisionAction.AUTO_APPROVE,
            max_tenure_months=36,
            rate_multiplier=0.85,
        ),
        RiskBand.B: RiskBandConfig(
            band=RiskBand.B,
            min_score=650, max_score=799,
            meaning="Good",
            action=DecisionAction.AUTO_APPROVE_STANDARD,
            max_tenure_months=24,
            rate_multiplier=1.0,
        ),
        RiskBand.C: RiskBandConfig(
            band=RiskBand.C,
            min_score=500, max_score=649,
            meaning="Borderline",
            action=DecisionAction.MANUAL_REVIEW,
            max_tenure_months=12,
            rate_multiplier=1.25,
        ),
        RiskBand.D: RiskBandConfig(
            band=RiskBand.D,
            min_score=300, max_score=499,
            meaning="High Risk",
            action=DecisionAction.DECLINE_COACHING,
            max_tenure_months=0,
            rate_multiplier=0.0,
        ),
        RiskBand.E: RiskBandConfig(
            band=RiskBand.E,
            min_score=0, max_score=299,
            meaning="Blacklist",
            action=DecisionAction.HARD_DECLINE,
            max_tenure_months=0,
            rate_multiplier=0.0,
        ),
    })

    # Decision rules thresholds
    max_loan_to_revenue_ratio: float = 0.40  # CBN affordability test
    min_business_age_months: int = 6
    max_debt_to_income_ratio: float = 0.50
    min_monthly_revenue_ngn: float = 50_000.0

    # Fraud thresholds
    max_applications_per_bvn_per_day: int = 3
    max_applications_per_device_per_day: int = 5
    velocity_window_hours: int = 24

    def get_band(self, score: int) -> RiskBandConfig:
        """Get the risk band configuration for a given score."""
        for band_config in self.bands.values():
            if band_config.min_score <= score <= band_config.max_score:
                return band_config
        # Default to E-band if score is out of range
        return self.bands[RiskBand.E]

    def get_decision(self, score: int) -> DecisionAction:
        """Get the routing decision for a given score."""
        return self.get_band(score).action
