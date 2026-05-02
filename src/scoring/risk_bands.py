"""
YADEM Risk Band Mapping
=============================================================================
Standalone module for mapping scores to risk bands A-E.
"""

from typing import Dict, Tuple
from enum import Enum


class RiskBand(str, Enum):
    A = "A"  # 800-1000: Excellent
    B = "B"  # 650-799:  Good
    C = "C"  # 500-649:  Borderline
    D = "D"  # 300-499:  High Risk
    E = "E"  # 0-299:    Blacklist


BAND_DEFINITIONS = {
    RiskBand.A: {"min": 800, "max": 1000, "label": "Excellent", "routing": "AUTO_APPROVE",
                 "max_tenure_months": 36, "rate_multiplier": 1.0},
    RiskBand.B: {"min": 650, "max": 799, "label": "Good", "routing": "AUTO_APPROVE",
                 "max_tenure_months": 24, "rate_multiplier": 1.15},
    RiskBand.C: {"min": 500, "max": 649, "label": "Borderline", "routing": "MANUAL_REVIEW",
                 "max_tenure_months": 12, "rate_multiplier": 1.35},
    RiskBand.D: {"min": 300, "max": 499, "label": "High Risk", "routing": "DECLINE",
                 "max_tenure_months": 0, "rate_multiplier": 0},
    RiskBand.E: {"min": 0, "max": 299, "label": "Blacklist", "routing": "HARD_DECLINE",
                 "max_tenure_months": 0, "rate_multiplier": 0},
}


def score_to_band(score: int) -> RiskBand:
    """Map a YADEM score (0-1000) to a risk band."""
    score = max(0, min(1000, score))
    for band, defn in BAND_DEFINITIONS.items():
        if defn["min"] <= score <= defn["max"]:
            return band
    return RiskBand.E


def get_band_details(band: RiskBand) -> Dict:
    """Get full details for a risk band."""
    return {
        "band": band.value,
        **BAND_DEFINITIONS[band],
    }


def get_routing_decision(score: int) -> Dict:
    """Get the routing decision for a given score."""
    band = score_to_band(score)
    details = BAND_DEFINITIONS[band]
    return {
        "score": score,
        "risk_band": band.value,
        "label": details["label"],
        "decision": details["routing"],
        "max_tenure_months": details["max_tenure_months"],
        "rate_multiplier": details["rate_multiplier"],
    }
