"""
YADEM Credit Bureau Data Ingestion
=============================================================================
Integrates with Nigerian credit bureaus: CRC, First Central, XDS Credit.
For MVP: accepts pre-formatted bureau reports as JSON.
"""

from typing import Dict, Optional
from loguru import logger


class CreditBureauIngester:
    """Ingests and normalizes credit bureau data."""

    SUPPORTED_BUREAUS = {"crc", "first_central", "xds", "manual"}

    def __init__(self, bureau: str = "manual"):
        self.bureau = bureau
        logger.info(f"Credit Bureau Ingester initialized (bureau: {bureau})")

    def ingest(self, raw_data: Dict) -> Dict:
        """
        Normalize bureau report data.

        Expected format:
        {
            "bureau_score": 650,
            "total_accounts": 5,
            "active_accounts": 3,
            "delinquent_accounts": 0,
            "total_outstanding": 500000,
            "monthly_obligations": 45000,
            "max_days_past_due": 0,
            "enquiry_count_6m": 2,
            "oldest_account_months": 36,
            "credit_utilization": 0.45,
        }
        """
        normalized = {
            "bureau_score": raw_data.get("bureau_score", 0),
            "total_accounts": raw_data.get("total_accounts", 0),
            "active_accounts": raw_data.get("active_accounts", 0),
            "delinquent_accounts": raw_data.get("delinquent_accounts", 0),
            "total_outstanding": raw_data.get("total_outstanding", 0),
            "monthly_obligations": raw_data.get("monthly_obligations", 0),
            "max_days_past_due": raw_data.get("max_days_past_due", 0),
            "enquiry_count_6m": raw_data.get("enquiry_count_6m", 0),
            "oldest_account_months": raw_data.get("oldest_account_months", 0),
            "credit_utilization": raw_data.get("credit_utilization", 0.0),
            "has_active_default": raw_data.get("delinquent_accounts", 0) > 0,
            "bureau_source": self.bureau,
        }

        logger.info(f"Bureau data ingested: score={normalized['bureau_score']}")
        return normalized

    def compute_risk_indicators(self, bureau_data: Dict) -> Dict:
        """Derive risk indicators from bureau data."""
        score = bureau_data.get("bureau_score", 0)
        return {
            "bureau_risk_level": (
                "low" if score >= 700 else
                "medium" if score >= 500 else
                "high"
            ),
            "debt_burden_ratio": (
                bureau_data.get("monthly_obligations", 0) /
                max(1, bureau_data.get("monthly_income_estimate", 100000))
            ),
            "credit_history_depth": (
                "deep" if bureau_data.get("oldest_account_months", 0) >= 24 else
                "moderate" if bureau_data.get("oldest_account_months", 0) >= 12 else
                "thin"
            ),
            "enquiry_velocity": (
                "high" if bureau_data.get("enquiry_count_6m", 0) >= 5 else
                "normal"
            ),
        }
