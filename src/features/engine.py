"""
YADEM Feature Engineering Engine
=============================================================================
Transforms raw data into 100+ meaningful features organized by the
"5 C's of Credit" framework adapted for African SMEs:
  - Capacity Proxies: cash flow, revenue, affordability
  - Character Proxies: network diversity, payment consistency
  - Capital Proxies: assets, balances, investment
  - Collateral Proxies: business formalization, registrations
  - Conditions Proxies: sector, geography, macro environment

This is Stage 1 of the YADEM AI Engine pipeline.
"""

import numpy as np
import pandas as pd
from typing import List
from loguru import logger


class FeatureEngine:
    """
    Master feature engineering pipeline that generates 100+ features
    from raw SME applicant data across all 7 data categories.
    """

    def __init__(self):
        self.computed_feature_names: List[str] = []

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all engineered features from raw applicant data.
        
        Args:
            df: Raw applicant DataFrame with columns from all 7 data categories.
            
        Returns:
            DataFrame with original columns plus engineered features.
        """
        logger.info(f"Engineering features for {len(df)} applicants...")

        df = df.copy()

        # --- Capacity Proxies ---
        df = self._capacity_features(df)

        # --- Character Proxies ---
        df = self._character_features(df)

        # --- Capital Proxies ---
        df = self._capital_features(df)

        # --- Collateral Proxies ---
        df = self._collateral_features(df)

        # --- Conditions Proxies ---
        df = self._conditions_features(df)

        # --- Interaction Features ---
        df = self._interaction_features(df)

        # --- Ratio Features ---
        df = self._ratio_features(df)

        # Track computed features
        self.computed_feature_names = [
            c for c in df.columns if c.startswith("feat_")
        ]
        logger.info(
            f"Engineered {len(self.computed_feature_names)} new features "
            f"(total columns: {df.shape[1]})"
        )
        return df

    def _capacity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features measuring ability to repay."""

        # Revenue trends
        df["feat_revenue_trend"] = np.where(
            df["avg_monthly_revenue_6m"] > 0,
            (df["avg_monthly_revenue_3m"] - df["avg_monthly_revenue_6m"])
            / (df["avg_monthly_revenue_6m"] + 1),
            0,
        )

        # Affordability ratio (loan amount vs. revenue)
        df["feat_loan_to_revenue_ratio"] = (
            df["requested_loan_amount_ngn"]
            / (df["avg_monthly_revenue_6m"] * 6 + 1)
        )

        # Monthly debt service capacity
        df["feat_monthly_surplus"] = (
            df["avg_monthly_revenue_6m"]
            - df["total_outflows_6m"] / 6
        )

        # Net cash flow strength
        df["feat_net_cashflow_ratio"] = (
            (df["total_inflows_6m"] - df["total_outflows_6m"])
            / (df["total_inflows_6m"] + 1)
        )

        # Revenue stability (inverse of volatility)
        df["feat_revenue_stability"] = 1 / (1 + df["cashflow_volatility_6m"])

        # Debt-to-income (if bureau data available)
        df["feat_debt_to_income"] = np.where(
            df["avg_monthly_revenue_6m"] > 0,
            df["total_outstanding_debt_ngn"]
            / (df["avg_monthly_revenue_6m"] * 12 + 1),
            0,
        )

        # Balance buffer (min balance as % of avg revenue)
        df["feat_balance_buffer"] = (
            df["min_monthly_balance_6m"]
            / (df["avg_monthly_revenue_6m"] + 1)
        )

        # Transaction intensity
        df["feat_transaction_intensity"] = (
            df["num_credit_transactions_6m"] + df["num_debit_transactions_6m"]
        )

        return df

    def _character_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features measuring willingness to repay and trustworthiness."""

        # Network diversity (number of unique counterparties)
        df["feat_network_diversity"] = np.log1p(df["unique_counterparties"])

        # Utility payment discipline
        df["feat_utility_discipline"] = (
            df["utility_payment_consistency"]
            * (1 - df["num_missed_utility_payments_12m"] / 12)
        ).clip(0, 1)

        # Bureau behavior score
        df["feat_bureau_behavior"] = np.where(
            df["has_bureau_record"] == 1,
            (df["num_on_time_payments_12m"]
             / (df["num_on_time_payments_12m"] + df["num_delinquencies_12m"] + 1)),
            0.5,  # Neutral for thin-file
        )

        # Delinquency severity
        df["feat_delinquency_severity"] = (
            df["num_delinquencies_12m"] * df["worst_delinquency_days"] / 30
        )

        # Prior repayment reliability
        df["feat_prior_reliability"] = np.where(
            df["is_returning_applicant"] == 1,
            df["prior_repayment_rate"]
            * (1 - df["prior_defaults"] * 0.5),
            0.5,  # Neutral for new applicants
        )

        # Psychometric composite (already computed but add normalized version)
        df["feat_psychometric_score"] = df["psychometric_composite"]

        # SIM tenure (stability proxy)
        df["feat_sim_stability"] = np.log1p(df["sim_age_months"]) / np.log1p(120)

        return df

    def _capital_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features measuring financial assets and capital."""

        # Average balance as months of revenue
        df["feat_balance_months"] = (
            df["avg_monthly_balance"]
            / (df["avg_monthly_revenue_6m"] + 1)
        )

        # POS business value
        df["feat_pos_value"] = np.where(
            df["has_pos_terminal"] == 1,
            np.log1p(df["pos_monthly_volume_ngn"]),
            0,
        )

        # Mobile money activity
        df["feat_mobile_money_activity"] = np.where(
            df["has_mobile_money"] == 1,
            np.log1p(df["mobile_money_monthly_volume"]),
            0,
        )

        # Total digital presence value
        df["feat_digital_presence"] = (
            df["has_pos_terminal"]
            + df["has_mobile_money"]
            + df["has_ecommerce_presence"]
            + df["has_social_media_business"]
        ) / 4

        # E-commerce quality
        df["feat_ecommerce_quality"] = np.where(
            df["has_ecommerce_presence"] == 1,
            df["ecommerce_avg_rating"] / 5 * (1 - df["ecommerce_return_rate"]),
            0,
        )

        return df

    def _collateral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features measuring business formalization and tangibility."""

        # Business maturity score
        df["feat_business_maturity"] = np.log1p(
            df["business_age_months"]
        ) / np.log1p(180)

        # Formalization score
        df["feat_formalization_score"] = (
            df["is_registered_cac"] * 0.40
            + df["bvn_verified"] * 0.25
            + df["nin_verified"] * 0.20
            + df["cac_verified"] * 0.15
        )

        # Identity verification strength
        df["feat_identity_strength"] = df["identity_match_score"]

        # Business scale (employees)
        df["feat_business_scale"] = np.log1p(df["num_employees"])

        # Director involvement
        df["feat_governance"] = np.log1p(df["director_count"])

        return df

    def _conditions_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features measuring external conditions (sector, geography, macro)."""

        # Sector-adjusted performance
        df["feat_sector_performance"] = df["revenue_vs_sector_avg"]

        # Sector risk level
        df["feat_sector_risk"] = df["sector_risk_weight"]

        # Geographic risk
        df["feat_geo_risk"] = df["geographic_risk_multiplier"]

        # Market timing
        df["feat_market_timing"] = df["market_cycle_indicator"]

        # Combined environmental risk
        df["feat_environmental_risk"] = (
            df["sector_risk_weight"] * 0.5
            + df["geographic_risk_multiplier"] * 0.3
            + (1 - df["market_cycle_indicator"]) * 0.2
        )

        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-category interaction features for non-linear signal capture."""

        # Bureau score × psychometric (captures "thin-file but willing" applicants)
        df["feat_bureau_x_psychometric"] = np.where(
            df["has_bureau_record"] == 1,
            (df["bureau_score_avg"] / 850) * df["psychometric_composite"],
            df["psychometric_composite"] * 0.8,  # Rely more on psychometric if no bureau
        )

        # Alt-data strength × formal data weakness
        # (Higher alt score + weaker bureau = alt data is compensating)
        df["feat_alt_data_compensation"] = (
            df["feat_digital_presence"]
            * (1 - np.where(df["has_bureau_record"] == 1, df["bureau_score_avg"] / 850, 0))
        )

        # Revenue × business age (mature businesses with high revenue = strongest)
        df["feat_revenue_maturity"] = (
            np.log1p(df["avg_monthly_revenue_6m"])
            * df["feat_business_maturity"]
        )

        # Social engagement × e-commerce (cross-channel business strength)
        df["feat_cross_channel_strength"] = (
            df["social_engagement_score"]
            * np.where(df["has_ecommerce_presence"] == 1, df["ecommerce_avg_rating"] / 5, 0.3)
        )

        return df

    def _ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derived ratio features for gradient boosting performance."""

        # Loan amount to business age ratio
        df["feat_loan_age_ratio"] = (
            df["requested_loan_amount_ngn"]
            / (df["business_age_months"] + 1)
        )

        # Monthly installment estimate (simple)
        df["feat_est_monthly_installment"] = (
            df["requested_loan_amount_ngn"]
            / (df["requested_tenure_months"] + 1)
        )

        # Installment to revenue ratio
        df["feat_installment_to_revenue"] = (
            df["feat_est_monthly_installment"]
            / (df["avg_monthly_revenue_6m"] + 1)
        )

        # Bureau inquiries density (delinquencies per active loan)
        df["feat_delinquency_density"] = (
            df["num_delinquencies_12m"]
            / (df["num_active_loans"] + 1)
        )

        # Returning customer bonus
        df["feat_returning_bonus"] = np.where(
            df["is_returning_applicant"] == 1,
            df["prior_repayment_rate"] * (1 + df["prior_early_repayments"] * 0.1),
            0,
        )

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names."""
        return self.computed_feature_names
