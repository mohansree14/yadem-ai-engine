"""
YADEM Synthetic Data Generator
=============================================================================
Generates realistic synthetic Nigerian SME data covering all 7 data categories
as defined in the YADEM AI Engine architecture:
  1. Financial Data (bank transactions, mobile money, POS)
  2. Credit Bureau Data (CRC, First Central, XDS)
  3. Alternative Data (e-commerce, social media, utilities, telco)
  4. KYC/KYB Verification (BVN, NIN, CAC)
  5. Psychometric Data (risk attitudes, financial behaviour)
  6. Sector Intelligence (industry benchmarks)
  7. Loan History (prior YADEM performance)

This generates training data for the ensemble ML models before real data
integration is available.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from loguru import logger
import os


# Nigerian business sectors with risk profiles
SECTORS = {
    "retail_fmcg": {"risk_weight": 0.3, "avg_revenue_ngn": 2_500_000},
    "agriculture": {"risk_weight": 0.5, "avg_revenue_ngn": 1_800_000},
    "services": {"risk_weight": 0.25, "avg_revenue_ngn": 3_200_000},
    "light_manufacturing": {"risk_weight": 0.4, "avg_revenue_ngn": 4_500_000},
    "transport_logistics": {"risk_weight": 0.45, "avg_revenue_ngn": 3_800_000},
    "food_processing": {"risk_weight": 0.35, "avg_revenue_ngn": 2_200_000},
    "fashion_textiles": {"risk_weight": 0.3, "avg_revenue_ngn": 1_500_000},
    "technology": {"risk_weight": 0.2, "avg_revenue_ngn": 5_000_000},
    "healthcare": {"risk_weight": 0.15, "avg_revenue_ngn": 6_000_000},
    "education": {"risk_weight": 0.2, "avg_revenue_ngn": 2_800_000},
}

# Nigerian states grouped by economic zones
STATES = {
    "south_west": ["Lagos", "Ogun", "Oyo", "Osun", "Ondo", "Ekiti"],
    "south_east": ["Anambra", "Enugu", "Imo", "Abia", "Ebonyi"],
    "south_south": ["Rivers", "Delta", "Edo", "Cross River", "Akwa Ibom", "Bayelsa"],
    "north_central": ["Abuja", "Kwara", "Kogi", "Niger", "Benue", "Nassarawa", "Plateau"],
    "north_west": ["Kano", "Kaduna", "Sokoto", "Zamfara", "Kebbi", "Katsina", "Jigawa"],
    "north_east": ["Borno", "Adamawa", "Bauchi", "Gombe", "Yobe", "Taraba"],
}


class SyntheticDataGenerator:
    """
    Generates synthetic Nigerian SME applicant data that mirrors the
    distribution and characteristics described in the YADEM architecture.
    
    Default rate is calibrated at ~15-20% for the informal segment and
    ~5-8% for formal SMEs, matching African SME credit patterns.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def generate(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate a complete synthetic dataset with all 7 data categories.
        
        Args:
            n_samples: Number of SME applicant records to generate.
            
        Returns:
            DataFrame with 100+ features and a binary 'default' target column.
        """
        logger.info(f"Generating {n_samples} synthetic SME applicant records...")

        # --- 1. Base applicant identity ---
        df = self._generate_identity(n_samples)

        # --- 2. Financial data ---
        df = self._generate_financial_data(df)

        # --- 3. Credit bureau data ---
        df = self._generate_bureau_data(df)

        # --- 4. Alternative data ---
        df = self._generate_alt_data(df)

        # --- 5. KYC/KYB data ---
        df = self._generate_kyc_data(df)

        # --- 6. Psychometric data ---
        df = self._generate_psychometric_data(df)

        # --- 7. Sector intelligence ---
        df = self._generate_sector_data(df)

        # --- 8. Loan history ---
        df = self._generate_loan_history(df)

        # --- 9. Generate target variable (default) ---
        df = self._generate_target(df)

        logger.info(
            f"Generated dataset: {df.shape[0]} records, "
            f"{df.shape[1]} columns, "
            f"default rate: {df['default'].mean():.2%}"
        )
        return df

    def _generate_identity(self, n: int) -> pd.DataFrame:
        """Generate base applicant identity and business profile."""
        sectors = list(SECTORS.keys())
        all_states = [s for states in STATES.values() for s in states]
        zones = []
        for state in all_states:
            for zone, zone_states in STATES.items():
                if state in zone_states:
                    zones.append(zone)
                    break

        state_indices = self.rng.integers(0, len(all_states), size=n)

        return pd.DataFrame({
            "applicant_id": [f"YDM-{i:06d}" for i in range(n)],
            "business_sector": self.rng.choice(sectors, size=n),
            "business_state": [all_states[i] for i in state_indices],
            "business_zone": [zones[state_indices[j]] for j in range(n)],
            "business_age_months": self.rng.integers(1, 180, size=n),
            "num_employees": np.clip(
                self.rng.lognormal(mean=1.5, sigma=1.2, size=n).astype(int),
                1, 200
            ),
            "is_registered_cac": self.rng.choice([0, 1], size=n, p=[0.35, 0.65]),
            "owner_age": self.rng.integers(22, 65, size=n),
            "owner_gender": self.rng.choice(["M", "F"], size=n, p=[0.60, 0.40]),
            "owner_education": self.rng.choice(
                ["primary", "secondary", "tertiary", "postgraduate"],
                size=n, p=[0.10, 0.30, 0.45, 0.15]
            ),
        })

    def _generate_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate bank transaction and financial data features."""
        n = len(df)
        sector_revenues = df["business_sector"].map(
            {k: v["avg_revenue_ngn"] for k, v in SECTORS.items()}
        ).values

        # Monthly revenue with sector-based variation
        base_revenue = self.rng.lognormal(
            mean=np.log(sector_revenues), sigma=0.6
        )
        df["avg_monthly_revenue_6m"] = np.clip(base_revenue, 20_000, 50_000_000)
        df["avg_monthly_revenue_3m"] = df["avg_monthly_revenue_6m"] * (
            1 + self.rng.normal(0, 0.15, size=n)
        )

        # Cash flow metrics
        df["total_inflows_6m"] = df["avg_monthly_revenue_6m"] * 6 * (
            1 + self.rng.uniform(-0.1, 0.2, size=n)
        )
        df["total_outflows_6m"] = df["total_inflows_6m"] * self.rng.uniform(
            0.5, 0.95, size=n
        )
        df["inflow_outflow_ratio"] = df["total_inflows_6m"] / (
            df["total_outflows_6m"] + 1
        )
        df["cashflow_volatility_6m"] = self.rng.exponential(0.3, size=n)
        df["avg_monthly_balance"] = df["avg_monthly_revenue_6m"] * self.rng.uniform(
            0.1, 2.0, size=n
        )
        df["min_monthly_balance_6m"] = df["avg_monthly_balance"] * self.rng.uniform(
            0.01, 0.8, size=n
        )

        # Debit-to-credit ratio (key risk signal)
        df["debit_to_credit_ratio_3m"] = self.rng.beta(5, 3, size=n)

        # Transaction patterns
        df["num_credit_transactions_6m"] = self.rng.poisson(45, size=n)
        df["num_debit_transactions_6m"] = self.rng.poisson(60, size=n)
        df["unique_counterparties"] = self.rng.poisson(15, size=n) + 1
        df["largest_single_inflow_ratio"] = self.rng.beta(2, 8, size=n)

        # Mobile money data
        df["has_mobile_money"] = self.rng.choice([0, 1], size=n, p=[0.25, 0.75])
        df["mobile_money_monthly_volume"] = np.where(
            df["has_mobile_money"] == 1,
            self.rng.lognormal(12, 1, size=n),
            0
        )

        # POS data
        df["has_pos_terminal"] = self.rng.choice([0, 1], size=n, p=[0.40, 0.60])
        df["pos_monthly_transactions"] = np.where(
            df["has_pos_terminal"] == 1,
            self.rng.poisson(200, size=n),
            0
        )
        df["pos_monthly_volume_ngn"] = np.where(
            df["has_pos_terminal"] == 1,
            self.rng.lognormal(13, 0.8, size=n),
            0
        )

        return df

    def _generate_bureau_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate credit bureau data (CRC, First Central, XDS)."""
        n = len(df)

        # Bureau score (0-999 range used by Nigerian bureaus)
        df["has_bureau_record"] = self.rng.choice(
            [0, 1], size=n, p=[0.30, 0.70]
        )
        df["bureau_score_crc"] = np.where(
            df["has_bureau_record"] == 1,
            self.rng.normal(550, 120, size=n).clip(200, 850),
            0
        )
        df["bureau_score_first_central"] = np.where(
            df["has_bureau_record"] == 1,
            df["bureau_score_crc"] + self.rng.normal(0, 30, size=n),
            0
        ).clip(0, 850)
        df["bureau_score_xds"] = np.where(
            df["has_bureau_record"] == 1,
            df["bureau_score_crc"] + self.rng.normal(0, 25, size=n),
            0
        ).clip(0, 850)

        # Composite bureau score
        df["bureau_score_avg"] = np.where(
            df["has_bureau_record"] == 1,
            (df["bureau_score_crc"] + df["bureau_score_first_central"] +
             df["bureau_score_xds"]) / 3,
            0
        )

        # Bureau trajectory (improving=1, stable=0, deteriorating=-1)
        df["bureau_score_trajectory"] = np.where(
            df["has_bureau_record"] == 1,
            self.rng.choice([-1, 0, 1], size=n, p=[0.20, 0.45, 0.35]),
            0
        )

        # Credit history details
        df["num_active_loans"] = np.where(
            df["has_bureau_record"] == 1,
            self.rng.poisson(1.2, size=n),
            0
        )
        df["total_outstanding_debt_ngn"] = np.where(
            df["num_active_loans"] > 0,
            self.rng.lognormal(13, 1.5, size=n),
            0
        )
        df["num_delinquencies_12m"] = np.where(
            df["has_bureau_record"] == 1,
            self.rng.poisson(0.5, size=n),
            0
        )
        df["worst_delinquency_days"] = np.where(
            df["num_delinquencies_12m"] > 0,
            self.rng.choice([30, 60, 90, 120, 180], size=n, p=[0.4, 0.25, 0.2, 0.1, 0.05]),
            0
        )
        df["has_active_default"] = np.where(
            df["worst_delinquency_days"] >= 90, 1, 0
        )

        # Positive credit reporting
        df["num_on_time_payments_12m"] = np.where(
            df["has_bureau_record"] == 1,
            self.rng.poisson(10, size=n),
            0
        )

        return df

    def _generate_alt_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate alternative data features (e-commerce, social, telco, utilities)."""
        n = len(df)

        # E-commerce presence
        df["has_ecommerce_presence"] = self.rng.choice(
            [0, 1], size=n, p=[0.55, 0.45]
        )
        df["ecommerce_monthly_orders"] = np.where(
            df["has_ecommerce_presence"] == 1,
            self.rng.poisson(25, size=n),
            0
        )
        df["ecommerce_avg_rating"] = np.where(
            df["has_ecommerce_presence"] == 1,
            self.rng.beta(8, 2, size=n) * 5,
            0
        )
        df["ecommerce_return_rate"] = np.where(
            df["has_ecommerce_presence"] == 1,
            self.rng.beta(2, 20, size=n),
            0
        )

        # Social media business signals
        df["has_social_media_business"] = self.rng.choice(
            [0, 1], size=n, p=[0.30, 0.70]
        )
        df["social_engagement_score"] = np.where(
            df["has_social_media_business"] == 1,
            self.rng.beta(3, 2, size=n),
            0
        )
        df["social_followers_count"] = np.where(
            df["has_social_media_business"] == 1,
            self.rng.lognormal(6, 1.5, size=n).astype(int),
            0
        )

        # Utility bill payments
        df["utility_payment_consistency"] = self.rng.beta(5, 2, size=n)
        df["num_missed_utility_payments_12m"] = self.rng.poisson(1.5, size=n)
        df["avg_monthly_utility_spend_ngn"] = self.rng.lognormal(
            9.5, 0.8, size=n
        )

        # Telco metadata
        df["sim_age_months"] = self.rng.integers(6, 120, size=n)
        df["avg_monthly_recharge_ngn"] = self.rng.lognormal(8, 0.8, size=n)
        df["data_usage_consistency"] = self.rng.beta(4, 2, size=n)
        df["num_unique_contacts_30d"] = self.rng.poisson(50, size=n) + 5

        return df

    def _generate_kyc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate KYC/KYB verification data."""
        n = len(df)

        df["bvn_verified"] = self.rng.choice([0, 1], size=n, p=[0.05, 0.95])
        df["nin_verified"] = self.rng.choice([0, 1], size=n, p=[0.10, 0.90])
        df["cac_verified"] = np.where(
            df["is_registered_cac"] == 1,
            self.rng.choice([0, 1], size=n, p=[0.05, 0.95]),
            0
        )
        df["identity_match_score"] = np.where(
            (df["bvn_verified"] == 1) & (df["nin_verified"] == 1),
            self.rng.beta(15, 1, size=n),
            self.rng.beta(5, 3, size=n)
        )
        df["director_count"] = np.where(
            df["is_registered_cac"] == 1,
            self.rng.integers(1, 5, size=n),
            1
        )

        return df

    def _generate_psychometric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate psychometric assessment scores.
        Evaluates financial behaviour and risk attitudes of beneficial owners.
        """
        n = len(df)

        # Financial discipline index (0-1)
        df["financial_discipline_score"] = self.rng.beta(5, 3, size=n)

        # Risk attitude (0=risk-averse, 1=risk-seeking)
        df["risk_attitude_score"] = self.rng.beta(3, 4, size=n)

        # Willingness to repay (psychometric proxy, 0-1)
        df["repayment_willingness_score"] = self.rng.beta(6, 2, size=n)

        # Financial literacy (0-1)
        df["financial_literacy_score"] = self.rng.beta(4, 3, size=n)

        # Planning horizon (short=0, long=1)
        df["planning_horizon_score"] = self.rng.beta(3, 3, size=n)

        # Composite psychometric score
        df["psychometric_composite"] = (
            df["financial_discipline_score"] * 0.30 +
            (1 - df["risk_attitude_score"]) * 0.15 +
            df["repayment_willingness_score"] * 0.30 +
            df["financial_literacy_score"] * 0.15 +
            df["planning_horizon_score"] * 0.10
        )

        return df

    def _generate_sector_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate sector intelligence features."""
        n = len(df)

        # Industry-adjusted revenue percentile
        sector_avgs = df["business_sector"].map(
            {k: v["avg_revenue_ngn"] for k, v in SECTORS.items()}
        ).values
        df["revenue_vs_sector_avg"] = df["avg_monthly_revenue_6m"] / (
            sector_avgs + 1
        )
        df["industry_revenue_percentile"] = self.rng.beta(5, 5, size=n)

        # Sector risk
        df["sector_risk_weight"] = df["business_sector"].map(
            {k: v["risk_weight"] for k, v in SECTORS.items()}
        )

        # Geographic risk multiplier
        zone_risk = {
            "south_west": 0.8, "south_east": 0.85, "south_south": 0.9,
            "north_central": 0.9, "north_west": 1.1, "north_east": 1.3,
        }
        df["geographic_risk_multiplier"] = df["business_zone"].map(zone_risk)

        # Market cycle indicator (0=downturn, 1=growth)
        df["market_cycle_indicator"] = self.rng.beta(4, 3, size=n)

        # Seasonal adjustment factor
        df["seasonal_adjustment"] = 1.0 + self.rng.normal(0, 0.1, size=n)

        return df

    def _generate_loan_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate prior loan history with YADEM."""
        n = len(df)

        df["is_returning_applicant"] = self.rng.choice(
            [0, 1], size=n, p=[0.75, 0.25]
        )
        df["prior_loans_count"] = np.where(
            df["is_returning_applicant"] == 1,
            self.rng.poisson(1.5, size=n) + 1,
            0
        )
        df["prior_repayment_rate"] = np.where(
            df["prior_loans_count"] > 0,
            self.rng.beta(8, 1.5, size=n),
            0
        )
        df["prior_early_repayments"] = np.where(
            df["prior_loans_count"] > 0,
            self.rng.binomial(df["prior_loans_count"], 0.3),
            0
        )
        df["prior_defaults"] = np.where(
            df["prior_loans_count"] > 0,
            self.rng.binomial(1, 0.08, size=n),
            0
        )
        df["days_since_last_loan"] = np.where(
            df["prior_loans_count"] > 0,
            self.rng.integers(30, 730, size=n),
            0
        )

        # Loan amount requested
        df["requested_loan_amount_ngn"] = (
            df["avg_monthly_revenue_6m"] * self.rng.uniform(0.5, 4.0, size=n)
        )
        df["requested_tenure_months"] = self.rng.choice(
            [3, 6, 9, 12, 18, 24], size=n
        )

        return df

    def _generate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate binary default target variable based on realistic risk factors.
        The default probability is a function of financial health, bureau score,
        psychometric scores, and sector risk — calibrated at ~12-18% overall.
        """
        n = len(df)

        # Build a latent risk score from multiple factors
        risk_score = np.zeros(n)

        # Financial signals (higher inflow ratio = lower risk)
        risk_score -= np.log1p(df["inflow_outflow_ratio"].values) * 0.8
        risk_score += df["debit_to_credit_ratio_3m"].values * 1.5
        risk_score += df["cashflow_volatility_6m"].values * 0.6

        # Bureau signals
        bureau_norm = df["bureau_score_avg"].values / 850
        risk_score -= np.where(df["has_bureau_record"] == 1, bureau_norm * 2.0, 0)
        risk_score += df["num_delinquencies_12m"].values * 0.5
        risk_score += df["has_active_default"].values * 3.0

        # Alt data signals
        risk_score -= df["utility_payment_consistency"].values * 0.5
        risk_score += df["num_missed_utility_payments_12m"].values * 0.3

        # Psychometric signals
        risk_score -= df["psychometric_composite"].values * 1.5

        # Sector & geographic risk
        risk_score += df["sector_risk_weight"].values * 0.8
        risk_score += df["geographic_risk_multiplier"].values * 0.3

        # Business maturity
        risk_score -= np.log1p(df["business_age_months"].values) * 0.3
        risk_score -= df["is_registered_cac"].values * 0.5

        # Loan history
        risk_score -= df["prior_repayment_rate"].values * 1.0
        risk_score += df["prior_defaults"].values * 2.0

        # KYC verification
        risk_score -= df["identity_match_score"].values * 0.3

        # Convert to probability via sigmoid
        prob_default = 1 / (1 + np.exp(-risk_score))

        # Calibrate overall default rate to ~15%
        prob_default = prob_default * 0.3  # Scale down
        prob_default = np.clip(prob_default, 0.01, 0.95)

        # Sample binary default
        df["default"] = (self.rng.uniform(size=n) < prob_default).astype(int)

        # Store the probability for analysis
        df["_prob_default"] = prob_default

        return df

    def save(
        self,
        df: pd.DataFrame,
        output_dir: str = "./data/synthetic",
        filename: str = "yadem_synthetic_data.csv",
    ) -> str:
        """Save generated data to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Synthetic data saved to {filepath}")
        return filepath

    def generate_and_save(
        self,
        n_samples: int = 5000,
        output_dir: str = "./data/synthetic",
    ) -> Tuple[pd.DataFrame, str]:
        """Generate and save synthetic data in one call."""
        df = self.generate(n_samples)
        filepath = self.save(df, output_dir)
        return df, filepath


if __name__ == "__main__":
    generator = SyntheticDataGenerator(seed=42)
    df, path = generator.generate_and_save(n_samples=5000)
    print(f"\nDataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"Saved to: {path}")
    print(f"\nColumn list ({len(df.columns)} columns):")
    for col in df.columns:
        print(f"  - {col}")
