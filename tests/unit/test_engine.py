"""
YADEM Unit Tests — Core Engine Components
"""

import pytest
import numpy as np
import pandas as pd


class TestSyntheticDataGenerator:
    """Test the synthetic data generator."""

    def test_generate_default_samples(self):
        from src.data.synthetic.generator import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate(n_samples=100)
        assert len(df) == 100
        assert "default" in df.columns
        assert df["default"].isin([0, 1]).all()

    def test_generate_has_all_data_categories(self):
        from src.data.synthetic.generator import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate(n_samples=50)
        # Financial
        assert "avg_monthly_revenue_6m" in df.columns
        # Bureau
        assert "bureau_score_crc" in df.columns
        # Alt data
        assert "utility_payment_consistency" in df.columns
        # KYC
        assert "bvn_verified" in df.columns
        # Psychometric
        assert "psychometric_composite" in df.columns
        # Sector
        assert "sector_risk_weight" in df.columns
        # Loan history
        assert "prior_repayment_rate" in df.columns

    def test_default_rate_reasonable(self):
        from src.data.synthetic.generator import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate(n_samples=2000)
        rate = df["default"].mean()
        assert 0.03 < rate < 0.40, f"Default rate {rate:.2%} outside expected range"


class TestFeatureEngine:
    """Test feature engineering."""

    def test_compute_features(self):
        from src.data.synthetic.generator import SyntheticDataGenerator
        from src.features.engine import FeatureEngine
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate(n_samples=50)
        fe = FeatureEngine()
        df_feat = fe.compute_features(df)
        feat_cols = [c for c in df_feat.columns if c.startswith("feat_")]
        assert len(feat_cols) >= 20, f"Expected 20+ features, got {len(feat_cols)}"


class TestCreditScorer:
    """Test the credit scorer."""

    def test_probability_to_score(self):
        from src.scoring.scorer import CreditScorer
        scorer = CreditScorer()
        # Low default prob = high score
        assert scorer._probability_to_score(0.05) > 800
        # High default prob = low score
        assert scorer._probability_to_score(0.90) < 200
        # Mid range
        score_mid = scorer._probability_to_score(0.50)
        assert 300 < score_mid < 700

    def test_risk_band_assignment(self):
        from src.config.risk_config import RiskConfig, RiskBand
        config = RiskConfig()
        assert config.get_band(900).band == RiskBand.A
        assert config.get_band(700).band == RiskBand.B
        assert config.get_band(550).band == RiskBand.C
        assert config.get_band(400).band == RiskBand.D
        assert config.get_band(100).band == RiskBand.E


class TestFraudScreener:
    """Test fraud screening."""

    def test_clean_applicant_passes(self):
        from src.fraud.screener import FraudScreener
        screener = FraudScreener()
        result = screener.screen(bvn="12345678901")
        assert result.passed is True
        assert result.risk_level == "low"

    def test_blacklisted_bvn_fails(self):
        from src.fraud.screener import FraudScreener
        screener = FraudScreener()
        screener.add_to_blacklist(bvn="99999999999")
        result = screener.screen(bvn="99999999999")
        assert result.passed is False
        assert result.risk_level == "critical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
