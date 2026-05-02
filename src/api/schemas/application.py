"""
YADEM API Request/Response Schemas
Pydantic models for the REST API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class FinancialData(BaseModel):
    """Financial data input for scoring."""
    avg_monthly_revenue_6m: float = Field(ge=0, description="Average monthly revenue over 6 months (NGN)")
    avg_monthly_revenue_3m: Optional[float] = None
    total_inflows_6m: Optional[float] = None
    total_outflows_6m: Optional[float] = None
    debit_to_credit_ratio_3m: Optional[float] = Field(None, ge=0, le=1)
    cashflow_volatility_6m: Optional[float] = None
    avg_monthly_balance: Optional[float] = None
    has_mobile_money: int = 0
    mobile_money_monthly_volume: float = 0
    has_pos_terminal: int = 0
    pos_monthly_transactions: int = 0
    pos_monthly_volume_ngn: float = 0


class BureauData(BaseModel):
    """Credit bureau data input."""
    has_bureau_record: int = 0
    bureau_score_crc: float = 0
    bureau_score_first_central: float = 0
    bureau_score_xds: float = 0
    num_active_loans: int = 0
    total_outstanding_debt_ngn: float = 0
    num_delinquencies_12m: int = 0
    worst_delinquency_days: int = 0
    has_active_default: int = 0
    num_on_time_payments_12m: int = 0


class AlternativeData(BaseModel):
    """Alternative data input."""
    has_ecommerce_presence: int = 0
    ecommerce_monthly_orders: int = 0
    ecommerce_avg_rating: float = 0
    has_social_media_business: int = 0
    social_engagement_score: float = 0
    utility_payment_consistency: float = 0.5
    num_missed_utility_payments_12m: int = 0
    sim_age_months: int = 12
    data_usage_consistency: float = 0.5


class PsychometricData(BaseModel):
    """Psychometric assessment data."""
    financial_discipline_score: float = Field(0.5, ge=0, le=1)
    risk_attitude_score: float = Field(0.5, ge=0, le=1)
    repayment_willingness_score: float = Field(0.5, ge=0, le=1)
    financial_literacy_score: float = Field(0.5, ge=0, le=1)
    planning_horizon_score: float = Field(0.5, ge=0, le=1)


class ScoreRequest(BaseModel):
    """Complete scoring request."""
    # Identity
    applicant_id: Optional[str] = None
    bvn: str = Field(..., min_length=11, max_length=11, description="Bank Verification Number")

    # Business info
    business_sector: str = "retail_fmcg"
    business_state: str = "Lagos"
    business_age_months: int = Field(ge=0)
    num_employees: int = Field(ge=1, default=1)
    is_registered_cac: int = Field(0, ge=0, le=1)

    # Owner info
    owner_age: int = Field(ge=18, le=80, default=35)
    owner_gender: str = "M"
    owner_education: str = "tertiary"

    # KYC
    bvn_verified: int = 1
    nin_verified: int = 0
    cac_verified: int = 0

    # Loan request
    requested_loan_amount_ngn: float = Field(gt=0)
    requested_tenure_months: int = Field(ge=1, le=36, default=12)

    # Data categories
    financial: FinancialData
    bureau: Optional[BureauData] = None
    alternative: Optional[AlternativeData] = None
    psychometric: Optional[PsychometricData] = None

    # Fraud check inputs
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None

    # Consent
    consent_token: Optional[str] = None


class KYCRequest(BaseModel):
    """KYC verification request."""
    bvn: str = Field(..., min_length=11, max_length=11)
    nin: Optional[str] = Field(None, min_length=11, max_length=11)
    cac_number: Optional[str] = None


class FraudCheckRequest(BaseModel):
    """Fraud check request."""
    bvn: str = Field(..., min_length=11, max_length=11)
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None
    cac_number: Optional[str] = None
    application_amount: float = 0
    business_state: Optional[str] = None


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class ExplanationFactor(BaseModel):
    """Single explanation factor."""
    feature: str
    shap_value: float
    direction: str
    description: str


class ExplanationResponse(BaseModel):
    """SHAP explanation response."""
    top_risk_factors: List[ExplanationFactor]
    top_strengths: List[ExplanationFactor]
    explanation_text: str


class FraudCheckResponse(BaseModel):
    """Fraud check response."""
    passed: bool
    checks: Dict[str, bool]
    flags: List[str]
    risk_level: str


class ScoreResponse(BaseModel):
    """Complete scoring response."""
    applicant_id: str
    yadem_score: int = Field(ge=0, le=1000)
    probability_of_default: float
    risk_band: str
    risk_band_meaning: str
    decision: str
    max_tenure_months: int
    rate_multiplier: float
    individual_model_scores: Dict[str, float]
    decision_rules_applied: List[str]
    fraud_check: FraudCheckResponse
    explanation: Optional[ExplanationResponse] = None
    processing_time_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class KYCResponse(BaseModel):
    """KYC verification response."""
    verified: bool
    bvn_verified: bool
    nin_verified: bool
    identity_match_score: float
    business_status: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    version: str
    uptime_seconds: float
