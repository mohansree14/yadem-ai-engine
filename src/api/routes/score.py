"""
YADEM Scoring API Route
POST /api/v1/score — The main credit decisioning endpoint.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
import time
import uuid
import numpy as np
import pandas as pd

from src.api.schemas.application import (
    ScoreRequest, ScoreResponse, FraudCheckResponse,
    ExplanationResponse, ExplanationFactor,
)

router = APIRouter()


def get_engine():
    """Get the singleton engine instance."""
    from src.api.main import engine_state
    return engine_state


@router.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_applicant(request: ScoreRequest):
    """
    Score an SME loan application.
    
    Runs the complete YADEM pipeline:
    1. Data assembly & feature engineering
    2. Ensemble ML scoring (LR + RF + XGBoost)
    3. Score generation (0-1000) & risk band mapping
    4. Decision rules application
    5. SHAP explainability
    6. Fraud screening (parallel)
    
    Target: Complete in under 48 seconds.
    """
    start_time = time.time()
    state = get_engine()

    if not state["models_loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")

    applicant_id = request.applicant_id or f"YDM-{uuid.uuid4().hex[:8].upper()}"

    try:
        # Step 1: Assemble applicant data into DataFrame row
        row = _assemble_applicant_data(request)
        df = pd.DataFrame([row])

        # Step 2: Feature engineering
        from src.features.engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.compute_features(df)

        # Step 3: Preprocess features
        cleaner = state["cleaner"]
        X = cleaner.transform(df)

        # Step 4: Ensemble prediction
        ensemble = state["ensemble"]
        prob_default = float(ensemble.predict_proba(X)[0])
        individual_probs = {
            k: float(v[0])
            for k, v in ensemble.get_individual_predictions(X).items()
        }

        # Step 5: Fraud screening
        fraud_screener = state["fraud_screener"]
        fraud_result = fraud_screener.screen(
            bvn=request.bvn,
            device_fingerprint=request.device_fingerprint,
            ip_address=request.ip_address,
            cac_number=None,
            application_amount=request.requested_loan_amount_ngn,
            business_state=request.business_state,
        )

        # Step 6: Credit scoring & decision rules
        scorer = state["scorer"]
        decision = scorer.score(
            applicant_id=applicant_id,
            prob_default=prob_default,
            individual_probs=individual_probs,
            applicant_data=row,
            fraud_result={
                "passed": fraud_result.passed,
                "flags": fraud_result.flags,
            },
        )

        # Step 7: SHAP explanation
        explanation = None
        if state.get("explainer") and state["explainer"].is_fitted:
            try:
                expl = state["explainer"].explain(X, applicant_id, top_n=3)
                explanation = ExplanationResponse(
                    top_risk_factors=[
                        ExplanationFactor(
                            feature=f.feature_name,
                            shap_value=f.shap_value,
                            direction=f.direction,
                            description=f.human_readable,
                        )
                        for f in expl.top_positive_factors
                    ],
                    top_strengths=[
                        ExplanationFactor(
                            feature=f.feature_name,
                            shap_value=f.shap_value,
                            direction=f.direction,
                            description=f.human_readable,
                        )
                        for f in expl.top_negative_factors
                    ],
                    explanation_text=expl.explanation_text,
                )
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        processing_time = (time.time() - start_time) * 1000

        return ScoreResponse(
            applicant_id=applicant_id,
            yadem_score=decision.yadem_score,
            probability_of_default=decision.probability_of_default,
            risk_band=decision.risk_band,
            risk_band_meaning=decision.risk_band_meaning,
            decision=decision.decision,
            max_tenure_months=decision.max_tenure_months,
            rate_multiplier=decision.rate_multiplier,
            individual_model_scores=decision.individual_scores,
            decision_rules_applied=decision.decision_rules_applied,
            fraud_check=FraudCheckResponse(
                passed=fraud_result.passed,
                checks=fraud_result.checks,
                flags=fraud_result.flags,
                risk_level=fraud_result.risk_level,
            ),
            explanation=explanation,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _assemble_applicant_data(req: ScoreRequest) -> dict:
    """Convert API request into a flat dict matching training data schema."""
    row = {
        "applicant_id": req.applicant_id or "",
        "business_sector": req.business_sector,
        "business_state": req.business_state,
        "business_zone": _state_to_zone(req.business_state),
        "business_age_months": req.business_age_months,
        "num_employees": req.num_employees,
        "is_registered_cac": req.is_registered_cac,
        "owner_age": req.owner_age,
        "owner_gender": req.owner_gender,
        "owner_education": req.owner_education,
        "bvn_verified": req.bvn_verified,
        "nin_verified": req.nin_verified,
        "cac_verified": req.cac_verified,
        "requested_loan_amount_ngn": req.requested_loan_amount_ngn,
        "requested_tenure_months": req.requested_tenure_months,
    }

    # Financial data
    fin = req.financial
    row.update({
        "avg_monthly_revenue_6m": fin.avg_monthly_revenue_6m,
        "avg_monthly_revenue_3m": fin.avg_monthly_revenue_3m or fin.avg_monthly_revenue_6m,
        "total_inflows_6m": fin.total_inflows_6m or fin.avg_monthly_revenue_6m * 6,
        "total_outflows_6m": fin.total_outflows_6m or fin.avg_monthly_revenue_6m * 4,
        "debit_to_credit_ratio_3m": fin.debit_to_credit_ratio_3m or 0.5,
        "cashflow_volatility_6m": fin.cashflow_volatility_6m or 0.3,
        "avg_monthly_balance": fin.avg_monthly_balance or fin.avg_monthly_revenue_6m * 0.5,
        "min_monthly_balance_6m": (fin.avg_monthly_balance or fin.avg_monthly_revenue_6m * 0.5) * 0.2,
        "num_credit_transactions_6m": 40,
        "num_debit_transactions_6m": 55,
        "unique_counterparties": 15,
        "largest_single_inflow_ratio": 0.15,
        "has_mobile_money": fin.has_mobile_money,
        "mobile_money_monthly_volume": fin.mobile_money_monthly_volume,
        "has_pos_terminal": fin.has_pos_terminal,
        "pos_monthly_transactions": fin.pos_monthly_transactions,
        "pos_monthly_volume_ngn": fin.pos_monthly_volume_ngn,
    })

    # Bureau data
    bureau = req.bureau
    if bureau:
        row.update({
            "has_bureau_record": bureau.has_bureau_record,
            "bureau_score_crc": bureau.bureau_score_crc,
            "bureau_score_first_central": bureau.bureau_score_first_central,
            "bureau_score_xds": bureau.bureau_score_xds,
            "bureau_score_avg": (bureau.bureau_score_crc + bureau.bureau_score_first_central + bureau.bureau_score_xds) / 3 if bureau.has_bureau_record else 0,
            "bureau_score_trajectory": 0,
            "num_active_loans": bureau.num_active_loans,
            "total_outstanding_debt_ngn": bureau.total_outstanding_debt_ngn,
            "num_delinquencies_12m": bureau.num_delinquencies_12m,
            "worst_delinquency_days": bureau.worst_delinquency_days,
            "has_active_default": bureau.has_active_default,
            "num_on_time_payments_12m": bureau.num_on_time_payments_12m,
        })
    else:
        row.update({k: 0 for k in [
            "has_bureau_record", "bureau_score_crc", "bureau_score_first_central",
            "bureau_score_xds", "bureau_score_avg", "bureau_score_trajectory",
            "num_active_loans", "total_outstanding_debt_ngn",
            "num_delinquencies_12m", "worst_delinquency_days",
            "has_active_default", "num_on_time_payments_12m",
        ]})

    # Alternative data
    alt = req.alternative
    if alt:
        row.update({
            "has_ecommerce_presence": alt.has_ecommerce_presence,
            "ecommerce_monthly_orders": alt.ecommerce_monthly_orders,
            "ecommerce_avg_rating": alt.ecommerce_avg_rating,
            "ecommerce_return_rate": 0.05,
            "has_social_media_business": alt.has_social_media_business,
            "social_engagement_score": alt.social_engagement_score,
            "social_followers_count": 500,
            "utility_payment_consistency": alt.utility_payment_consistency,
            "num_missed_utility_payments_12m": alt.num_missed_utility_payments_12m,
            "avg_monthly_utility_spend_ngn": 15000,
            "sim_age_months": alt.sim_age_months,
            "avg_monthly_recharge_ngn": 5000,
            "data_usage_consistency": alt.data_usage_consistency,
            "num_unique_contacts_30d": 50,
        })
    else:
        row.update({
            "has_ecommerce_presence": 0, "ecommerce_monthly_orders": 0,
            "ecommerce_avg_rating": 0, "ecommerce_return_rate": 0,
            "has_social_media_business": 0, "social_engagement_score": 0,
            "social_followers_count": 0, "utility_payment_consistency": 0.5,
            "num_missed_utility_payments_12m": 0, "avg_monthly_utility_spend_ngn": 10000,
            "sim_age_months": 24, "avg_monthly_recharge_ngn": 3000,
            "data_usage_consistency": 0.5, "num_unique_contacts_30d": 30,
        })

    # Psychometric data
    psych = req.psychometric
    if psych:
        row.update({
            "financial_discipline_score": psych.financial_discipline_score,
            "risk_attitude_score": psych.risk_attitude_score,
            "repayment_willingness_score": psych.repayment_willingness_score,
            "financial_literacy_score": psych.financial_literacy_score,
            "planning_horizon_score": psych.planning_horizon_score,
            "psychometric_composite": (
                psych.financial_discipline_score * 0.30 +
                (1 - psych.risk_attitude_score) * 0.15 +
                psych.repayment_willingness_score * 0.30 +
                psych.financial_literacy_score * 0.15 +
                psych.planning_horizon_score * 0.10
            ),
        })
    else:
        row.update({
            "financial_discipline_score": 0.5, "risk_attitude_score": 0.5,
            "repayment_willingness_score": 0.5, "financial_literacy_score": 0.5,
            "planning_horizon_score": 0.5, "psychometric_composite": 0.5,
        })

    # KYC
    row["identity_match_score"] = 0.95 if req.bvn_verified and req.nin_verified else 0.6
    row["director_count"] = 1

    # Sector intelligence defaults
    from src.data.synthetic.generator import SECTORS, STATES
    sector_info = SECTORS.get(req.business_sector, {"risk_weight": 0.3, "avg_revenue_ngn": 2500000})
    row["revenue_vs_sector_avg"] = fin.avg_monthly_revenue_6m / (sector_info["avg_revenue_ngn"] + 1)
    row["industry_revenue_percentile"] = 0.5
    row["sector_risk_weight"] = sector_info["risk_weight"]

    zone = _state_to_zone(req.business_state)
    zone_risk = {"south_west": 0.8, "south_east": 0.85, "south_south": 0.9,
                 "north_central": 0.9, "north_west": 1.1, "north_east": 1.3}
    row["geographic_risk_multiplier"] = zone_risk.get(zone, 0.9)
    row["market_cycle_indicator"] = 0.6
    row["seasonal_adjustment"] = 1.0

    # Loan history defaults
    row.update({
        "is_returning_applicant": 0, "prior_loans_count": 0,
        "prior_repayment_rate": 0, "prior_early_repayments": 0,
        "prior_defaults": 0, "days_since_last_loan": 0,
    })

    return row


def _state_to_zone(state: str) -> str:
    """Map Nigerian state to geo-political zone."""
    from src.data.synthetic.generator import STATES
    for zone, states in STATES.items():
        if state in states:
            return zone
    return "south_west"
