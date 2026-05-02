"""
YADEM Fraud Check Route
POST /api/v1/fraud-check — Standalone fraud screening endpoint.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, Dict
import time
from loguru import logger

from src.fraud.screener import FraudScreener

router = APIRouter()
_screener = FraudScreener()


class FraudCheckRequest(BaseModel):
    bvn: str = Field(..., description="Bank Verification Number")
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None
    application_amount: float = 0.0
    applicant_name: Optional[str] = None
    cac_number: Optional[str] = None
    business_state: Optional[str] = None


class FraudCheckResponse(BaseModel):
    passed: bool
    risk_level: str
    checks: Dict
    flags: list
    processing_time_ms: float


@router.post("/fraud-check", response_model=FraudCheckResponse)
async def check_fraud(request: FraudCheckRequest):
    """Run standalone fraud screening outside of the scoring pipeline."""
    start = time.time()

    result = _screener.screen(
        bvn=request.bvn,
        device_fingerprint=request.device_fingerprint,
        ip_address=request.ip_address,
        cac_number=request.cac_number,
        application_amount=request.application_amount,
        business_state=request.business_state,
    )

    elapsed = (time.time() - start) * 1000

    logger.info(f"Fraud check: passed={result.passed} risk={result.risk_level}")

    return FraudCheckResponse(
        passed=result.passed,
        risk_level=result.risk_level,
        checks=result.checks,
        flags=result.flags,
        processing_time_ms=round(elapsed, 2),
    )
