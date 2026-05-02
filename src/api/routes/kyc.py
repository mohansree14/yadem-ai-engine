"""
YADEM KYC/KYB Verification Route
POST /api/v1/kyc — Verify Nigerian business and individual identity.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import hashlib
import random
import time
from loguru import logger

router = APIRouter()


class KYCRequest(BaseModel):
    bvn: Optional[str] = Field(None, description="Bank Verification Number (11 digits)")
    nin: Optional[str] = Field(None, description="National ID Number")
    cac_number: Optional[str] = Field(None, description="CAC Registration Number")
    business_name: Optional[str] = None
    director_name: Optional[str] = None


class KYCResponse(BaseModel):
    verified: bool
    bvn_valid: Optional[bool] = None
    nin_valid: Optional[bool] = None
    cac_valid: Optional[bool] = None
    identity_match_score: float = 0.0
    business_status: Optional[str] = None
    risk_flags: list = []
    processing_time_ms: float = 0.0


@router.post("/kyc", response_model=KYCResponse)
async def verify_kyc(request: KYCRequest):
    """
    Verify applicant identity and business registration.

    In production, this would call:
      - NIBSS BVN Validation API
      - NIMC NIN Verification API
      - CAC Public Search API

    For MVP, returns simulated verification results.
    """
    start = time.time()
    risk_flags = []

    # BVN validation (simulated)
    bvn_valid = None
    if request.bvn:
        if len(request.bvn) != 11 or not request.bvn.isdigit():
            risk_flags.append("INVALID_BVN_FORMAT")
            bvn_valid = False
        else:
            bvn_valid = True  # In production: call NIBSS API

    # NIN validation (simulated)
    nin_valid = None
    if request.nin:
        if len(request.nin) != 11 or not request.nin.isdigit():
            risk_flags.append("INVALID_NIN_FORMAT")
            nin_valid = False
        else:
            nin_valid = True

    # CAC validation (simulated)
    cac_valid = None
    business_status = None
    if request.cac_number:
        cac_valid = True  # In production: call CAC API
        business_status = "active"

    # Identity match score (simulated cross-reference)
    checks_passed = sum(1 for v in [bvn_valid, nin_valid, cac_valid] if v is True)
    total_checks = sum(1 for v in [bvn_valid, nin_valid, cac_valid] if v is not None)
    identity_match = checks_passed / total_checks if total_checks > 0 else 0.0

    verified = identity_match >= 0.5 and len(risk_flags) == 0

    elapsed = (time.time() - start) * 1000
    logger.info(f"KYC check: verified={verified} match={identity_match:.2f} time={elapsed:.0f}ms")

    return KYCResponse(
        verified=verified,
        bvn_valid=bvn_valid,
        nin_valid=nin_valid,
        cac_valid=cac_valid,
        identity_match_score=round(identity_match, 4),
        business_status=business_status,
        risk_flags=risk_flags,
        processing_time_ms=round(elapsed, 2),
    )
