"""
YADEM KYC/KYB Verification Service
=============================================================================
Verifies identity via BVN, NIN, and business via CAC number.
For MVP: validation logic only. Production: integrate NIBSS/NIMC/CAC APIs.
"""

from typing import Dict, Optional
from loguru import logger
import re


class KYCKYBVerifier:
    """Verifies Nigerian individual and business identity documents."""

    def __init__(self):
        logger.info("KYC/KYB Verifier initialized")

    def verify_bvn(self, bvn: str) -> Dict:
        """Validate BVN format and simulate NIBSS lookup."""
        is_valid_format = bool(re.match(r"^\d{11}$", bvn))
        return {
            "document_type": "BVN",
            "value": bvn[:4] + "***" + bvn[-2:] if len(bvn) >= 6 else "***",
            "format_valid": is_valid_format,
            "verified": is_valid_format,  # Production: NIBSS API call
            "provider": "NIBSS",
        }

    def verify_nin(self, nin: str) -> Dict:
        """Validate NIN format and simulate NIMC lookup."""
        is_valid_format = bool(re.match(r"^\d{11}$", nin))
        return {
            "document_type": "NIN",
            "value": nin[:4] + "***" + nin[-2:] if len(nin) >= 6 else "***",
            "format_valid": is_valid_format,
            "verified": is_valid_format,
            "provider": "NIMC",
        }

    def verify_cac(self, cac_number: str, business_name: Optional[str] = None) -> Dict:
        """Validate CAC registration."""
        is_valid_format = bool(re.match(r"^(RC|BN|IT)\d{4,10}$", cac_number, re.IGNORECASE))
        reg_type = "LLC" if cac_number.upper().startswith("RC") else \
                   "Business Name" if cac_number.upper().startswith("BN") else "IT"
        return {
            "document_type": "CAC",
            "cac_number": cac_number,
            "registration_type": reg_type,
            "format_valid": is_valid_format,
            "business_status": "active" if is_valid_format else "unknown",
            "verified": is_valid_format,
        }

    def full_verification(self, bvn: str = None, nin: str = None,
                          cac_number: str = None, business_name: str = None) -> Dict:
        """Run all available verifications and return composite result."""
        results = {}
        checks_passed = 0
        total_checks = 0

        if bvn:
            results["bvn"] = self.verify_bvn(bvn)
            total_checks += 1
            if results["bvn"]["verified"]:
                checks_passed += 1

        if nin:
            results["nin"] = self.verify_nin(nin)
            total_checks += 1
            if results["nin"]["verified"]:
                checks_passed += 1

        if cac_number:
            results["cac"] = self.verify_cac(cac_number, business_name)
            total_checks += 1
            if results["cac"]["verified"]:
                checks_passed += 1

        match_score = checks_passed / total_checks if total_checks > 0 else 0.0

        return {
            "overall_verified": match_score >= 0.5,
            "identity_match_score": round(match_score, 4),
            "checks_passed": checks_passed,
            "total_checks": total_checks,
            "details": results,
        }
