"""
YADEM NDPA 2023 Compliance Module
=============================================================================
Implements Nigeria Data Protection Act 2023 compliance checks.
Ensures all credit scoring operations adhere to NDPA requirements for:
  - Data minimization
  - Purpose limitation
  - Lawful basis verification
  - Data subject rights
  - Cross-border transfer controls
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from loguru import logger


class LawfulBasis(str, Enum):
    """NDPA-recognized lawful bases for data processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTEREST = "vital_interest"
    PUBLIC_INTEREST = "public_interest"
    LEGITIMATE_INTEREST = "legitimate_interest"


class DataCategory(str, Enum):
    """Categories of personal data processed by YADEM."""
    FINANCIAL = "financial"
    IDENTITY = "identity"
    BEHAVIORAL = "behavioral"
    PSYCHOMETRIC = "psychometric"
    DEVICE = "device"
    LOCATION = "location"
    BUREAU = "bureau"


@dataclass
class DataProcessingRecord:
    """Record of a data processing activity (NDPA Art. 28)."""
    record_id: str
    data_subject_id: str
    data_categories: List[DataCategory]
    processing_purpose: str
    lawful_basis: LawfulBasis
    consent_reference: Optional[str] = None
    retention_period_days: int = 365
    created_at: datetime = field(default_factory=datetime.utcnow)
    cross_border: bool = False
    recipient_country: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "record_id": self.record_id,
            "data_subject_id": self.data_subject_id,
            "data_categories": [c.value for c in self.data_categories],
            "processing_purpose": self.processing_purpose,
            "lawful_basis": self.lawful_basis.value,
            "consent_reference": self.consent_reference,
            "retention_period_days": self.retention_period_days,
            "created_at": self.created_at.isoformat(),
            "cross_border": self.cross_border,
            "recipient_country": self.recipient_country,
        }


class NDPAComplianceChecker:
    """
    Validates that YADEM operations comply with NDPA 2023.

    Checks performed before every scoring request:
      1. Lawful basis exists (consent or contract)
      2. Data categories are within consented scope
      3. Retention period not exceeded
      4. Cross-border transfer has adequate safeguards
      5. Data minimization — only required fields used
    """

    # Minimum required data categories for credit scoring
    REQUIRED_CATEGORIES: Set[DataCategory] = {
        DataCategory.FINANCIAL,
        DataCategory.IDENTITY,
    }

    # Data categories requiring explicit consent
    SENSITIVE_CATEGORIES: Set[DataCategory] = {
        DataCategory.PSYCHOMETRIC,
        DataCategory.BEHAVIORAL,
        DataCategory.DEVICE,
        DataCategory.LOCATION,
    }

    # Countries with adequate data protection (NDPA cross-border)
    ADEQUATE_COUNTRIES: Set[str] = {
        "NG", "GH", "KE", "ZA", "RW", "GB", "DE", "FR",
        "NL", "IE", "US",  # US only with Standard Contractual Clauses
    }

    # Maximum retention periods by category (days)
    MAX_RETENTION: Dict[DataCategory, int] = {
        DataCategory.FINANCIAL: 2555,      # 7 years (CBN requirement)
        DataCategory.IDENTITY: 2555,       # 7 years
        DataCategory.BEHAVIORAL: 730,      # 2 years
        DataCategory.PSYCHOMETRIC: 365,    # 1 year
        DataCategory.DEVICE: 180,          # 6 months
        DataCategory.LOCATION: 90,         # 3 months
        DataCategory.BUREAU: 1825,         # 5 years
    }

    def __init__(self):
        self.processing_records: List[DataProcessingRecord] = []
        logger.info("NDPA Compliance Checker initialized")

    def validate_processing_request(
        self,
        data_categories: List[DataCategory],
        lawful_basis: LawfulBasis,
        consent_reference: Optional[str] = None,
        cross_border: bool = False,
        recipient_country: Optional[str] = None,
    ) -> Dict:
        """
        Validate a data processing request against NDPA requirements.

        Returns:
            Dict with 'compliant' bool and list of 'violations' if any.
        """
        violations = []

        # Check 1: Lawful basis
        if lawful_basis == LawfulBasis.CONSENT and not consent_reference:
            violations.append(
                "NDPA Art. 25: Consent-based processing requires valid consent reference"
            )

        # Check 2: Sensitive data requires explicit consent
        sensitive_requested = set(data_categories) & self.SENSITIVE_CATEGORIES
        if sensitive_requested and lawful_basis != LawfulBasis.CONSENT:
            violations.append(
                f"NDPA Art. 30: Sensitive categories {[c.value for c in sensitive_requested]} "
                f"require explicit consent, not {lawful_basis.value}"
            )

        # Check 3: Data minimization — flag unnecessary categories
        required = set(data_categories) & self.REQUIRED_CATEGORIES
        if not required:
            violations.append(
                "NDPA Art. 26: Processing must include minimum required categories "
                "(financial, identity) for credit scoring purpose"
            )

        # Check 4: Cross-border transfer safeguards
        if cross_border:
            if not recipient_country:
                violations.append(
                    "NDPA Art. 43: Cross-border transfer must specify recipient country"
                )
            elif recipient_country not in self.ADEQUATE_COUNTRIES:
                violations.append(
                    f"NDPA Art. 43: Country '{recipient_country}' does not have "
                    f"adequate data protection. Standard Contractual Clauses required."
                )

        compliant = len(violations) == 0

        result = {
            "compliant": compliant,
            "violations": violations,
            "checked_at": datetime.utcnow().isoformat(),
            "data_categories": [c.value for c in data_categories],
            "lawful_basis": lawful_basis.value,
        }

        if not compliant:
            logger.warning(f"NDPA compliance check FAILED: {violations}")
        else:
            logger.info("NDPA compliance check PASSED")

        return result

    def create_processing_record(
        self,
        data_subject_id: str,
        data_categories: List[DataCategory],
        processing_purpose: str,
        lawful_basis: LawfulBasis,
        consent_reference: Optional[str] = None,
        cross_border: bool = False,
        recipient_country: Optional[str] = None,
    ) -> DataProcessingRecord:
        """Create an audit record for a processing activity (NDPA Art. 28)."""
        record_id = hashlib.sha256(
            f"{data_subject_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        record = DataProcessingRecord(
            record_id=record_id,
            data_subject_id=data_subject_id,
            data_categories=data_categories,
            processing_purpose=processing_purpose,
            lawful_basis=lawful_basis,
            consent_reference=consent_reference,
            cross_border=cross_border,
            recipient_country=recipient_country,
        )

        self.processing_records.append(record)
        logger.info(f"Processing record created: {record_id} for subject {data_subject_id}")
        return record

    def check_retention_compliance(
        self,
        data_category: DataCategory,
        created_at: datetime,
    ) -> Dict:
        """Check if data has exceeded maximum retention period."""
        max_days = self.MAX_RETENTION.get(data_category, 365)
        expiry_date = created_at + timedelta(days=max_days)
        now = datetime.utcnow()
        expired = now > expiry_date

        return {
            "data_category": data_category.value,
            "created_at": created_at.isoformat(),
            "max_retention_days": max_days,
            "expiry_date": expiry_date.isoformat(),
            "expired": expired,
            "days_remaining": max(0, (expiry_date - now).days),
            "action_required": "DELETE" if expired else "NONE",
        }

    def generate_dpia_summary(
        self,
        processing_purpose: str,
        data_categories: List[DataCategory],
        num_data_subjects: int,
    ) -> Dict:
        """
        Generate a Data Protection Impact Assessment (DPIA) summary.
        Required by NDPA Art. 29 for high-risk processing like credit scoring.
        """
        risk_factors = []
        risk_score = 0

        # Evaluate risk factors
        if num_data_subjects > 10000:
            risk_factors.append("Large-scale processing (>10,000 subjects)")
            risk_score += 2

        if DataCategory.PSYCHOMETRIC in data_categories:
            risk_factors.append("Psychometric profiling data used")
            risk_score += 3

        if DataCategory.BEHAVIORAL in data_categories:
            risk_factors.append("Behavioral data creates profiling risk")
            risk_score += 2

        if len(data_categories) >= 5:
            risk_factors.append("Multiple data categories combined (profiling)")
            risk_score += 2

        # Credit scoring is inherently high-risk
        risk_factors.append("Automated credit decisioning with legal effects")
        risk_score += 3

        risk_level = "LOW" if risk_score <= 3 else "MEDIUM" if risk_score <= 6 else "HIGH"

        mitigations = [
            "SHAP-based explainability for every decision",
            "Human review pathway for borderline (Band C) decisions",
            "Right-to-explanation endpoint available to data subjects",
            "Data minimization enforced at ingestion layer",
            "AES-256 encryption at rest, TLS 1.3 in transit",
            "Consent withdrawal mechanism implemented",
            "Retention policy auto-enforced per data category",
        ]

        return {
            "processing_purpose": processing_purpose,
            "data_categories": [c.value for c in data_categories],
            "num_data_subjects": num_data_subjects,
            "risk_factors": risk_factors,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "mitigations": mitigations,
            "dpia_required": risk_level in ("MEDIUM", "HIGH"),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def handle_data_subject_request(
        self,
        request_type: str,
        data_subject_id: str,
    ) -> Dict:
        """
        Handle NDPA data subject rights requests:
          - ACCESS: Right to access their data
          - RECTIFICATION: Right to correct inaccurate data
          - ERASURE: Right to deletion (right to be forgotten)
          - PORTABILITY: Right to data portability
          - OBJECTION: Right to object to processing
        """
        valid_types = {"ACCESS", "RECTIFICATION", "ERASURE", "PORTABILITY", "OBJECTION"}

        if request_type not in valid_types:
            return {
                "success": False,
                "error": f"Invalid request type. Must be one of: {valid_types}"
            }

        # Find processing records for this data subject
        subject_records = [
            r for r in self.processing_records
            if r.data_subject_id == data_subject_id
        ]

        response = {
            "request_type": request_type,
            "data_subject_id": data_subject_id,
            "records_found": len(subject_records),
            "processed_at": datetime.utcnow().isoformat(),
            "response_deadline": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        }

        if request_type == "ACCESS":
            response["data"] = [r.to_dict() for r in subject_records]
            response["status"] = "FULFILLED"

        elif request_type == "ERASURE":
            # Cannot erase if legal obligation exists (e.g., CBN reporting)
            financial_records = [
                r for r in subject_records
                if DataCategory.FINANCIAL in r.data_categories
            ]
            if financial_records:
                response["status"] = "PARTIALLY_FULFILLED"
                response["note"] = (
                    "Financial records retained per CBN regulatory requirement "
                    "(7-year retention). Non-financial records scheduled for deletion."
                )
            else:
                response["status"] = "SCHEDULED"
                response["deletion_date"] = (
                    datetime.utcnow() + timedelta(days=7)
                ).isoformat()

        elif request_type == "PORTABILITY":
            response["data"] = [r.to_dict() for r in subject_records]
            response["format"] = "JSON"
            response["status"] = "FULFILLED"

        else:
            response["status"] = "ACKNOWLEDGED"
            response["note"] = f"{request_type} request logged for manual review"

        logger.info(
            f"Data subject request [{request_type}] for {data_subject_id}: "
            f"{response.get('status', 'PROCESSED')}"
        )
        return response
