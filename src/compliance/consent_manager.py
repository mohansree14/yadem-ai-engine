"""
YADEM Consent Manager
=============================================================================
Manages consent lifecycle for data processing. Every credit scoring request
must have a valid consent token before personal data is accessed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import secrets
from loguru import logger


class ConsentType(str, Enum):
    CREDIT_CHECK = "credit_check"
    DATA_SHARING = "data_sharing"
    MARKETING = "marketing"
    PROFILING = "profiling"
    BUREAU_PULL = "bureau_pull"


class ConsentStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


@dataclass
class ConsentRecord:
    consent_id: str
    data_subject_id: str
    consent_type: ConsentType
    status: ConsentStatus = ConsentStatus.ACTIVE
    granted_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    purpose: str = "credit_scoring"
    channel: str = "api"

    def is_valid(self) -> bool:
        if self.status != ConsentStatus.ACTIVE:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict:
        return {
            "consent_id": self.consent_id,
            "data_subject_id": self.data_subject_id,
            "consent_type": self.consent_type.value,
            "status": self.status.value,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "withdrawn_at": self.withdrawn_at.isoformat() if self.withdrawn_at else None,
        }


class ConsentManager:
    """Manages consent tokens for YADEM data processing activities."""

    CONSENT_VALIDITY = {
        ConsentType.CREDIT_CHECK: 90,
        ConsentType.DATA_SHARING: 30,
        ConsentType.MARKETING: 365,
        ConsentType.PROFILING: 90,
        ConsentType.BUREAU_PULL: 30,
    }

    REQUIRED_FOR_SCORING: Set[ConsentType] = {
        ConsentType.CREDIT_CHECK,
        ConsentType.PROFILING,
    }

    def __init__(self):
        self._store: Dict[str, ConsentRecord] = {}
        self._subject_index: Dict[str, List[str]] = {}
        logger.info("Consent Manager initialized")

    def grant_consent(self, data_subject_id: str, consent_type: ConsentType,
                      purpose: str = "credit_scoring", validity_days: Optional[int] = None,
                      channel: str = "api") -> ConsentRecord:
        consent_id = f"cns_{secrets.token_hex(16)}"
        if validity_days is None:
            validity_days = self.CONSENT_VALIDITY.get(consent_type, 90)

        record = ConsentRecord(
            consent_id=consent_id, data_subject_id=data_subject_id,
            consent_type=consent_type, expires_at=datetime.utcnow() + timedelta(days=validity_days),
            purpose=purpose, channel=channel,
        )
        self._store[consent_id] = record
        self._subject_index.setdefault(data_subject_id, []).append(consent_id)
        logger.info(f"Consent granted: {consent_id} | type={consent_type.value}")
        return record

    def verify_consent(self, consent_id: str, required_type: Optional[ConsentType] = None) -> Dict:
        record = self._store.get(consent_id)
        if not record:
            return {"valid": False, "reason": "Consent token not found"}
        if record.expires_at and datetime.utcnow() > record.expires_at:
            record.status = ConsentStatus.EXPIRED
            return {"valid": False, "reason": "Consent has expired"}
        if record.status == ConsentStatus.WITHDRAWN:
            return {"valid": False, "reason": "Consent withdrawn"}
        if required_type and record.consent_type != required_type:
            return {"valid": False, "reason": f"Type mismatch: need {required_type.value}"}
        return {"valid": True, "consent_id": consent_id, "data_subject_id": record.data_subject_id}

    def verify_scoring_consent(self, data_subject_id: str) -> Dict:
        subject_consents = self._subject_index.get(data_subject_id, [])
        active_types = {self._store[cid].consent_type for cid in subject_consents
                        if cid in self._store and self._store[cid].is_valid()}
        missing = self.REQUIRED_FOR_SCORING - active_types
        if missing:
            return {"authorized": False, "missing_consents": [c.value for c in missing]}
        return {"authorized": True, "active_consents": [c.value for c in active_types]}

    def withdraw_consent(self, consent_id: str) -> Dict:
        record = self._store.get(consent_id)
        if not record:
            return {"success": False, "error": "Not found"}
        record.status = ConsentStatus.WITHDRAWN
        record.withdrawn_at = datetime.utcnow()
        logger.info(f"Consent withdrawn: {consent_id}")
        return {"success": True, "consent_id": consent_id, "withdrawn_at": record.withdrawn_at.isoformat()}
