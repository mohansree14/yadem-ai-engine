"""
YADEM Fraud Screening Module
=============================================================================
Runs parallel fraud checks alongside credit scoring. A fraud flag can
override a favourable credit score. Three components:
  1. Karma Blacklist - shared industry registry of known fraudsters
  2. Device Fingerprinting - detect repeat/farm applications
  3. Velocity Checks - unusual application patterns
"""

import hashlib
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger


@dataclass
class FraudResult:
    """Result of fraud screening."""
    passed: bool
    checks: Dict[str, bool]
    flags: List[str]
    risk_level: str  # "low", "medium", "high", "critical"
    details: Dict[str, Any]


class FraudScreener:
    """
    Fraud detection engine with Karma Blacklist, Device Fingerprinting,
    and Velocity Checks. Runs in parallel with credit scoring.
    """

    def __init__(self):
        # In-memory stores (would be Redis/DB in production)
        self._blacklist_bvns: Set[str] = set()
        self._blacklist_cac: Set[str] = set()
        self._blacklist_devices: Set[str] = set()
        self._application_log: Dict[str, List[datetime]] = defaultdict(list)
        self._device_log: Dict[str, List[Dict]] = defaultdict(list)

    def screen(
        self,
        bvn: str,
        device_fingerprint: Optional[str] = None,
        ip_address: Optional[str] = None,
        cac_number: Optional[str] = None,
        application_amount: float = 0,
        business_state: Optional[str] = None,
    ) -> FraudResult:
        """
        Run all fraud checks for an applicant.
        
        Returns:
            FraudResult with pass/fail and detailed flags.
        """
        flags = []
        checks = {}
        details = {}

        # Check 1: Karma Blacklist
        blacklist_result = self._check_blacklist(bvn, cac_number, device_fingerprint)
        checks["blacklist"] = blacklist_result["passed"]
        if not blacklist_result["passed"]:
            flags.extend(blacklist_result["flags"])

        # Check 2: Device Fingerprinting
        if device_fingerprint:
            device_result = self._check_device(device_fingerprint, bvn)
            checks["device"] = device_result["passed"]
            if not device_result["passed"]:
                flags.extend(device_result["flags"])
        else:
            checks["device"] = True

        # Check 3: Velocity Checks
        velocity_result = self._check_velocity(
            bvn, application_amount, ip_address, business_state
        )
        checks["velocity"] = velocity_result["passed"]
        if not velocity_result["passed"]:
            flags.extend(velocity_result["flags"])

        # Determine overall result
        passed = all(checks.values())
        if not checks["blacklist"]:
            risk_level = "critical"
        elif not checks["device"] or not checks["velocity"]:
            risk_level = "high"
        else:
            risk_level = "low"

        # Log the application
        self._application_log[bvn].append(datetime.utcnow())
        if device_fingerprint:
            self._device_log[device_fingerprint].append({
                "bvn": bvn, "timestamp": datetime.utcnow(),
                "amount": application_amount,
            })

        return FraudResult(
            passed=passed,
            checks=checks,
            flags=flags,
            risk_level=risk_level,
            details=details,
        )

    def _check_blacklist(
        self, bvn: str, cac: Optional[str], device: Optional[str]
    ) -> Dict[str, Any]:
        """Check against Karma Blacklist registry."""
        flags = []
        if bvn in self._blacklist_bvns:
            flags.append(f"BLACKLIST: BVN {bvn[:4]}**** found in Karma registry")
        if cac and cac in self._blacklist_cac:
            flags.append(f"BLACKLIST: CAC {cac} found in Karma registry")
        if device and device in self._blacklist_devices:
            flags.append("BLACKLIST: Device previously flagged for fraud")

        return {"passed": len(flags) == 0, "flags": flags}

    def _check_device(
        self, device_fingerprint: str, current_bvn: str
    ) -> Dict[str, Any]:
        """Check device fingerprint for repeat/farm applications."""
        flags = []
        history = self._device_log.get(device_fingerprint, [])

        # Check for multiple BVNs from same device
        unique_bvns = set(h["bvn"] for h in history)
        unique_bvns.add(current_bvn)
        if len(unique_bvns) > 2:
            flags.append(
                f"DEVICE: {len(unique_bvns)} different identities "
                f"from same device"
            )

        # Check for rapid succession (application farm)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        recent = [h for h in history if h["timestamp"] > cutoff]
        if len(recent) >= 5:
            flags.append(
                f"DEVICE_FARM: {len(recent)} applications from "
                f"this device in past hour"
            )

        return {"passed": len(flags) == 0, "flags": flags}

    def _check_velocity(
        self,
        bvn: str,
        amount: float,
        ip_address: Optional[str],
        business_state: Optional[str],
    ) -> Dict[str, Any]:
        """Check for unusual application velocity patterns."""
        flags = []
        now = datetime.utcnow()
        history = self._application_log.get(bvn, [])

        # Multiple applications in 24 hours
        cutoff_24h = now - timedelta(hours=24)
        recent_24h = [t for t in history if t > cutoff_24h]
        if len(recent_24h) >= 3:
            flags.append(
                f"VELOCITY: {len(recent_24h)} applications from "
                f"BVN in past 24 hours"
            )

        return {"passed": len(flags) == 0, "flags": flags}

    # --- Admin methods ---
    def add_to_blacklist(self, bvn: Optional[str] = None,
                         cac: Optional[str] = None,
                         device: Optional[str] = None) -> None:
        """Add entries to the Karma Blacklist."""
        if bvn:
            self._blacklist_bvns.add(bvn)
        if cac:
            self._blacklist_cac.add(cac)
        if device:
            self._blacklist_devices.add(device)
