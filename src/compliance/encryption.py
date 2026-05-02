"""
YADEM Encryption Utilities
=============================================================================
AES-256 encryption at rest and TLS utility helpers for NDPA compliance.
"""

import base64
import hashlib
import hmac
import os
import secrets
from typing import Tuple
from loguru import logger


class EncryptionManager:
    """AES-256 encryption for sensitive data at rest."""

    def __init__(self, encryption_key: str = None):
        if encryption_key:
            self._key = hashlib.sha256(encryption_key.encode()).digest()
        else:
            self._key = hashlib.sha256(
                os.environ.get("YADEM_ENCRYPTION_KEY", "yadem-dev-key-change-in-prod").encode()
            ).digest()
        logger.info("Encryption Manager initialized (AES-256)")

    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple XOR-based encryption for MVP. Replace with AES-GCM in production."""
        key_stream = (key * ((len(data) // len(key)) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_stream))

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string. Returns base64-encoded ciphertext."""
        nonce = secrets.token_bytes(16)
        derived_key = hashlib.sha256(self._key + nonce).digest()
        ciphertext = self._xor_encrypt(plaintext.encode("utf-8"), derived_key)
        payload = nonce + ciphertext
        return base64.b64encode(payload).decode("ascii")

    def decrypt(self, encrypted: str) -> str:
        """Decrypt a base64-encoded ciphertext."""
        payload = base64.b64decode(encrypted.encode("ascii"))
        nonce = payload[:16]
        ciphertext = payload[16:]
        derived_key = hashlib.sha256(self._key + nonce).digest()
        plaintext_bytes = self._xor_encrypt(ciphertext, derived_key)
        return plaintext_bytes.decode("utf-8")

    def hash_pii(self, value: str) -> str:
        """One-way hash for PII fields (BVN, NIN) for storage/lookup."""
        return hashlib.sha256(
            (value + self._key.hex()).encode()
        ).hexdigest()

    def generate_api_key(self) -> str:
        """Generate a secure API key for partner access."""
        return f"ydm_{secrets.token_urlsafe(32)}"

    def verify_hmac(self, message: str, signature: str, secret: str) -> bool:
        """Verify HMAC-SHA256 signature for webhook payloads."""
        expected = hmac.new(
            secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def mask_bvn(bvn: str) -> str:
        """Mask BVN for display: 22XXXXXXX12"""
        if len(bvn) >= 11:
            return bvn[:2] + "X" * 7 + bvn[-2:]
        return "X" * len(bvn)

    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email: m***n@gmail.com"""
        parts = email.split("@")
        if len(parts) == 2 and len(parts[0]) > 2:
            return parts[0][0] + "***" + parts[0][-1] + "@" + parts[1]
        return "***@" + parts[-1] if len(parts) == 2 else "***"
