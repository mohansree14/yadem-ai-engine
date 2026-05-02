"""
YADEM Open Banking Data Ingestion
=============================================================================
Ingests bank statement data via Open Banking APIs (Mono, Okra, Stitch).
For MVP: accepts pre-formatted JSON bank statement data.
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from loguru import logger


class OpenBankingIngester:
    """Ingests and normalizes bank statement data from multiple providers."""

    SUPPORTED_PROVIDERS = {"mono", "okra", "stitch", "manual"}

    # Standard transaction categories
    CATEGORIES = {
        "salary": ["salary", "payroll", "wage", "stipend"],
        "loan_repayment": ["loan", "repayment", "installment", "mortgage"],
        "utility": ["electricity", "phcn", "water", "dstv", "gotv", "internet"],
        "transfer_in": ["credit", "transfer", "received", "deposit"],
        "transfer_out": ["debit", "transfer", "sent", "withdrawal"],
        "pos": ["pos", "point of sale", "terminal"],
        "mobile_money": ["mobile", "momo", "opay", "palmpay", "kuda"],
    }

    def __init__(self, provider: str = "manual"):
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Provider must be one of {self.SUPPORTED_PROVIDERS}")
        self.provider = provider
        logger.info(f"Open Banking Ingester initialized (provider: {provider})")

    def ingest(self, raw_data: Dict) -> pd.DataFrame:
        """
        Ingest bank statement data and return normalized DataFrame.

        Expected raw_data format:
        {
            "account_id": "...",
            "bank_name": "...",
            "account_type": "savings|current",
            "currency": "NGN",
            "transactions": [
                {"date": "2025-01-15", "description": "...",
                 "amount": 50000.0, "type": "credit", "balance": 250000.0},
                ...
            ]
        }
        """
        transactions = raw_data.get("transactions", [])
        if not transactions:
            logger.warning("No transactions provided")
            return pd.DataFrame()

        df = pd.DataFrame(transactions)

        # Normalize columns
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        df["type"] = df["type"].str.lower().fillna("unknown")

        # Add metadata
        df["bank_name"] = raw_data.get("bank_name", "unknown")
        df["account_type"] = raw_data.get("account_type", "unknown")
        df["currency"] = raw_data.get("currency", "NGN")

        # Categorize transactions
        df["category"] = df["description"].apply(self._categorize)

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"Ingested {len(df)} transactions from {df['bank_name'].iloc[0]}")
        return df

    def _categorize(self, description: str) -> str:
        """Categorize a transaction based on description text."""
        if not isinstance(description, str):
            return "other"
        desc_lower = description.lower()
        for category, keywords in self.CATEGORIES.items():
            if any(kw in desc_lower for kw in keywords):
                return category
        return "other"

    def compute_summary(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics from bank statements."""
        if df.empty:
            return {}

        credits = df[df["type"] == "credit"]["amount"]
        debits = df[df["type"] == "debit"]["amount"]

        return {
            "total_credits": float(credits.sum()),
            "total_debits": float(debits.sum()),
            "avg_monthly_inflow": float(credits.sum() / max(1, df["date"].dt.to_period("M").nunique())),
            "avg_monthly_outflow": float(debits.sum() / max(1, df["date"].dt.to_period("M").nunique())),
            "transaction_count": len(df),
            "date_range_days": (df["date"].max() - df["date"].min()).days if len(df) > 1 else 0,
            "net_cashflow": float(credits.sum() - debits.sum()),
        }
