"""
YADEM Alternative Data Ingestion
=============================================================================
Ingests non-traditional data sources: mobile money, POS, e-commerce.
"""

from typing import Dict, List
from loguru import logger


class AltDataIngester:
    """Ingests alternative data for thin-file SMEs."""

    def __init__(self):
        logger.info("Alt Data Ingester initialized")

    def ingest_mobile_money(self, data: Dict) -> Dict:
        """Normalize mobile money transaction data (OPay, PalmPay, Kuda)."""
        return {
            "mm_transaction_count": data.get("transaction_count", 0),
            "mm_total_volume": data.get("total_volume", 0),
            "mm_avg_transaction": data.get("avg_transaction", 0),
            "mm_unique_recipients": data.get("unique_recipients", 0),
            "mm_consistency_score": data.get("consistency_score", 0.5),
            "mm_months_active": data.get("months_active", 0),
        }

    def ingest_pos_data(self, data: Dict) -> Dict:
        """Normalize POS terminal data."""
        return {
            "pos_daily_average": data.get("daily_average", 0),
            "pos_peak_day_volume": data.get("peak_day_volume", 0),
            "pos_active_days_pct": data.get("active_days_pct", 0),
            "pos_avg_ticket_size": data.get("avg_ticket_size", 0),
            "pos_terminal_count": data.get("terminal_count", 0),
        }

    def ingest_ecommerce(self, data: Dict) -> Dict:
        """Normalize e-commerce platform data (Jumia, Konga ratings)."""
        return {
            "ecom_seller_rating": data.get("seller_rating", 0),
            "ecom_total_orders": data.get("total_orders", 0),
            "ecom_return_rate": data.get("return_rate", 0),
            "ecom_months_selling": data.get("months_selling", 0),
            "ecom_revenue_trend": data.get("revenue_trend", "stable"),
        }

    def ingest_all(self, mobile_money: Dict = None, pos: Dict = None,
                    ecommerce: Dict = None) -> Dict:
        """Ingest all available alternative data sources."""
        combined = {}
        if mobile_money:
            combined.update(self.ingest_mobile_money(mobile_money))
        if pos:
            combined.update(self.ingest_pos_data(pos))
        if ecommerce:
            combined.update(self.ingest_ecommerce(ecommerce))
        combined["alt_data_sources_count"] = sum(
            1 for d in [mobile_money, pos, ecommerce] if d
        )
        logger.info(f"Alt data ingested from {combined['alt_data_sources_count']} sources")
        return combined
