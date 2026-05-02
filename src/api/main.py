"""
YADEM AI Credit Decisioning Engine — FastAPI Application
=============================================================================
Exposes the YADEM AI Engine as a B2B REST API.

Endpoints:
  POST /api/v1/score         — Score an SME loan application
  POST /api/v1/kyc           — KYC/KYB verification
  POST /api/v1/fraud-check   — Standalone fraud check
  GET  /api/v1/explain/{id}  — Retrieve explanation for a decision
  GET  /api/v1/health        — Health check and model status
"""

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import score, health, kyc, fraud, explain
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.config.settings import settings
from src.models.ensemble import EnsembleScorer
from src.scoring.scorer import CreditScorer
from src.fraud.screener import FraudScreener
from src.data.processing.cleaner import DataCleaner
from src.explainability.shap_explainer import SHAPExplainer
from src.config.risk_config import RiskConfig


# Global engine state
engine_state = {
    "ensemble": None,
    "cleaner": None,
    "scorer": None,
    "fraud_screener": None,
    "explainer": None,
    "models_loaded": False,
    "start_time": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("=" * 60)
    logger.info("YADEM AI Credit Decisioning Engine Starting...")
    logger.info("=" * 60)

    # Initialize components
    engine_state["scorer"] = CreditScorer(RiskConfig())
    engine_state["fraud_screener"] = FraudScreener()
    engine_state["explainer"] = SHAPExplainer()

    # Try to load pre-trained models
    model_dir = settings.model_registry_path
    if os.path.exists(os.path.join(model_dir, "ensemble_meta.joblib")):
        try:
            ensemble = EnsembleScorer()
            ensemble.load(model_dir)
            engine_state["ensemble"] = ensemble

            cleaner_path = os.path.join(model_dir, "data_cleaner.joblib")
            if os.path.exists(cleaner_path):
                import joblib
                engine_state["cleaner"] = joblib.load(cleaner_path)

            engine_state["models_loaded"] = True
            logger.info("✓ Pre-trained models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            logger.info("Run train.py first to train models")
    else:
        logger.info("No pre-trained models found. Run train.py first.")

    yield

    logger.info("YADEM Engine shutting down...")


# Create FastAPI app
app = FastAPI(
    title="YADEM AI Credit Decisioning Engine",
    description=(
        "AI-Powered Credit Infrastructure for African SMEs. "
        "Provides credit scoring, fraud detection, and explainable decisions "
        "for B2B lender integration."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting Middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

# Routes
app.include_router(score.router, prefix=settings.api_prefix)
app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(kyc.router, prefix=settings.api_prefix)
app.include_router(fraud.router, prefix=settings.api_prefix)
app.include_router(explain.router, prefix=settings.api_prefix)


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "YADEM AI Credit Decisioning Engine",
        "version": "1.0.0",
        "description": "Closing Africa's $150B+ SME Financing Gap",
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }
