"""
YADEM Explanation Route
GET /api/v1/explain/{application_id} — Retrieve SHAP explanation for a decision.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from loguru import logger

router = APIRouter()

# In-memory explanation cache (in production: database-backed)
_explanation_cache: Dict[str, Dict] = {}


class ExplanationResponse(BaseModel):
    application_id: str
    yadem_score: Optional[int] = None
    risk_band: Optional[str] = None
    top_factors: List[Dict] = []
    shap_values: Dict = {}
    human_readable: str = ""
    counterfactual: Optional[Dict] = None


def store_explanation(application_id: str, explanation_data: Dict):
    """Store an explanation for later retrieval."""
    _explanation_cache[application_id] = explanation_data
    logger.debug(f"Stored explanation for application {application_id}")


@router.get("/explain/{application_id}", response_model=ExplanationResponse)
async def get_explanation(application_id: str):
    """
    Retrieve the SHAP-based explanation for a scored application.
    This endpoint fulfills the NDPA right-to-explanation requirement.
    """
    data = _explanation_cache.get(application_id)

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No explanation found for application {application_id}. "
                   f"Score the application first via POST /api/v1/score."
        )

    return ExplanationResponse(
        application_id=application_id,
        yadem_score=data.get("yadem_score"),
        risk_band=data.get("risk_band"),
        top_factors=data.get("top_factors", []),
        shap_values=data.get("shap_values", {}),
        human_readable=data.get("human_readable", ""),
        counterfactual=data.get("counterfactual"),
    )
