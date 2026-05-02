# YADEM — Developer Quick Start Guide

> Get the credit engine running on your machine in 5 minutes.

---

## Step 1: Clone & Install

```bash
git clone <repo-url>
cd yadem-ai-engine
pip install -r requirements.txt
```

## Step 2: Configure

```bash
cp .env.example .env
```

Open `.env` and set at minimum:
```env
YADEM_API_KEYS=your-key-here
```

## Step 3: Train Models

```bash
python train.py
```

You should see:
```
TRAINING COMPLETE
  Test AUC-ROC:   0.9928
  Model dir:      ./models
  Features used:  115
```

## Step 4: Start API

```bash
uvicorn src.api.main:app --reload --port 8000
```

## Step 5: Test

Open http://localhost:8000/docs — this is the interactive Swagger UI.

Or use curl:
```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "bvn": "22345678901",
    "business_sector": "retail_fmcg",
    "business_age_months": 48,
    "requested_loan_amount_ngn": 2000000,
    "financial": {"avg_monthly_revenue_6m": 850000}
  }'
```

---

## Key Files to Know

| If you want to... | Look at... |
|-------------------|-----------|
| Understand the full pipeline | `train.py` |
| Change model hyperparameters | `src/config/model_config.py` |
| Change risk band thresholds | `src/config/risk_config.py` |
| Add new features | `src/features/engine.py` |
| Modify API endpoints | `src/api/routes/` |
| Add business rules | `src/scoring/decision_rules.py` |
| See request/response shapes | `src/api/schemas/application.py` |
| Understand the ensemble | `src/models/ensemble.py` |

---

## Common Commands

```bash
python train.py                                    # Train models
uvicorn src.api.main:app --reload --port 8000      # Start API
python -m pytest tests/ -v                         # Run tests
cd frontend && python -m http.server 8080          # View landing page
```
