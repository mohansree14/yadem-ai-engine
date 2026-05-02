# YADEM AI Credit Decisioning Engine

> **AI-Powered Credit Infrastructure for African SMEs**
> Closing the $150B+ SME financing gap with explainable, fair machine learning.

---

## Table of Contents

1. [What is YADEM?](#what-is-yadem)
2. [The Problem We Solve](#the-problem-we-solve)
3. [Architecture Overview](#architecture-overview)
4. [Project Structure](#project-structure)
5. [Tech Stack](#tech-stack)
6. [Getting Started](#getting-started)
7. [Module-by-Module Documentation](#module-by-module-documentation)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Deployment](#deployment)
12. [Frontend Landing Page](#frontend-landing-page)
13. [What's Production-Ready vs What Needs Work](#whats-production-ready-vs-what-needs-work)

---

## What is YADEM?

YADEM is a **B2B REST API** that takes raw SME applicant data and returns a complete credit decision in under 200ms. Lenders integrate it into their platforms via a single API call and receive:

- A **credit score** (0–1,000)
- A **risk band** (A through E)
- An **automated decision** (approve / review / decline)
- A **SHAP explanation** of why
- A **fraud screening** result

It is built specifically for **Nigerian/African SMEs** who lack traditional credit bureau history.

---

## The Problem We Solve

| Problem | Impact |
|---------|--------|
| 80% of SMEs have no credit bureau record | Traditional scorecards reject them by default |
| Manual underwriting takes 2–6 weeks | Only 12% of applications reach a decision |
| No explanations given on rejection | Violates NDPA 2023 and erodes trust |

YADEM solves this by scoring SMEs using **alternative data** (mobile money, POS, bank statements) combined with traditional bureau data when available.

---

## Architecture Overview

Every loan application flows through a **7-stage pipeline**:

```
┌─────────────────────────────────────────────────────────────┐
│                    YADEM SCORING PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DATA INGESTION                                          │
│     Bank Statements (Open Banking) + Credit Bureau +        │
│     Mobile Money + POS + KYC (BVN/NIN/CAC)                  │
│                          ↓                                  │
│  2. FEATURE ENGINEERING                                     │
│     115 features using "5 C's of Credit" framework          │
│     (Capacity, Capital, Character, Collateral, Conditions)  │
│                          ↓                                  │
│  3. ENSEMBLE SCORING (3 models vote)                        │
│     ┌──────────────┬──────────────┬──────────────┐          │
│     │  Logistic     │  Random      │  XGBoost     │          │
│     │  Regression   │  Forest      │              │          │
│     │  (30% weight) │  (30% weight)│  (40% weight)│          │
│     └──────┬───────┴──────┬───────┴──────┬───────┘          │
│            └──────────────┼──────────────┘                  │
│                           ↓                                  │
│  4. SCORE GENERATION (0–1,000)                              │
│     Band A (800+) → Auto Approve                            │
│     Band B (650-799) → Approve Standard                     │
│     Band C (500-649) → Manual Review                        │
│     Band D (350-499) → Decline (with appeal)                │
│     Band E (<350) → Auto Decline                            │
│                          ↓                                  │
│  5. SHAP EXPLAINABILITY                                     │
│     Top risk factors + strengths in plain language           │
│                          ↓                                  │
│  6. FRAUD SCREENING (runs in parallel with 3-5)             │
│     Karma Blacklist + Device Fingerprint + Velocity Check   │
│                          ↓                                  │
│  7. API RESPONSE                                            │
│     Score + Band + Decision + Explanation + Fraud Result     │
│     Delivered as JSON in <200ms                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
yadem-ai-engine/
│
├── train.py                    # Run this to train all models
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project config (pytest, ruff, mypy)
├── Makefile                    # Automation commands
├── .env.example                # Environment variable template
├── .gitignore
│
├── src/                        # All source code
│   ├── config/                 # Configuration
│   │   ├── settings.py         # App settings (Pydantic, env vars)
│   │   ├── risk_config.py      # Risk band thresholds
│   │   └── model_config.py     # ML model hyperparameters
│   │
│   ├── data/                   # Data layer
│   │   ├── synthetic/
│   │   │   └── generator.py    # Generates fake Nigerian SME data for training
│   │   ├── processing/
│   │   │   └── cleaner.py      # Cleans, imputes missing values, balances classes
│   │   └── ingestion/
│   │       ├── open_banking.py # Bank statement parser (Mono, Okra format)
│   │       ├── credit_bureau.py# CRC, First Central, XDS bureau data
│   │       ├── alt_data.py     # Mobile money, POS, e-commerce data
│   │       └── kyc_kyb.py      # BVN, NIN, CAC verification
│   │
│   ├── features/
│   │   └── engine.py           # Transforms raw data → 115 ML features
│   │
│   ├── models/                 # Machine Learning models
│   │   ├── logistic.py         # Logistic Regression wrapper
│   │   ├── random_forest.py    # Random Forest wrapper
│   │   ├── xgboost_model.py    # XGBoost wrapper
│   │   └── ensemble.py         # Combines all 3 models with weighted voting
│   │
│   ├── scoring/                # Score + Decision logic
│   │   ├── scorer.py           # Probability → 0-1000 score + risk band
│   │   ├── risk_bands.py       # A-E band definitions and routing rules
│   │   └── decision_rules.py   # 6 business rules (CBN DTI, sector risk, etc.)
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py   # SHAP TreeExplainer for feature importance
│   │   └── report_generator.py # Human-readable credit reports
│   │
│   ├── fraud/
│   │   └── screener.py         # Blacklist + Device + Velocity fraud checks
│   │
│   ├── compliance/             # Nigerian Data Protection Act (NDPA) 2023
│   │   ├── ndpa.py             # Lawful basis checks, DPIA generation
│   │   ├── consent_manager.py  # Grant, verify, withdraw consent tokens
│   │   ├── bias_auditor.py     # Fairness checks (disparate impact, etc.)
│   │   └── encryption.py       # AES-256 encryption + PII masking
│   │
│   └── api/                    # REST API (FastAPI)
│       ├── main.py             # App startup, model loading, route registration
│       ├── schemas/
│       │   └── application.py  # Request/response Pydantic models
│       ├── routes/
│       │   ├── score.py        # POST /api/v1/score
│       │   ├── health.py       # GET  /api/v1/health
│       │   ├── kyc.py          # POST /api/v1/kyc
│       │   ├── fraud.py        # POST /api/v1/fraud-check
│       │   └── explain.py      # GET  /api/v1/explain/{id}
│       └── middleware/
│           ├── auth.py         # API key validation
│           ├── rate_limit.py   # Token bucket rate limiting
│           └── consent.py      # Consent token validation
│
├── models/                     # Trained model artifacts (generated by train.py)
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── ensemble_meta.joblib
│   ├── data_cleaner.joblib
│   └── shap_explainer.joblib
│
├── tests/
│   └── unit/
│       └── test_engine.py      # Unit tests for core components
│
├── docker/
│   ├── Dockerfile.api          # Docker image for API
│   └── docker-compose.yml      # Full stack (API + Redis + Postgres)
│
├── .github/workflows/
│   ├── ci.yml                  # Lint + test on every push
│   └── cd.yml                  # Docker build + deploy on tags
│
└── frontend/                   # Landing page (standalone HTML/CSS/JS)
    ├── index.html
    ├── css/style.css
    └── js/main.js
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **API** | FastAPI | Fastest Python API framework, auto-generates Swagger docs |
| **ML Models** | scikit-learn, XGBoost | Industry standard, proven in production credit scoring |
| **Explainability** | SHAP | Gold standard for ML interpretability, required by regulators |
| **Data** | Pandas, NumPy | Fast data manipulation for feature engineering |
| **Class Balancing** | imbalanced-learn (SMOTE) | Handles the fact that defaults are rare (imbalanced data) |
| **Config** | Pydantic Settings | Type-safe config with environment variable validation |
| **Logging** | Loguru | Clean, structured logging with rotation |
| **Serialization** | Joblib | Saves/loads trained models to disk |
| **Encryption** | cryptography (AES-256) | Encrypts PII data at rest |
| **Containerization** | Docker + Compose | Consistent deployment across environments |
| **CI/CD** | GitHub Actions | Automated testing and deployment |

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### 1. Install Dependencies

```bash
cd yadem-ai-engine
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set:
#   YADEM_API_KEYS=your-api-key-here
#   YADEM_ENCRYPTION_KEY=your-32-byte-key-here
```

### 3. Train Models

```bash
python train.py
```

This will:
- Generate 5,000 synthetic Nigerian SME records
- Engineer 115 features from the raw data
- Train 3 ML models (Logistic Regression, Random Forest, XGBoost)
- Fit the SHAP explainer
- Save all artifacts to `./models/`

Expected output:
```
Test AUC-ROC:   0.9928
Test Gini:      0.9857
Test KS:        0.9388
Features used:  115
Training time:  ~4 seconds
```

### 4. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. Test It

Access the interactive Swagger UI at `/docs` on the local server, or:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "bvn": "22345678901",
    "business_sector": "retail_fmcg",
    "business_age_months": 48,
    "requested_loan_amount_ngn": 2000000,
    "financial": {
      "avg_monthly_revenue_6m": 850000,
      "cashflow_volatility_6m": 0.15,
      "avg_monthly_balance": 1200000
    }
  }'
```

---

## Module-by-Module Documentation

### `src/config/` — Configuration

| File | What It Does |
|------|-------------|
| `settings.py` | Loads all config from environment variables using Pydantic. Includes: API keys, model paths, CORS origins, encryption keys. |
| `risk_config.py` | Defines the 5 risk bands (A–E) with their score thresholds, decisions, max tenures, and rate multipliers. |
| `model_config.py` | Stores hyperparameters for all 3 ML models. Change these to tune model performance. |

### `src/data/synthetic/generator.py` — Synthetic Data

Generates realistic fake data mimicking Nigerian SME borrowers. Includes:
- Business demographics (sector, state, age, employees)
- Financial data (revenue, expenses, bank balance)
- Credit bureau records (when available)
- Mobile money and POS transaction data
- KYC flags (BVN verified, CAC registered)
- A realistic `default` label based on financial health signals

**When to use:** For training and testing when you don't have real borrower data yet.

### `src/data/processing/cleaner.py` — Data Cleaning

- Imputes missing values (median for numbers, mode for categories)
- Clips outliers using IQR method
- Encodes categorical variables (one-hot encoding)
- Applies SMOTE to balance defaulters vs non-defaulters
- Saves the fitted cleaner so the same transforms apply to new data

### `src/data/ingestion/` — Data Ingestion

| File | Source | What It Parses |
|------|--------|---------------|
| `open_banking.py` | Mono, Okra APIs | Bank statement transactions → monthly revenue, expenses, balance |
| `credit_bureau.py` | CRC, First Central, XDS | Bureau scores, active loans, delinquencies, payment history |
| `alt_data.py` | Mobile Money, POS | Transaction volumes, frequency, consistency scores |
| `kyc_kyb.py` | NIBSS, NIMC, CAC | BVN validation, NIN matching, business registration status |

**Note:** These currently use **simulated responses**. For production, replace the mock functions with actual API client calls.

### `src/features/engine.py` — Feature Engineering

The heart of the system. Transforms raw applicant data into **115 predictive features** organized by the "5 C's of Credit":

| Category | Example Features | Count |
|----------|-----------------|-------|
| **Capacity** | Revenue/expense ratio, DTI ratio, cashflow volatility | ~25 |
| **Capital** | Bank balance ratio, savings rate, net worth proxy | ~20 |
| **Character** | Bureau payment history, delinquency rate, bureau score | ~25 |
| **Collateral** | Has collateral flag, CAC registration, asset indicators | ~15 |
| **Conditions** | Sector risk, business age, economic region, employee count | ~15 |
| **Alternative** | Mobile money velocity, POS consistency, social media presence | ~15 |

### `src/models/` — Machine Learning Models

Three models, each with a different strength:

| Model | File | Strength | Weight |
|-------|------|----------|--------|
| Logistic Regression | `logistic.py` | Interpretable, stable, fast | 30% |
| Random Forest | `random_forest.py` | Handles noise, robust to outliers | 30% |
| XGBoost | `xgboost_model.py` | Highest accuracy, captures non-linear patterns | 40% |

**`ensemble.py`** — Combines all 3 predictions using weighted averaging:
```
final_probability = 0.30 * LR + 0.30 * RF + 0.40 * XGB
```

Each model wrapper has `.train()`, `.predict()`, `.save()`, and `.load()` methods.

### `src/scoring/` — Score Generation & Decision Rules

| File | What It Does |
|------|-------------|
| `scorer.py` | Converts the ensemble probability (0.0–1.0) into a YADEM Score (0–1,000) and maps it to a risk band (A–E). |
| `risk_bands.py` | Defines each band: A=Excellent (800+), B=Good (650-799), C=Borderline (500-649), D=High Risk (350-499), E=Blacklist (<350). |
| `decision_rules.py` | 6 hard-coded business rules that can **override** the model. Example: if DTI > 70%, force decline regardless of score. |

**Decision Rules Applied:**
1. CBN Affordability Test (DTI ratio check)
2. Active Default Block (any active default = auto decline)
3. Sector Risk Override (high-risk sectors capped at Band C)
4. Minimum Business Age (< 6 months = manual review)
5. Fraud Flag Override (fraud detected = auto decline)
6. Repeat Borrower Incentive (good history = band upgrade)

### `src/explainability/` — SHAP Explainability

| File | What It Does |
|------|-------------|
| `shap_explainer.py` | Uses SHAP TreeExplainer to decompose each prediction into per-feature contributions. Shows which features pushed the score up or down. |
| `report_generator.py` | Takes SHAP values and generates a human-readable text report. Example: "Your cashflow stability contributed +45 points to your score." |

### `src/fraud/screener.py` — Fraud Detection

Runs **3 checks in parallel** with credit scoring:

| Check | What It Does |
|-------|-------------|
| **Karma Blacklist** | Checks BVN, CAC, and device fingerprint against a shared industry fraud registry |
| **Device Fingerprinting** | Detects if multiple identities are applying from the same device (application farms) |
| **Velocity Check** | Flags if the same BVN submits 3+ applications in 24 hours |

If any check fails, the fraud result **overrides** a favourable credit decision.

### `src/compliance/` — NDPA 2023 Compliance

| File | What It Does |
|------|-------------|
| `ndpa.py` | Validates lawful basis for processing, generates Data Protection Impact Assessments (DPIA), enforces data retention limits |
| `consent_manager.py` | Issues consent tokens, verifies them before data access, handles withdrawal requests |
| `bias_auditor.py` | Checks model fairness across demographics (gender, region, sector) using disparate impact ratio and equal opportunity metrics |
| `encryption.py` | AES-256 encryption for PII at rest, plus a PII masking utility (e.g., BVN `223****8901`) |

### `src/api/` — REST API

| File | What It Does |
|------|-------------|
| `main.py` | FastAPI application. Loads trained models on startup. Registers all routes and middleware. |
| `schemas/application.py` | Pydantic models defining the exact shape of every request and response. |
| `routes/score.py` | **POST /api/v1/score** — The main endpoint. Takes applicant data, runs the full pipeline, returns decision. |
| `routes/health.py` | **GET /api/v1/health** — Returns whether models are loaded and uptime. |
| `routes/kyc.py` | **POST /api/v1/kyc** — Standalone KYC verification (BVN + NIN + CAC). |
| `routes/fraud.py` | **POST /api/v1/fraud-check** — Standalone fraud screening without credit scoring. |
| `routes/explain.py` | **GET /api/v1/explain/{id}** — Retrieve SHAP explanation for a previous decision. |
| `middleware/auth.py` | Validates `X-API-Key` header against configured keys. |
| `middleware/rate_limit.py` | Token bucket rate limiter (100 requests/minute default). |
| `middleware/consent.py` | Validates consent token before processing personal data. |

---

## API Reference

### POST `/api/v1/score`

Score an SME loan application.

**Request Body:**
```json
{
  "bvn": "22345678901",
  "business_sector": "retail_fmcg",
  "business_state": "Lagos",
  "business_age_months": 48,
  "num_employees": 8,
  "is_registered_cac": 1,
  "owner_age": 38,
  "owner_education": "tertiary",
  "requested_loan_amount_ngn": 2000000,
  "requested_tenure_months": 12,
  "financial": {
    "avg_monthly_revenue_6m": 850000,
    "avg_monthly_revenue_3m": 900000,
    "cashflow_volatility_6m": 0.15,
    "avg_monthly_balance": 1200000,
    "has_mobile_money": 1,
    "mobile_money_monthly_volume": 250000
  },
  "bureau": {
    "has_bureau_record": 1,
    "bureau_score_crc": 680,
    "num_active_loans": 1,
    "num_on_time_payments_12m": 10
  }
}
```

**Response:**
```json
{
  "applicant_id": "YDM-4422A012",
  "yadem_score": 685,
  "probability_of_default": 0.2552,
  "risk_band": "B",
  "risk_band_meaning": "Good",
  "decision": "AUTO_APPROVE_STANDARD",
  "max_tenure_months": 24,
  "rate_multiplier": 1.0,
  "individual_model_scores": {
    "logistic_regression": 0.7782,
    "random_forest": 0.13,
    "xgboost": 0.0378
  },
  "fraud_check": {
    "passed": true,
    "checks": { "blacklist": true, "device": true, "velocity": true },
    "flags": [],
    "risk_level": "low"
  },
  "processing_time_ms": 156.06
}
```

### POST `/api/v1/kyc`
```json
// Request
{ "bvn": "22345678901", "nin": "98765432101", "cac_number": "RC1234567" }

// Response
{ "verified": true, "bvn_valid": true, "nin_valid": true, "cac_valid": true, "identity_match_score": 1.0, "business_status": "active" }
```

### POST `/api/v1/fraud-check`
```json
// Request
{ "bvn": "22345678901", "application_amount": 5000000 }

// Response
{ "passed": true, "risk_level": "low", "checks": { "blacklist": true, "device": true, "velocity": true }, "flags": [] }
```

### GET `/api/v1/health`
```json
{ "status": "healthy", "models_loaded": true, "version": "1.0.0", "uptime_seconds": 13.58 }
```

---

## Configuration

All configuration is managed via **environment variables**. Copy `.env.example` to `.env`:

| Variable | Description | Default |
|----------|------------|---------|
| `YADEM_API_KEYS` | Comma-separated API keys for auth | `dev-key-123` |
| `YADEM_ENCRYPTION_KEY` | 32-byte AES key for PII encryption | Auto-generated |
| `YADEM_MODEL_PATH` | Path to trained model artifacts | `./models` |
| `YADEM_CORS_ORIGINS` | Allowed CORS origins | `*` |
| `YADEM_LOG_LEVEL` | Logging level | `INFO` |
| `YADEM_RATE_LIMIT` | Requests per minute | `100` |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Or using Make
make test
```

---

## Deployment

### Docker (Recommended)

```bash
# Build and start everything
docker-compose -f docker/docker-compose.yml up --build

# Or using Make
make docker-up
```

This starts:
- **API** on port 8000
- **Redis** on port 6379 (for rate limiting / caching)
- **PostgreSQL** on port 5432 (for persistent storage)

### Manual

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## Frontend Landing Page

A standalone product page located in `frontend/`. No build step required.

```bash
cd frontend
python -m http.server 8080
# View the application on the local server port 8080
```

**Sections:** Hero → Problem → 7-Stage Pipeline → 9 Capabilities → Live API Demo → Performance Metrics → Tech Stack → CTA → Footer

---

## What's Production-Ready vs What Needs Work

### ✅ Production-Ready

- Training pipeline (synthetic data → features → models → artifacts)
- 3-model ensemble with weighted voting
- Score generation and risk band mapping
- SHAP explainability
- Fraud screening (blacklist, device, velocity)
- REST API with auth, rate limiting, and consent middleware
- NDPA compliance module
- Docker + CI/CD configuration
- Landing page

### ⚠️ Needs Real Integration (Currently Simulated)

| Module | Current State | What's Needed |
|--------|--------------|---------------|
| `ingestion/kyc_kyb.py` | Returns mock BVN/NIN/CAC results | Connect to NIBSS BVN API, NIMC NIN API, CAC registry |
| `ingestion/open_banking.py` | Parses sample data | Connect to Mono / Okra APIs |
| `ingestion/credit_bureau.py` | Returns mock bureau scores | Connect to CRC / First Central / XDS APIs |
| `ingestion/alt_data.py` | Returns mock mobile money data | Connect to MoMo / POS provider APIs |
| `fraud/screener.py` | Uses in-memory blacklist | Connect to shared Karma Blacklist database |
| Database | In-memory state | Add PostgreSQL for decision audit trail |

### 🔜 Future Work

- Real borrower data for model retraining
- Lender dashboard (React/Next.js)
- Borrower portal (application status, explanation view)
- Grafana + Prometheus monitoring
- MLflow experiment tracking
- A/B testing framework for model versions

---

## Makefile Commands

```bash
make install     # Install Python dependencies
make train       # Generate data + train models
make serve       # Start API server (with hot reload)
make test        # Run test suite
make lint        # Run ruff linter
make docker-up   # Start full stack via Docker
make docker-down # Stop Docker stack
make clean       # Remove model artifacts and caches
```

---

## Questions?

This engine was built by the YADEM engineering team. For questions:
- API issues → Check `/docs` (Swagger UI) on your local instance.
- Model questions → See `train.py` and `src/models/ensemble.py`
- Compliance questions → See `src/compliance/ndpa.py`
