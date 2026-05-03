# YADEM Prototype Presentation Strategy ⚡

This guide outlines how to present YADEM as a state-of-the-art prototype for events, investor pitches, or demo days.

## 1. The Narrative Arc (The Pitch)

| Phase | Key Message | Script Highlight |
|-------|-------------|------------------|
| **The Problem** | Financial Invisibility | "48M Nigerian SMEs drive 50% of GDP, yet 80% are rejected by banks because they have no 'traditional' credit history." |
| **The Solution** | Intelligent Infrastructure | "YADEM is the AI layer that turns alternative data—bank statements, POS, and mobile money—into an instant, bankable credit score." |
| **The Demo** | Speed & Intelligence | "Let's score a real-world SME in under 200ms." |
| **The USP** | Transparency (SHAP) | "We don't just say 'No'. We tell the lender exactly why, and how the SME can improve." |

---

## 2. The Demo Flow (Live Prototype)

When presenting, follow this 3-step interactive journey:

### Step 1: The "Invisible" Applicant
*   **Scenario**: A retail business owner in Lagos, 4 years in business, ₦850k monthly revenue, but zero credit bureau history.
*   **Action**: Input these details into the simulator.
*   **Expected Outcome**: YADEM finds signals in their cash flow volatility and business age that traditional bureaus miss.

### Step 2: The Ensemble Verdict
*   **Highlight**: Point out that **three different AI models** (Logistic Regression, Random Forest, XGBoost) just reached a consensus.
*   **Visual**: Show the score (e.g., 685 - Band B).

### Step 3: The "Why" (Explainability)
*   **Highlight**: This is the most important part for regulators (NDPA 2023).
*   **Visual**: Show the SHAP explanation (e.g., "+45 points for revenue consistency, -12 points for requested loan size").

---

## 3. Technical "Wow" Factors (For Q&A)

Be ready to mention these if technical experts ask questions:

1.  **The 5 C's Framework**: "We don't just look at cash. Our 115 features map to Capacity, Capital, Character, Collateral, and Conditions."
2.  **Hybrid Ensemble**: "We use a weighted ensemble of XGBoost (for accuracy) and Logistic Regression (for stability)."
3.  **Real-Time Fraud**: "Fraud screening (Karma Blacklist, Device Fingerprint) runs in parallel with the scoring pipeline."
4.  **Regulatory First**: "Built for NDPA 2023 compliance with built-in consent management and bias auditing."

---

## 4. Setting up the Event Environment

1.  **Local Server**: Run the API locally (`uvicorn src.api.main:app`) to ensure zero latency due to Wi-Fi issues.
2.  **Fullscreen Mode**: Use `F11` in the browser to show the Landing Page without browser UI.
3.  **QR Code**: Have a QR code on your last slide or table that links to the live landing page (if hosted) or your LinkedIn.

---

## 5. Potential "Gotcha" Questions

*   **Q: Where do you get the data?**
    *   *A: We integrate with Open Banking providers like Mono and Okra, plus direct integrations with Credit Bureaus and POS providers.*
*   **Q: How do you handle bias?**
    *   *A: We have a dedicated 'Bias Auditor' module that checks for disparate impact across gender and region before any model is deployed.*
*   **Q: Is the score legally binding?**
    *   *A: It's a decisioning recommendation. Lenders can set their own 'Hard Rules' (e.g., DTI ratios) that override the AI score.*
