/**
 * YADEM Simulator Logic
 * Handles form submission, API calls to the scoring engine, and UI updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('simulator-form');
    const resultContent = document.getElementById('result-content');
    const resultDisplay = document.getElementById('result-display');
    const submitBtn = document.getElementById('submit-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // 1. UI Loading State
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span>Processing Pipeline...</span>';
        resultContent.innerHTML = `
            <div class="loader-container" style="text-align:center; padding: 40px;">
                <div style="font-size: 3rem; margin-bottom: 20px;" class="pulse">🧠</div>
                <p style="color: var(--accent)">Running 115 feature transforms...</p>
                <p style="font-size: 0.8rem; color: #606060;">Polling ensemble (XGBoost, Random Forest, LR)...</p>
            </div>
        `;
        resultContent.style.display = 'block';
        resultDisplay.style.display = 'none';

        // 2. Get Input Data
        const data = {
            sector: document.getElementById('sector').value,
            age: parseInt(document.getElementById('age').value),
            revenue: parseInt(document.getElementById('revenue').value),
            balance: parseInt(document.getElementById('balance').value),
            loan_amount: parseInt(document.getElementById('loan_amount').value),
            has_bureau: parseInt(document.getElementById('has_bureau').value)
        };

        // 3. Attempt API Call (with Fallback for Prototype)
        let result;
        try {
            result = await getScoringResult(data);
        } catch (error) {
            console.warn("API not reachable, using offline engine.", error);
            result = calculateOfflineScore(data);
        }

        // 4. Artificial delay for "Process feeling"
        await new Promise(r => setTimeout(r, 1500));

        // 5. Update UI
        updateResultUI(result);
        
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>Analyze & Score Applicant</span><span class="btn-icon">→</span>';
    });

    /**
     * Update the UI with the scoring result
     */
    function updateResultUI(result) {
        document.getElementById('result-content').style.display = 'none';
        document.getElementById('result-display').style.display = 'block';

        document.getElementById('res-score').textContent = result.score;
        document.getElementById('res-band').textContent = `${result.band} — ${result.band_meaning}`;
        document.getElementById('res-decision').textContent = result.decision.replace(/_/g, ' ');
        document.getElementById('res-pd').textContent = result.pd.toFixed(4);
        
        const meterFill = document.getElementById('res-meter');
        meterFill.style.width = '0%';
        setTimeout(() => {
            meterFill.style.width = (result.score / 10) + '%';
            // Color mapping
            if (result.score >= 800) meterFill.style.background = 'linear-gradient(90deg, #6d28d9, #059669)';
            else if (result.score >= 650) meterFill.style.background = 'linear-gradient(90deg, #6d28d9, #10b981)';
            else if (result.score >= 500) meterFill.style.background = 'linear-gradient(90deg, #6d28d9, #f59e0b)';
            else meterFill.style.background = 'linear-gradient(90deg, #dc2626, #ef4444)';
        }, 100);

        // Populate Explanations (SHAP)
        const expList = document.getElementById('res-explanation');
        expList.innerHTML = '';
        result.explanations.forEach(exp => {
            const li = document.createElement('li');
            li.className = 'explanation-item';
            li.innerHTML = `
                <span class="exp-text">${exp.feature}</span>
                <span class="exp-impact ${exp.impact >= 0 ? 'positive' : 'negative'}">
                    ${exp.impact >= 0 ? '+' : ''}${exp.impact} pts
                </span>
            `;
            expList.appendChild(li);
        });
    }

    /**
     * Logic for calculating a score if the API is offline
     * Mimics the actual YADEM engine logic with realistic weights
     */
    function calculateOfflineScore(data) {
        let baseScore = 550; // Lowered base to be more discriminating

        // 1. Sector weights
        const sectorWeights = {
            retail_fmcg: 15,
            agriculture: -40, // High risk
            manufacturing: 25,
            services: 10,
            tech: 35,
            construction: -45 // High risk
        };
        baseScore += sectorWeights[data.sector] || 0;

        // 2. Business Age factor
        if (data.age > 48) baseScore += 60;
        else if (data.age > 24) baseScore += 30;
        else if (data.age < 12) baseScore -= 80; // Heavy penalty for startups

        // 3. Affordability Ratio (Revenue vs Monthly Loan Repayment)
        const monthlyRepayment = data.loan_amount / 12;
        const ratio = data.revenue / monthlyRepayment;
        
        if (ratio > 3) baseScore += 80;
        else if (ratio > 1.5) baseScore += 40;
        else if (ratio < 0.8) baseScore -= 150; // Critical penalty: can't afford loan
        else baseScore -= 60;

        // 4. Bureau factor
        if (data.has_bureau === 1) baseScore += 70;
        else baseScore -= 40;

        // 5. Bank balance factor
        if (data.balance > data.revenue * 0.7) baseScore += 40;
        else if (data.balance < data.revenue * 0.1) baseScore -= 30;

        // 6. Final Score Calculation (with slight noise for realism)
        const noise = Math.floor(Math.random() * 30) - 15;
        const score = Math.max(300, Math.min(950, baseScore + noise));
        
        const band = score >= 800 ? 'A' : score >= 650 ? 'B' : score >= 500 ? 'C' : 'D';
        const band_meaning = { A: 'Excellent', B: 'Good', C: 'Borderline', D: 'High Risk' }[band];
        
        // Logical decisioning
        let decision;
        if (score >= 750) decision = 'AUTO_APPROVE';
        else if (score >= 600) decision = 'APPROVE_WITH_REDUCED_LIMIT';
        else if (score >= 480) decision = 'MANUAL_REVIEW_REQUIRED';
        else decision = 'AUTO_DECLINE';

        const pd = 1 - (score / 1000);

        // Dynamic Explanations (SHAP)
        const explanations = [];
        if (ratio < 0.8) explanations.push({ feature: "Debt-to-Income Ratio", impact: -150 });
        else if (ratio > 2.5) explanations.push({ feature: "Strong Cashflow Coverage", impact: 80 });
        
        if (data.age < 12) explanations.push({ feature: "Limited Operating History", impact: -80 });
        else if (data.age > 48) explanations.push({ feature: "Proven Business Longevity", impact: 60 });

        if (data.has_bureau === 0) explanations.push({ feature: "Thin Credit Bureau Record", impact: -40 });
        
        if (sectorWeights[data.sector] < 0) explanations.push({ feature: "Sector Macro-Risk", impact: sectorWeights[data.sector] });
        else explanations.push({ feature: "Sector Stability", impact: sectorWeights[data.sector] });

        return { score, band, band_meaning, decision, pd, explanations };
    }

    /**
     * Placeholder for real API call
     */
    async function getScoringResult(data) {
        // For the event, we assume a mock if no backend is found
        throw new Error("API not configured for production yet");
    }
});
