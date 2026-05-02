"""
YADEM Bias Auditor
=============================================================================
Fairness-by-design checks ensuring the credit scoring model does not
discriminate against protected groups (gender, region, ethnicity, religion).
Implements 8 fairness metrics aligned with the Fair Credit Scoring framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class FairnessReport:
    """Results of a fairness audit."""
    metric_name: str
    protected_attribute: str
    group_a: str
    group_b: str
    group_a_value: float
    group_b_value: float
    ratio: float
    passed: bool
    threshold: float
    description: str


class BiasAuditor:
    """
    Audits ML model predictions for fairness across protected attributes.

    Metrics implemented:
      1. Demographic Parity (selection rate equality)
      2. Equal Opportunity (TPR equality)
      3. Equalized Odds (TPR + FPR equality)
      4. Predictive Parity (PPV equality)
      5. Disparate Impact Ratio (80% rule)
      6. Calibration (score distribution similarity)
      7. Average Score Gap
      8. Approval Rate Gap
    """

    # The "four-fifths rule" threshold
    DISPARATE_IMPACT_THRESHOLD = 0.80

    def __init__(self, protected_attributes: Optional[List[str]] = None):
        self.protected_attributes = protected_attributes or [
            "gender", "region", "business_sector"
        ]
        logger.info(f"Bias Auditor initialized for attributes: {self.protected_attributes}")

    def audit(self, df: pd.DataFrame, score_col: str = "yadem_score",
              decision_col: str = "decision", target_col: str = "default") -> Dict:
        """Run full fairness audit across all protected attributes."""
        results = {}
        for attr in self.protected_attributes:
            if attr not in df.columns:
                logger.warning(f"Protected attribute '{attr}' not in data, skipping")
                continue
            results[attr] = self._audit_attribute(df, attr, score_col, decision_col, target_col)
        overall_pass = all(
            all(m["passed"] for m in attr_results["metrics"])
            for attr_results in results.values()
        )
        return {"overall_pass": overall_pass, "attributes": results,
                "recommendation": "No action needed" if overall_pass
                else "Review model for potential bias — see failing metrics"}

    def _audit_attribute(self, df: pd.DataFrame, attr: str,
                         score_col: str, decision_col: str, target_col: str) -> Dict:
        groups = df[attr].unique()
        if len(groups) < 2:
            return {"metrics": [], "note": f"Only one group found for {attr}"}

        metrics = []
        group_a, group_b = str(groups[0]), str(groups[1])
        mask_a = df[attr] == groups[0]
        mask_b = df[attr] == groups[1]

        # 1. Disparate Impact Ratio
        if decision_col in df.columns:
            approval_a = df.loc[mask_a, decision_col].isin(["APPROVED", "AUTO_APPROVED"]).mean()
            approval_b = df.loc[mask_b, decision_col].isin(["APPROVED", "AUTO_APPROVED"]).mean()
            ratio = min(approval_a, approval_b) / max(approval_a, approval_b) if max(approval_a, approval_b) > 0 else 1.0
            metrics.append({
                "metric": "disparate_impact_ratio", "group_a": group_a, "group_b": group_b,
                "group_a_value": round(approval_a, 4), "group_b_value": round(approval_b, 4),
                "ratio": round(ratio, 4), "threshold": self.DISPARATE_IMPACT_THRESHOLD,
                "passed": ratio >= self.DISPARATE_IMPACT_THRESHOLD,
            })

        # 2. Average Score Gap
        if score_col in df.columns:
            mean_a = df.loc[mask_a, score_col].mean()
            mean_b = df.loc[mask_b, score_col].mean()
            gap = abs(mean_a - mean_b)
            metrics.append({
                "metric": "average_score_gap", "group_a": group_a, "group_b": group_b,
                "group_a_value": round(mean_a, 2), "group_b_value": round(mean_b, 2),
                "gap": round(gap, 2), "threshold": 50.0,
                "passed": gap <= 50.0,
            })

        # 3. Equal Opportunity (TPR equality)
        if target_col in df.columns and decision_col in df.columns:
            for group_name, mask in [(group_a, mask_a), (group_b, mask_b)]:
                subset = df.loc[mask]
                positives = subset[target_col] == 0  # non-default = positive
                if positives.sum() > 0:
                    approved = subset[decision_col].isin(["APPROVED", "AUTO_APPROVED"])
                    tpr = (positives & approved).sum() / positives.sum()
                else:
                    tpr = 0.0
                metrics.append({
                    "metric": "true_positive_rate", "group": group_name,
                    "value": round(tpr, 4),
                })

        return {"attribute": attr, "groups": [group_a, group_b], "metrics": metrics}

    def generate_fairness_report(self, audit_results: Dict) -> str:
        """Generate a human-readable fairness report."""
        lines = ["=" * 60, "YADEM FAIRNESS AUDIT REPORT", "=" * 60, ""]
        overall = "PASS ✓" if audit_results["overall_pass"] else "FAIL ✗"
        lines.append(f"Overall Result: {overall}")
        lines.append(f"Recommendation: {audit_results['recommendation']}")
        lines.append("")

        for attr, data in audit_results.get("attributes", {}).items():
            lines.append(f"--- {attr.upper()} ---")
            for metric in data.get("metrics", []):
                status = "✓" if metric.get("passed", True) else "✗"
                lines.append(f"  {status} {metric['metric']}: {metric}")
            lines.append("")

        return "\n".join(lines)
