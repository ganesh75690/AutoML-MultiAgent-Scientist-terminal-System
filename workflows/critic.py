from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CriticResult:
    issues: list[str]
    suggestions: list[str]
    severity: str


class AICriticAgent:
    """Audit the workflow for weak preprocessing, leakage, and modeling gaps."""

    def review(self, summary: dict[str, Any]) -> CriticResult:
        issues: list[str] = []
        suggestions: list[str] = []
        metrics = summary.get("metrics", {})
        best_model = summary.get("best_model", {})
        task_type = summary.get("task_type", "classification")

        if summary.get("risk_assessment", {}).get("risk_score", 0) >= 60:
            issues.append("The dataset carries moderate to high risk and deserves stricter validation.")
        if task_type == "classification" and metrics.get("f1_score", metrics.get("accuracy", 0)) < 0.8:
            issues.append("Classification performance is below a strong production threshold.")
            suggestions.append("Try stronger imbalance handling, feature selection, or calibration.")
        if task_type == "regression" and metrics.get("r2_score", 0) < 0.75:
            issues.append("Regression performance suggests the signal may be weak or noisy.")
            suggestions.append("Experiment with additional engineered features and domain-specific transformations.")
        if best_model.get("name") in {"Decision Tree", "Decision Tree Regressor"}:
            issues.append("The final model may be too brittle if the tree is shallowly tuned.")
            suggestions.append("Consider an ensemble or boosting model for better generalization.")
        if not summary.get("explainability", {}).get("top_features"):
            issues.append("Explainability output is thin, which limits trust and auditability.")
            suggestions.append("Enable more stable model-agnostic attribution methods on future runs.")

        if not suggestions:
            suggestions.append("Keep the current workflow, but monitor drift and retrain on a schedule.")
        severity = "high" if len(issues) >= 3 else "medium" if len(issues) >= 1 else "low"
        return CriticResult(issues=issues or ["No major issues detected."], suggestions=suggestions, severity=severity)
