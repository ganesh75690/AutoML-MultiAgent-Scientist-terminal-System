from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RiskAssessmentResult:
    risk_score: float
    quality_score: float
    findings: list[str]
    pii_columns: list[str]
    sensitive_columns: list[str]


class RiskAssessmentAgent:
    """Detect data quality issues, leakage risk, bias risk, and privacy signals."""

    PII_HINTS = ("email", "phone", "ssn", "social", "passport", "dob", "date_of_birth", "address", "name")
    SENSITIVE_HINTS = ("income", "salary", "medical", "health", "ethnicity", "gender", "race", "religion", "credit")

    def assess(self, dataset_summary: dict[str, Any], target_column: str, feature_columns: list[str]) -> RiskAssessmentResult:
        missing_map = dataset_summary.get("missing_values", {})
        total_missing = sum(int(value) for value in missing_map.values())
        rows = max(1, int(dataset_summary.get("rows", 1)))
        missing_ratio = total_missing / max(1, rows * max(1, len(missing_map)))
        duplicate_penalty = 0.0
        findings: list[str] = []

        if missing_ratio > 0.1:
            findings.append("Missing data is above a comfortable threshold.")
            duplicate_penalty += 15
        if dataset_summary.get("task_type") == "classification" and dataset_summary.get("target_summary", {}).get("distribution"):
            distribution = dataset_summary["target_summary"]["distribution"]
            values = list(distribution.values())
            if values and min(values) / max(1, sum(values)) < 0.2:
                findings.append("Class imbalance may bias the model toward the majority class.")
                duplicate_penalty += 12
        if dataset_summary.get("feature_count", 0) > 40:
            findings.append("Feature count is large enough to increase overfitting risk.")
            duplicate_penalty += 8

        pii_columns = [column for column in feature_columns + [target_column] if self._matches_any(column, self.PII_HINTS)]
        sensitive_columns = [column for column in feature_columns + [target_column] if self._matches_any(column, self.SENSITIVE_HINTS)]
        if pii_columns:
            findings.append("Potential PII columns detected and should be governed carefully.")
            duplicate_penalty += 20
        if sensitive_columns:
            findings.append("Sensitive business fields were detected.")
            duplicate_penalty += 8

        quality_score = max(0.0, 100.0 - (missing_ratio * 100.0 * 0.8) - len(pii_columns) * 8 - len(sensitive_columns) * 3)
        risk_score = min(100.0, duplicate_penalty + (100.0 - quality_score) * 0.5)
        return RiskAssessmentResult(
            risk_score=round(risk_score, 2),
            quality_score=round(quality_score, 2),
            findings=findings or ["No major risk signals detected from lightweight inspection."],
            pii_columns=pii_columns,
            sensitive_columns=sensitive_columns,
        )

    def _matches_any(self, column: str, hints: tuple[str, ...]) -> bool:
        normalized = column.lower().replace(" ", "_")
        return any(hint in normalized for hint in hints)
