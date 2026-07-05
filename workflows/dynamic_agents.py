from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TemporarySpecialistAgent:
    name: str
    purpose: str
    confidence: float
    explanation: str


class DynamicAgentFactory:
    """Create temporary specialist agents when the dataset suggests a niche domain."""

    def create(self, dataset_summary: dict[str, Any], risk_profile: dict[str, Any]) -> list[TemporarySpecialistAgent]:
        feature_columns = dataset_summary.get("feature_columns", [])
        target_summary = dataset_summary.get("target_summary", {})
        task_type = dataset_summary.get("task_type", "classification")
        agents: list[TemporarySpecialistAgent] = []

        if self._looks_like_time_series(feature_columns):
            agents.append(
                TemporarySpecialistAgent(
                    name="Forecasting Agent",
                    purpose="Inspect temporal columns and suggest lag-aware handling.",
                    confidence=0.88,
                    explanation="Date-like columns suggest a forecasting lens could improve feature design.",
                )
            )
        if self._looks_like_nlp(feature_columns):
            agents.append(
                TemporarySpecialistAgent(
                    name="NLP Expert Agent",
                    purpose="Inspect free-text fields and recommend text processing.",
                    confidence=0.84,
                    explanation="Text-heavy columns may benefit from token-aware feature engineering.",
                )
            )
        if self._looks_like_fraud(task_type, target_summary, risk_profile):
            agents.append(
                TemporarySpecialistAgent(
                    name="Fraud Detection Agent",
                    purpose="Strengthen anomaly-aware treatment for highly imbalanced targets.",
                    confidence=0.92,
                    explanation="Class imbalance and risk signals justify a fraud-oriented specialist.",
                )
            )
        if self._looks_like_recommendation(feature_columns):
            agents.append(
                TemporarySpecialistAgent(
                    name="Recommendation System Agent",
                    purpose="Check whether entity-id columns indicate a recommender framing.",
                    confidence=0.79,
                    explanation="Multiple identifier-like columns can hint at a recommendation task.",
                )
            )
        if self._looks_like_anomaly(task_type, risk_profile):
            agents.append(
                TemporarySpecialistAgent(
                    name="Anomaly Detection Agent",
                    purpose="Recommend anomaly-aware validation and monitoring.",
                    confidence=0.8,
                    explanation="Higher risk and unusual patterns make anomaly handling relevant.",
                )
            )
        if not agents:
            agents.append(
                TemporarySpecialistAgent(
                    name="Generalist Specialist Agent",
                    purpose="Provide broad domain guidance for the current dataset.",
                    confidence=0.74,
                    explanation="No strong niche signal was detected, so a generalist specialist is safer.",
                )
            )
        return agents

    def _looks_like_time_series(self, feature_columns: list[str]) -> bool:
        tokens = ("date", "time", "timestamp", "month", "day", "year")
        return any(any(token in str(column).lower() for token in tokens) for column in feature_columns)

    def _looks_like_nlp(self, feature_columns: list[str]) -> bool:
        tokens = ("text", "review", "comment", "description", "message", "content", "note")
        return any(any(token in str(column).lower() for token in tokens) for column in feature_columns)

    def _looks_like_fraud(self, task_type: str, target_summary: dict[str, Any], risk_profile: dict[str, Any]) -> bool:
        distribution = target_summary.get("distribution", {})
        values = list(distribution.values())
        imbalance = min(values) / max(1, sum(values)) if values else 1.0
        return task_type == "classification" and imbalance < 0.2 and risk_profile.get("risk_score", 0) >= 35

    def _looks_like_recommendation(self, feature_columns: list[str]) -> bool:
        tokens = ("user", "item", "product", "rating", "movie", "purchase")
        return sum(any(token in str(column).lower() for token in tokens) for column in feature_columns) >= 2

    def _looks_like_anomaly(self, task_type: str, risk_profile: dict[str, Any]) -> bool:
        return task_type == "classification" and risk_profile.get("risk_score", 0) >= 50
