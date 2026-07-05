from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MentorNote:
    title: str
    explanation: str


class AIMentorAgent:
    """Translate technical results into simple, user-friendly language."""

    def teach(self, summary: dict[str, Any]) -> list[MentorNote]:
        best_model = summary.get("best_model", {})
        metrics = summary.get("metrics", {})
        task_type = summary.get("task_type", "classification")
        notes = [
            MentorNote(
                title="Why this model won",
                explanation=self._winner_explanation(best_model, metrics, task_type),
            ),
            MentorNote(
                title="What the cleaning stage did",
                explanation="The workflow removed duplicates, filled missing values, and clipped extreme outliers so the model saw cleaner input.",
            ),
            MentorNote(
                title="How to think about the report",
                explanation="The report combines accuracy, error metrics, and feature explanations so you can judge both performance and trustworthiness.",
            ),
        ]
        return notes

    def _winner_explanation(self, best_model: dict[str, Any], metrics: dict[str, Any], task_type: str) -> str:
        name = best_model.get("name", "the selected model")
        if task_type == "classification":
            score = metrics.get("f1_score", metrics.get("accuracy", 0.0))
            return f"{name} was selected because it achieved the strongest classification score while balancing prediction quality and robustness against overfitting."
        score = metrics.get("r2_score", 0.0)
        return f"{name} was selected because it provided the best regression fit and explained the target better than the other candidates."
