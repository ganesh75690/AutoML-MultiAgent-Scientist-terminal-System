from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlanItem:
    name: str
    reason: str
    confidence: float
    explanation: str


@dataclass(slots=True)
class ExecutionPlan:
    goal: str
    required_agents: list[str]
    estimated_time_minutes: int
    complexity: str
    memory_estimate_mb: int
    risk_level: str
    steps: list[str] = field(default_factory=list)
    agent_items: list[PlanItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class PlanningEngine:
    """Generate a compact execution plan before the workflow starts."""

    def build_plan(self, dataset_summary: dict[str, Any], risk_profile: dict[str, Any], dynamic_agents: list[str]) -> ExecutionPlan:
        rows = int(dataset_summary.get("rows", 0))
        columns = int(dataset_summary.get("columns", 0))
        task_type = dataset_summary.get("task_type", "classification")
        quality_score = float(dataset_summary.get("quality_score", 70))
        risk_score = float(risk_profile.get("risk_score", 50))

        if rows < 1_000:
            estimated_time = 10
            complexity = "low"
            memory_estimate = 512
        elif rows < 50_000:
            estimated_time = 25
            complexity = "medium"
            memory_estimate = 1024
        else:
            estimated_time = 45
            complexity = "high"
            memory_estimate = 2048

        if risk_score >= 70 or quality_score < 55:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        plan = ExecutionPlan(
            goal=f"Automatically build the best {task_type} pipeline for a CSV dataset.",
            required_agents=[
                "Supervisor Agent",
                "Data Cleaning Agent",
                "EDA Agent",
                "Feature Engineering Agent",
                "Model Selection Agent",
                "Tuning Agent",
                "Evaluation Agent",
                "Explainability Agent",
                "Report Agent",
                "Critic Agent",
                "Mentor Agent",
            ],
            estimated_time_minutes=estimated_time,
            complexity=complexity,
            memory_estimate_mb=memory_estimate,
            risk_level=risk_level,
            steps=[
                "Load dataset and infer target.",
                "Profile data quality and domain signals.",
                "Debate and compare candidate models.",
                "Tune the winner and evaluate on holdout data.",
                "Explain the result and generate the full report bundle.",
            ],
            notes=[
                f"Quality score estimate: {quality_score:.0f}/100.",
                f"Dynamic specialists enabled: {', '.join(dynamic_agents) if dynamic_agents else 'none'}.",
            ],
        )
        plan.agent_items = [
            PlanItem(
                name="Supervisor Agent",
                reason="Orchestrates the workflow and handles failures.",
                confidence=0.98,
                explanation="Coordinates planning, debate, retries, and final synthesis.",
            ),
            PlanItem(
                name="AI Debate Council",
                reason="Ensures model decisions are evidence-based and not premature.",
                confidence=0.91,
                explanation="Model-specialist agents will vote after comparing evidence.",
            ),
            PlanItem(
                name="Temporary Specialist Agents",
                reason="Extend analysis only when the dataset strongly suggests a domain.",
                confidence=0.87,
                explanation="Specialists are created only for the current task and then discarded.",
            ),
        ]
        return plan
