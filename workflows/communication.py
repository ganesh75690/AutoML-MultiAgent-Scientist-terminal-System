from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CommunicationMessage:
    sender: str
    recipient: str
    message: str
    confidence: float = 1.0
    explanation: str = ""


@dataclass(slots=True)
class DebateOutcome:
    winner: str
    votes: dict[str, int]
    confidence: float
    explanation: str


@dataclass(slots=True)
class CommunicationHub:
    """Capture agent-to-agent communication, debate, and decision timelines."""

    messages: list[CommunicationMessage] = field(default_factory=list)
    decision_timeline: list[str] = field(default_factory=list)
    votes: list[dict[str, Any]] = field(default_factory=list)

    def send(self, sender: str, recipient: str, message: str, confidence: float = 1.0, explanation: str = "") -> None:
        self.messages.append(
            CommunicationMessage(
                sender=sender,
                recipient=recipient,
                message=message,
                confidence=round(float(confidence), 3),
                explanation=explanation,
            )
        )

    def record_timeline(self, label: str) -> None:
        self.decision_timeline.append(label)

    def debate(self, opinions: list[dict[str, Any]]) -> DebateOutcome:
        vote_counts: dict[str, int] = {}
        weighted_scores: dict[str, float] = {}
        explanation_parts: list[str] = []
        for opinion in opinions:
            model_name = str(opinion.get("model_name", "unknown"))
            vote = int(opinion.get("vote", 1))
            confidence = float(opinion.get("confidence", 0.5))
            vote_counts[model_name] = vote_counts.get(model_name, 0) + vote
            weighted_scores[model_name] = weighted_scores.get(model_name, 0.0) + confidence * vote
            explanation_parts.append(f"{opinion.get('agent_name', 'Agent')} supported {model_name} with confidence {confidence:.2f}.")
        winner = max(weighted_scores, key=weighted_scores.get)
        confidence = min(0.99, max(weighted_scores.values()) / max(1, sum(abs(v) for v in vote_counts.values())))
        outcome = DebateOutcome(
            winner=winner,
            votes=vote_counts,
            confidence=round(confidence, 3),
            explanation=" ".join(explanation_parts),
        )
        self.votes.append({"winner": outcome.winner, "votes": outcome.votes, "confidence": outcome.confidence})
        return outcome
