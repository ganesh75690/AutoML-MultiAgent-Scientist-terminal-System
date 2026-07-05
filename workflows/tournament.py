from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TournamentMatch:
    round_name: str
    left: str
    right: str
    winner: str
    confidence: float


@dataclass(slots=True)
class TournamentResult:
    champion: str
    matches: list[TournamentMatch]


class ModelTournament:
    """Run a simple knockout tournament over candidate model scores."""

    def run(self, ranked_models: list[dict[str, Any]]) -> TournamentResult:
        if not ranked_models:
            return TournamentResult(champion="Unknown", matches=[])

        current_round = [dict(model) for model in ranked_models]
        matches: list[TournamentMatch] = []
        round_names = ["Quarter Finals", "Semi Finals", "Final"]
        round_index = 0

        while len(current_round) > 1:
            next_round: list[dict[str, Any]] = []
            round_name = round_names[min(round_index, len(round_names) - 1)]
            pairs = [current_round[index : index + 2] for index in range(0, len(current_round), 2)]
            for pair in pairs:
                if len(pair) == 1:
                    next_round.append(pair[0])
                    continue
                left = pair[0]
                right = pair[1]
                left_score = float(left.get("score", 0.0))
                right_score = float(right.get("score", 0.0))
                if left_score >= right_score:
                    winner = left
                    loser = right
                else:
                    winner = right
                    loser = left
                confidence = min(0.99, 0.5 + abs(left_score - right_score) / 2.0)
                matches.append(
                    TournamentMatch(
                        round_name=round_name,
                        left=str(left.get("name")),
                        right=str(right.get("name")),
                        winner=str(winner.get("name")),
                        confidence=round(confidence, 3),
                    )
                )
                next_round.append(winner)
            current_round = next_round
            round_index += 1

        champion = str(current_round[0].get("name"))
        return TournamentResult(champion=champion, matches=matches)
