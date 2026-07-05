from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from utils.helper import make_json_safe, save_json, load_json


@dataclass(slots=True)
class ExperimentArtifact:
    experiment_id: str
    directory: Path
    metadata_path: Path


class ExperimentTracker:
    """Track each run as a first-class experiment with comparable metadata."""

    def __init__(self, experiments_dir: Path) -> None:
        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.experiments_dir / "index.json"

    def start(self, dataset_signature: str) -> ExperimentArtifact:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"exp_{timestamp}_{dataset_signature[:8]}"
        directory = self.experiments_dir / experiment_id
        directory.mkdir(parents=True, exist_ok=True)
        return ExperimentArtifact(experiment_id=experiment_id, directory=directory, metadata_path=directory / "metadata.json")

    def finalize(self, artifact: ExperimentArtifact, metadata: dict[str, Any]) -> None:
        metadata = {**metadata, "experiment_id": artifact.experiment_id}
        save_json(metadata, artifact.metadata_path)
        self._update_index(metadata)

    def compare(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        return load_json(self.index_path)

    def _update_index(self, metadata: dict[str, Any]) -> None:
        index = self.compare()
        index.append({
            "experiment_id": metadata.get("experiment_id"),
            "dataset_signature": metadata.get("dataset_signature"),
            "winner": metadata.get("winner"),
            "metric": metadata.get("primary_metric"),
            "execution_seconds": metadata.get("execution_seconds"),
            "timestamp": metadata.get("timestamp"),
        })
        save_json(index, self.index_path)
