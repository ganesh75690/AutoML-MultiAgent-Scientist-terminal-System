from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.helper import load_json, save_json


@dataclass(slots=True)
class MemoryRecord:
    key: str
    value: Any


class AgentMemoryStore:
    """Persist lightweight global and per-agent memories between runs."""

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.global_memory_path = self.memory_dir / "global.json"
        self.agents_dir = self.memory_dir / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)

    def load_global(self) -> dict[str, Any]:
        return load_json(self.global_memory_path) if self.global_memory_path.exists() else {"history": [], "successful_models": [], "failures": []}

    def save_global(self, data: dict[str, Any]) -> None:
        save_json(data, self.global_memory_path)

    def load_agent(self, agent_name: str) -> dict[str, Any]:
        path = self.agents_dir / f"{agent_name.lower().replace(' ', '_')}.json"
        return load_json(path) if path.exists() else {"history": [], "failures": [], "wins": []}

    def save_agent(self, agent_name: str, data: dict[str, Any]) -> None:
        path = self.agents_dir / f"{agent_name.lower().replace(' ', '_')}.json"
        save_json(data, path)

    def record_failure(self, agent_name: str, failure: str) -> None:
        agent_memory = self.load_agent(agent_name)
        agent_memory.setdefault("failures", []).append(failure)
        self.save_agent(agent_name, agent_memory)

    def record_success(self, agent_name: str, payload: dict[str, Any]) -> None:
        agent_memory = self.load_agent(agent_name)
        agent_memory.setdefault("wins", []).append(payload)
        self.save_agent(agent_name, agent_memory)

    def remember_experiment(self, payload: dict[str, Any]) -> None:
        global_memory = self.load_global()
        global_memory.setdefault("history", []).append(payload)
        self.save_global(global_memory)
