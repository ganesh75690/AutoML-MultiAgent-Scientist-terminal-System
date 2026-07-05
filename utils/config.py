from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


@dataclass(slots=True)
class AppConfig:
    """Application-wide settings and artifact paths."""

    project_root: Path
    configs_dir: Path = field(init=False)
    datasets_dir: Path = field(init=False)
    docs_dir: Path = field(init=False)
    experiments_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    memory_dir: Path = field(init=False)
    plugins_dir: Path = field(init=False)
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = -1
    max_tuning_iterations: int = 12
    allow_polynomial_features: bool = True
    max_polynomial_numeric_features: int = 4
    outlier_clip_iqr: float = 1.5
    missing_column_threshold: float = 0.95
    resume: bool = True
    use_gpu: bool = field(default=False)
    artifacts_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.configs_dir = self.project_root / "configs"
        self.datasets_dir = self.project_root / "datasets"
        self.docs_dir = self.project_root / "docs"
        self.experiments_dir = self.project_root / "experiments"
        self.logs_dir = self.project_root / "logs"
        self.memory_dir = self.project_root / "memory"
        self.plugins_dir = self.project_root / "plugins"
        self.artifacts_dir = self.project_root
        self.models_dir = self.project_root / "models"
        self.reports_dir = self.project_root / "reports"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.predictions_dir = self.reports_dir / "predictions"
        for directory in (
            self.configs_dir,
            self.datasets_dir,
            self.docs_dir,
            self.experiments_dir,
            self.logs_dir,
            self.memory_dir,
            self.plugins_dir,
            self.models_dir,
            self.reports_dir,
            self.checkpoints_dir,
            self.predictions_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def detect_gpu_available() -> bool:
    """Best-effort GPU detection without introducing extra dependencies."""

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def build_config(project_root: Path | None = None) -> AppConfig:
    """Create a configuration object with sane defaults for this project."""

    root = project_root or Path(__file__).resolve().parents[1]
    config = AppConfig(project_root=root, use_gpu=detect_gpu_available())
    defaults = _load_defaults(config.configs_dir / "default.json")
    for key, value in defaults.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def _load_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}
