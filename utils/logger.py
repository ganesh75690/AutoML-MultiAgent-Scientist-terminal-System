from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass

from rich.console import Console
from rich.logging import RichHandler


@dataclass(slots=True)
class EnterpriseLoggers:
    workflow: logging.Logger
    agent: logging.Logger
    error: logging.Logger
    performance: logging.Logger
    audit: logging.Logger


def setup_enterprise_logging(log_dir: Path, name: str = "automl_scientist") -> EnterpriseLoggers:
    """Configure multiple enterprise log channels and a rich console logger."""

    log_dir.mkdir(parents=True, exist_ok=True)
    workflow_logger = logging.getLogger(name)
    if workflow_logger.handlers:
        return EnterpriseLoggers(
            workflow=workflow_logger,
            agent=logging.getLogger(f"{name}.agent"),
            error=logging.getLogger(f"{name}.error"),
            performance=logging.getLogger(f"{name}.performance"),
            audit=logging.getLogger(f"{name}.audit"),
        )

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console_handler = RichHandler(console=Console(), rich_tracebacks=True, show_path=False)
    console_handler.setLevel(logging.INFO)

    def _build_logger(suffix: str, filename: str, level: int = logging.INFO, console: bool = False) -> logging.Logger:
        logger = logging.getLogger(f"{name}.{suffix}")
        logger.setLevel(level)
        logger.propagate = False
        file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if console:
            logger.addHandler(console_handler)
        return logger

    workflow_logger = _build_logger("workflow", "workflow.log", console=True)
    agent_logger = _build_logger("agent", "agent.log")
    error_logger = _build_logger("error", "error.log", level=logging.ERROR)
    performance_logger = _build_logger("performance", "performance.log")
    audit_logger = _build_logger("audit", "audit.log")

    return EnterpriseLoggers(
        workflow=workflow_logger,
        agent=agent_logger,
        error=error_logger,
        performance=performance_logger,
        audit=audit_logger,
    )


def setup_logger(name: str = "automl_scientist", log_file: Path | None = None) -> logging.Logger:
    """Backward-compatible logger setup used by earlier entrypoints."""

    if log_file is not None:
        log_dir = log_file.parent
        return setup_enterprise_logging(log_dir, name=name).workflow
    return setup_enterprise_logging(Path.cwd() / "logs", name=name).workflow
