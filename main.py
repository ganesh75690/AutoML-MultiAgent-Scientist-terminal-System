from __future__ import annotations

from pathlib import Path

import typer

from agents.supervisor import SupervisorAgent
from utils.config import build_config
from utils.logger import setup_enterprise_logging


def main(dataset_path: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False)) -> None:
    """Run the complete AutoML workflow from a single CSV file."""

    config = build_config(Path(__file__).resolve().parent)
    loggers = setup_enterprise_logging(config.logs_dir)
    supervisor = SupervisorAgent(config=config, loggers=loggers)
    supervisor.run(dataset_path.resolve())


if __name__ == "__main__":
    typer.run(main)
