from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from utils.system import gather_system_stats


@dataclass(slots=True)
class DashboardSnapshot:
    title: str
    task: str
    active_model: str
    dataset_summary: dict[str, Any]
    success_rate: float


class TerminalDashboard:
    """Render the workflow state in a professional terminal dashboard."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def render(self, snapshot: DashboardSnapshot, agent_activity: list[str]) -> None:
        stats = gather_system_stats()
        table = Table(title="Live System Snapshot")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("CPU Usage", f"{stats.cpu_usage_percent:.1f}%")
        table.add_row("RAM Usage", f"{stats.memory_usage_percent:.1f}%")
        table.add_row("RAM Available", f"{stats.available_memory_gb:.2f} GB / {stats.total_memory_gb:.2f} GB")
        table.add_row("CPU Cores", str(stats.cpu_cores))
        table.add_row("Current Task", snapshot.task)
        table.add_row("Active Model", snapshot.active_model)
        table.add_row("Success Rate", f"{snapshot.success_rate:.1f}%")
        table.add_row("Rows", str(snapshot.dataset_summary.get("rows", "n/a")))
        table.add_row("Columns", str(snapshot.dataset_summary.get("columns", "n/a")))

        self.console.print(Panel.fit(f"[bold]{snapshot.title}[/bold]\n{snapshot.task}", border_style="cyan"))
        self.console.print(table)
        if agent_activity:
            activity_table = Table(title="Agent Activity")
            activity_table.add_column("Event", style="green")
            for item in agent_activity[-8:]:
                activity_table.add_row(item)
            self.console.print(activity_table)
