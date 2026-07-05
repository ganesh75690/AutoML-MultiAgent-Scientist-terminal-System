from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils.helper import now_utc


@dataclass(slots=True)
class ReportArtifacts:
    markdown_path: Path
    pdf_path: Path


class ReportAgent:
    """Assemble the final human-readable and presentation-friendly reports."""

    def generate(self, summary: dict[str, Any], output_dir: Path, docs_dir: Path) -> ReportArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = output_dir / "report.md"
        pdf_path = output_dir / "report.pdf"
        docs_bundle_dir = docs_dir / summary.get("experiment_id", output_dir.name)
        docs_bundle_dir.mkdir(parents=True, exist_ok=True)

        markdown_path.write_text(self._build_markdown(summary), encoding="utf-8")
        self._build_pdf(summary, pdf_path)
        self._write_docs_bundle(summary, docs_bundle_dir)
        return ReportArtifacts(markdown_path=markdown_path, pdf_path=pdf_path)

    def _build_markdown(self, summary: dict[str, Any]) -> str:
        lines = [
            "# AutoML Multi-Agent Scientist Report",
            "",
            f"Generated at: {now_utc()}",
            "",
            "## Dataset Summary",
            self._kv_block(summary.get("dataset_summary", {})),
            "",
            "## Cleaning Summary",
            self._kv_block(summary.get("cleaning_summary", {})),
            "",
            "## Intelligent Dataset Understanding",
            self._kv_block(summary.get("intelligence", {})),
            "",
            "## Planning",
            self._kv_block(summary.get("plan", {})),
            "",
            "## Risk Assessment",
            self._kv_block(summary.get("risk_assessment", {})),
            "",
            "## EDA Summary",
            self._kv_block(summary.get("eda_summary", {})),
            "",
            "## Feature Engineering",
            self._kv_block(summary.get("feature_summary", {})),
            "",
            "## Model Comparison",
            self._table_block(summary.get("model_comparison", [])),
            "",
            "## Debate and Tournament",
            self._kv_block(summary.get("debate", {})),
            self._kv_block(summary.get("tournament", {})),
            "",
            "## Best Model",
            self._kv_block(summary.get("best_model", {})),
            "",
            "## Evaluation Metrics",
            self._kv_block(summary.get("metrics", {})),
            "",
            "## Explainability",
            self._kv_block(summary.get("explainability", {})),
            "",
            "## Critic Review",
            self._kv_block(summary.get("critic", {})),
            "",
            "## Mentor Notes",
            self._mentor_block(summary.get("mentor", [])),
            "",
            "## Communication Log",
            self._communication_block(summary.get("communication_log", [])),
            "",
            "## Decision Timeline",
            self._timeline_block(summary.get("decision_timeline", [])),
            "",
            "## Recommendations",
            summary.get("recommendations", "Further validation and production monitoring are recommended."),
            "",
        ]
        return "\n".join(lines)

    def _build_pdf(self, summary: dict[str, Any], pdf_path: Path) -> None:
        with PdfPages(pdf_path) as pdf:
            self._page_with_text(
                pdf,
                "AutoML Multi-Agent Scientist",
                [
                    f"Generated at: {now_utc()}",
                    f"Target: {summary.get('dataset_summary', {}).get('target', 'unknown')}",
                    f"Task type: {summary.get('dataset_summary', {}).get('task_type', 'unknown')}",
                    f"Best model: {summary.get('best_model', {}).get('name', 'unknown')}",
                ],
            )
            self._page_with_text(pdf, "Cleaning and EDA", self._dict_lines(summary.get("cleaning_summary", {})) + self._dict_lines(summary.get("eda_summary", {})))
            self._page_with_text(pdf, "Model Comparison", self._comparison_lines(summary.get("model_comparison", [])))
            self._page_with_text(pdf, "Metrics and Explainability", self._dict_lines(summary.get("metrics", {})) + self._dict_lines(summary.get("explainability", {})))

    def _page_with_text(self, pdf: PdfPages, title: str, lines: list[str]) -> None:
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
        text = "\n".join(lines) if lines else "No details available."
        figure.text(0.08, 0.92, text, va="top", ha="left", fontsize=10, family="monospace")
        plt.axis("off")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    def _dict_lines(self, data: dict[str, Any]) -> list[str]:
        return [f"{key}: {value}" for key, value in data.items()]

    def _comparison_lines(self, rows: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for row in rows[:10]:
            lines.append(
                f"{row.get('name')}: mean={row.get('mean_score')}, std={row.get('std_score')}, fit={row.get('fit_status', 'ok')}"
            )
        return lines

    def _mentor_block(self, notes: list[dict[str, Any]] | list[Any]) -> str:
        if not notes:
            return "No mentor notes available."
        lines = []
        for note in notes:
            if isinstance(note, dict):
                lines.append(f"- {note.get('title')}: {note.get('explanation')}")
            else:
                lines.append(f"- {getattr(note, 'title', 'Note')}: {getattr(note, 'explanation', '')}")
        return "\n".join(lines)

    def _communication_block(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return "No agent communication recorded."
        return "\n".join(
            f"- {item.get('sender')} -> {item.get('recipient')}: {item.get('message')} (confidence {item.get('confidence')})"
            for item in messages
        )

    def _timeline_block(self, timeline: list[str]) -> str:
        if not timeline:
            return "No timeline available."
        return "\n".join(f"- {item}" for item in timeline)

    def _write_docs_bundle(self, summary: dict[str, Any], docs_bundle_dir: Path) -> None:
        model_card = docs_bundle_dir / "model_card.md"
        data_card = docs_bundle_dir / "data_card.md"
        experiment_summary = docs_bundle_dir / "experiment_summary.md"
        decision_log = docs_bundle_dir / "decision_log.md"
        executive_summary = docs_bundle_dir / "executive_summary.md"
        readme = docs_bundle_dir / "README.md"

        model_card.write_text(self._model_card(summary), encoding="utf-8")
        data_card.write_text(self._data_card(summary), encoding="utf-8")
        experiment_summary.write_text(self._experiment_summary(summary), encoding="utf-8")
        decision_log.write_text(self._decision_log(summary), encoding="utf-8")
        executive_summary.write_text(self._executive_summary(summary), encoding="utf-8")
        readme.write_text(self._executive_summary(summary), encoding="utf-8")

    def _model_card(self, summary: dict[str, Any]) -> str:
        best_model = summary.get("best_model", {})
        return "\n".join(
            [
                "# Model Card",
                f"- Best model: {best_model.get('name', 'unknown')}",
                f"- Metrics: {summary.get('metrics', {})}",
                f"- Explainability: {summary.get('explainability', {})}",
                f"- Risks: {summary.get('risk_assessment', {})}",
            ]
        )

    def _data_card(self, summary: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Data Card",
                f"- Dataset summary: {summary.get('dataset_summary', {})}",
                f"- Intelligent understanding: {summary.get('intelligence', {})}",
                f"- Cleaning summary: {summary.get('cleaning_summary', {})}",
            ]
        )

    def _experiment_summary(self, summary: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Experiment Summary",
                f"- Experiment ID: {summary.get('experiment_id')}",
                f"- Execution time: {summary.get('execution_seconds')} seconds",
                f"- Winner: {summary.get('best_model', {}).get('name')}",
                f"- Primary metric: {summary.get('metrics', {})}",
            ]
        )

    def _decision_log(self, summary: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Decision Log",
                f"- Plan: {summary.get('plan', {})}",
                f"- Debate: {summary.get('debate', {})}",
                f"- Tournament: {summary.get('tournament', {})}",
                f"- Critic review: {summary.get('critic', {})}",
            ]
        )

    def _executive_summary(self, summary: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Executive Summary",
                f"The workflow completed using {summary.get('best_model', {}).get('name', 'the selected model')}.",
                f"The run was scored with {summary.get('metrics', {})}.",
                f"Recommendations: {summary.get('recommendations', '')}",
            ]
        )

    def _kv_block(self, data: dict[str, Any]) -> str:
        if not data:
            return "No data available."
        return "\n".join(f"- {key}: {value}" for key, value in data.items())

    def _table_block(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No model comparison results available."
        header = "| Model | Mean Score | Std | Status |\n| --- | ---: | ---: | --- |"
        body = []
        for row in rows[:12]:
            body.append(
                f"| {row.get('name')} | {row.get('mean_score')} | {row.get('std_score')} | {row.get('fit_status', 'ok')} |"
            )
        return "\n".join([header, *body])
