from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import joblib
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from agents.cleaning import CleaningResult, DataCleaningAgent
from agents.eda import EDAResult, ExploratoryDataAnalysisAgent
from agents.evaluation import EvaluationResult, EvaluationAgent
from agents.explainability import ExplainabilityResult, ExplainabilityAgent
from agents.feature_engineering import FeatureEngineeringResult, FeatureEngineeringAgent
from agents.model_selection import ModelSelectionResult, ModelSelectionAgent
from agents.report import ReportAgent
from agents.tuning import HyperparameterTuningAgent, TuningResult
from utils.config import AppConfig
from utils.helper import (
    analyze_dataset_intelligence,
    classification_scoring,
    dataset_signature,
    detect_task_type,
    feature_profile,
    infer_target_column,
    load_dataset,
    load_json,
    now_utc,
    save_json,
    summarize_dataset,
    train_test_split_data,
)
from utils.system import gather_system_stats
from workflows.communication import CommunicationHub
from workflows.critic import AICriticAgent
from workflows.dashboard import DashboardSnapshot, TerminalDashboard
from workflows.dynamic_agents import DynamicAgentFactory
from workflows.experiment_tracker import ExperimentTracker
from workflows.memory import AgentMemoryStore
from workflows.mentor import AIMentorAgent
from workflows.planning import PlanningEngine
from workflows.plugin_loader import PluginLoader
from workflows.risk import RiskAssessmentAgent
from workflows.tournament import ModelTournament


@dataclass(slots=True)
class WorkflowContext:
    dataset_path: Path
    experiment_id: str
    checkpoint_dir: Path
    state_path: Path
    cleaned_path: Path
    artifacts_path: Path
    docs_path: Path


class SupervisorAgent:
    """Enterprise orchestration layer for the AutoML platform."""

    def __init__(self, config: AppConfig, loggers: Any) -> None:
        self.config = config
        self.loggers = loggers
        self.logger = getattr(loggers, "workflow", loggers)
        self.agent_logger = getattr(loggers, "agent", self.logger)
        self.error_logger = getattr(loggers, "error", self.logger)
        self.performance_logger = getattr(loggers, "performance", self.logger)
        self.audit_logger = getattr(loggers, "audit", self.logger)
        self.console = Console()
        self.dashboard = TerminalDashboard(self.console)
        self.cleaning_agent = DataCleaningAgent(config)
        self.eda_agent = ExploratoryDataAnalysisAgent()
        self.feature_agent = FeatureEngineeringAgent(config)
        self.model_agent = ModelSelectionAgent(config)
        self.tuning_agent = HyperparameterTuningAgent(config)
        self.evaluation_agent = EvaluationAgent()
        self.explainability_agent = ExplainabilityAgent(config)
        self.report_agent = ReportAgent()
        self.planning_engine = PlanningEngine()
        self.communication = CommunicationHub()
        self.memory_store = AgentMemoryStore(config.memory_dir)
        self.experiment_tracker = ExperimentTracker(config.experiments_dir)
        self.dynamic_factory = DynamicAgentFactory()
        self.risk_agent = RiskAssessmentAgent()
        self.critic_agent = AICriticAgent()
        self.mentor_agent = AIMentorAgent()
        self.plugin_loader = PluginLoader(config.plugins_dir)
        self.model_tournament = ModelTournament()

    def run(self, dataset_path: Path) -> dict[str, Any]:
        start_time = perf_counter()
        raw_df = load_dataset(dataset_path)
        target_column = infer_target_column(raw_df)
        task_type = detect_task_type(raw_df[target_column])
        profile = feature_profile(raw_df, target_column)
        intelligence = analyze_dataset_intelligence(raw_df, target_column)
        dataset_summary = summarize_dataset(raw_df, target_column, task_type)
        dataset_summary["feature_columns"] = profile.feature_columns
        dataset_summary["quality_score"] = intelligence.quality_score
        dataset_summary["business_domain"] = intelligence.business_domain
        dataset_summary["industry"] = intelligence.industry
        dataset_summary["problem_type"] = intelligence.problem_type
        dataset_summary["pii_columns"] = intelligence.pii_columns
        dataset_summary["sensitive_columns"] = intelligence.sensitive_columns
        dataset_summary["duplicate_patterns"] = intelligence.duplicate_patterns
        dataset_summary["missing_information"] = intelligence.missing_information

        risk_profile = self.risk_agent.assess(dataset_summary, target_column, profile.feature_columns)
        dynamic_agents = self.dynamic_factory.create(dataset_summary, {"risk_score": risk_profile.risk_score})
        plan = self.planning_engine.build_plan(dataset_summary, {"risk_score": risk_profile.risk_score}, [agent.name for agent in dynamic_agents])
        plugin_registry = self.plugin_loader.load()

        experiment = self.experiment_tracker.start(dataset_signature(dataset_path))
        context = self._build_context(dataset_path, experiment.experiment_id)
        context.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        context.artifacts_path.mkdir(parents=True, exist_ok=True)
        context.docs_path.mkdir(parents=True, exist_ok=True)

        global_memory = self.memory_store.load_global()
        reused_models = global_memory.get("successful_models", [])
        if reused_models:
            self.communication.send("Supervisor", "Model Selection Agent", "Reuse prior good priors if relevant.", 0.73, "Previous model memories are available.")

        self.console.rule("[bold cyan]AutoML Multi-Agent Scientist")
        self._print_plan(plan, risk_profile, dynamic_agents, plugin_registry.plugins)
        self._render_dashboard("Planning complete", "N/A", dataset_summary, 0.0)

        self.logger.info("[Supervisor] Starting workflow for %s", dataset_path.name)
        self.audit_logger.info("Experiment %s started for dataset %s", experiment.experiment_id, dataset_path.name)
        self.communication.record_timeline(f"{now_utc()} Dataset loaded")

        state = self._load_state(context.state_path)
        if state.get("complete") and context.artifacts_path.joinpath("report.md").exists() and context.artifacts_path.joinpath("report.pdf").exists():
            self.console.print("[green]Existing completed run detected. Reusing saved artifacts.[/green]")
            return state

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        task = progress.add_task("Starting workflow", total=8)

        with progress:
            cleaned_result = self._execute_with_recovery(
                "Cleaning Agent",
                lambda: self.cleaning_agent.run(raw_df, target_column),
                lambda: DataCleaningAgent(self.config),
                "cleaning_agent",
            )
            self._save_cleaned_dataset(cleaned_result, context)
            progress.update(task, advance=1, description="Cleaning complete")
            self.communication.send("Supervisor", "Cleaning Agent", "Remove duplicates, fill gaps, and clip outliers.", 0.97, "Focus on stable tabular preprocessing.")
            self.communication.record_timeline(f"{now_utc()} Cleaning complete")

            eda_result = self._execute_with_recovery(
                "EDA Agent",
                lambda: self.eda_agent.analyze(cleaned_result.cleaned_data, target_column, task_type),
                lambda: ExploratoryDataAnalysisAgent(),
                "eda_agent",
            )
            self._save_json_artifact(context.artifacts_path / "eda_summary.json", eda_result.summary)
            progress.update(task, advance=1, description="EDA complete")
            self._log_table("EDA Agent", eda_result.summary.get("shape", {}))
            self.communication.send("EDA Agent", "Supervisor", f"Target '{target_column}' and {task_type} problem confirmed.", 0.94, "Dataset shape and distributions inspected.")
            self.communication.record_timeline(f"{now_utc()} EDA complete")

            X = cleaned_result.cleaned_data.drop(columns=[target_column])
            y = cleaned_result.cleaned_data[target_column]
            split = train_test_split_data(X, y, task_type, self.config.test_size, self.config.random_seed)

            feature_result = self._execute_with_recovery(
                "Feature Engineering Agent",
                lambda: self.feature_agent.transform(split.X_train, target_column, task_type),
                lambda: FeatureEngineeringAgent(self.config),
                "feature_agent",
            )
            self._save_json_artifact(context.artifacts_path / "feature_summary.json", feature_result.summary)
            progress.update(task, advance=1, description="Feature engineering complete")
            self.communication.send("Feature Engineering Agent", "Supervisor", "Encode categoricals, scale numerics, and remove low-signal columns.", 0.93, "Prepared reusable preprocessing pipeline.")
            self.communication.record_timeline(f"{now_utc()} Feature engineering complete")

            scoring = classification_scoring(split.y_train) if task_type == "classification" else "r2"
            selection_result = self._execute_with_recovery(
                "Model Selection Agent",
                lambda: self.model_agent.compare_models(
                    split.X_train,
                    split.y_train,
                    feature_result.selected_columns,
                    feature_result.preprocessor,
                    task_type,
                    scoring,
                ),
                lambda: ModelSelectionAgent(self.config),
                "model_agent",
            )
            model_rows = [
                {
                    "name": item.name,
                    "mean_score": round(item.mean_score, 6),
                    "std_score": round(item.std_score, 6),
                    "fit_status": "ok",
                }
                for item in selection_result.results
            ]
            self._write_dataframe(pd.DataFrame(model_rows), context.artifacts_path / "model_comparison.csv")
            progress.update(task, advance=1, description="Model comparison complete")

            debate = self._run_debate(selection_result)
            tournament = self.model_tournament.run(
                [
                    {"name": item.name, "score": item.mean_score, "confidence": max(0.55, min(0.99, item.mean_score if item.mean_score <= 1 else item.mean_score / 100))}
                    for item in selection_result.results[:8]
                ]
            )
            final_winner = tournament.champion if tournament.champion != "Unknown" else selection_result.best_candidate.name
            winner_candidate = next((item for item in selection_result.results if item.name == final_winner), selection_result.best_candidate)
            self.communication.send("Supervisor", "Model Tournament", f"Champion model selected: {final_winner}", 0.96, "Used debate and knockout evidence to pick the final champion.")
            self.communication.record_timeline(f"{now_utc()} Model tournament complete")
            self._render_dashboard("Model tournament complete", final_winner, dataset_summary, 50.0)

            tuning_result = self._execute_with_recovery(
                "Tuning Agent",
                lambda: self.tuning_agent.tune(
                    split.X_train,
                    split.y_train,
                    feature_result.selected_columns,
                    feature_result.preprocessor,
                    winner_candidate.estimator,
                    task_type,
                    scoring,
                ),
                lambda: HyperparameterTuningAgent(self.config),
                "tuning_agent",
            )
            progress.update(task, advance=1, description="Tuning complete")
            self.communication.send("Tuning Agent", "Supervisor", f"Best params found using {tuning_result.search_strategy}.", 0.9, "Search space optimized around the champion model.")
            self.communication.record_timeline(f"{now_utc()} Tuning complete")

            tuned_model = tuning_result.tuned_pipeline
            model_path = context.checkpoint_dir / "best_model.joblib"
            joblib.dump(tuned_model, model_path)
            self._save_json_artifact(
                context.artifacts_path / "best_model.json",
                {
                    "name": final_winner,
                    "best_score": selection_result.best_candidate.mean_score,
                    "tuning_score": tuning_result.best_score,
                    "best_params": tuning_result.best_params,
                    "search_strategy": tuning_result.search_strategy,
                    "model_path": str(model_path),
                },
            )

            evaluation_result = self._execute_with_recovery(
                "Evaluation Agent",
                lambda: self.evaluation_agent.evaluate(
                    tuned_model,
                    split.X_test[feature_result.selected_columns],
                    split.y_test,
                    task_type,
                ),
                lambda: EvaluationAgent(),
                "evaluation_agent",
            )
            self._save_json_artifact(context.artifacts_path / "metrics.json", evaluation_result.metrics)
            self._write_dataframe(evaluation_result.predictions, context.artifacts_path / "predictions.csv")
            if evaluation_result.confusion is not None:
                self._save_json_artifact(context.artifacts_path / "confusion_matrix.json", {"matrix": evaluation_result.confusion})
            progress.update(task, advance=1, description="Evaluation complete")
            self.communication.send("Evaluation Agent", "Supervisor", self._evaluation_summary(evaluation_result), 0.95, "Holdout validation completed.")
            self.communication.record_timeline(f"{now_utc()} Evaluation complete")

            explainability_result = self._execute_with_recovery(
                "Explainability Agent",
                lambda: self.explainability_agent.explain(
                    tuned_model,
                    split.X_train,
                    feature_result.selected_columns,
                    context.artifacts_path,
                    y_reference=split.y_train.loc[split.X_train.index],
                ),
                lambda: ExplainabilityAgent(self.config),
                "explainability_agent",
            )
            self._save_json_artifact(
                context.artifacts_path / "explainability.json",
                {
                    "method": explainability_result.method,
                    "plot_path": str(explainability_result.plot_path),
                    "top_features": explainability_result.top_features,
                    "artifacts": explainability_result.artifacts,
                    "summary": explainability_result.summary,
                },
            )
            progress.update(task, advance=1, description="Explainability complete")
            self.communication.send("Explainability Agent", "Supervisor", "Generated SHAP, permutation, correlation, PDP, and local surrogate explanations.", 0.92, "Multiple explanation modes now available.")
            self.communication.record_timeline(f"{now_utc()} Explainability complete")

            critic_result = self.critic_agent.review(
                {
                    "metrics": evaluation_result.metrics,
                    "best_model": {"name": final_winner},
                    "risk_assessment": {"risk_score": risk_profile.risk_score},
                    "task_type": task_type,
                    "explainability": explainability_result.summary,
                }
            )
            mentor_notes = self.mentor_agent.teach({"best_model": {"name": final_winner}, "metrics": evaluation_result.metrics, "task_type": task_type})
            progress.update(task, advance=1, description="Critic and mentor complete")
            self.communication.send("Critic Agent", "Supervisor", "Workflow review completed.", 0.9, critic_result.severity)
            self.communication.send("Mentor Agent", "User", mentor_notes[0].explanation if mentor_notes else "Review complete.", 0.91, "Simple explanation prepared.")
            self.communication.record_timeline(f"{now_utc()} Review complete")

        execution_seconds = round(perf_counter() - start_time, 2)
        final_summary = {
            "experiment_id": experiment.experiment_id,
            "dataset_summary": dataset_summary,
            "intelligence": {
                "problem_type": intelligence.problem_type,
                "business_domain": intelligence.business_domain,
                "industry": intelligence.industry,
                "quality_score": intelligence.quality_score,
                "pii_columns": intelligence.pii_columns,
                "sensitive_columns": intelligence.sensitive_columns,
                "duplicate_patterns": intelligence.duplicate_patterns,
                "missing_information": intelligence.missing_information,
                "notes": intelligence.notes,
            },
            "risk_assessment": {
                "risk_score": risk_profile.risk_score,
                "quality_score": risk_profile.quality_score,
                "findings": risk_profile.findings,
                "pii_columns": risk_profile.pii_columns,
                "sensitive_columns": risk_profile.sensitive_columns,
            },
            "plan": {
                "goal": plan.goal,
                "required_agents": plan.required_agents,
                "estimated_time_minutes": plan.estimated_time_minutes,
                "complexity": plan.complexity,
                "memory_estimate_mb": plan.memory_estimate_mb,
                "risk_level": plan.risk_level,
                "steps": plan.steps,
                "notes": plan.notes,
            },
            "dynamic_agents": [asdict(agent) for agent in dynamic_agents],
            "plugin_contributions": plugin_registry.plugins,
            "cleaning_summary": cleaned_result.summary,
            "eda_summary": eda_result.summary,
            "feature_summary": feature_result.summary,
            "model_comparison": model_rows,
            "debate": debate,
            "tournament": {
                "champion": tournament.champion,
                "matches": [asdict(match) for match in tournament.matches],
            },
            "best_model": {
                "name": final_winner,
                "cv_score": round(selection_result.best_candidate.mean_score, 6),
                "tuning_score": round(tuning_result.best_score, 6) if tuning_result.best_score == tuning_result.best_score else None,
                "best_params": tuning_result.best_params,
            },
            "metrics": evaluation_result.metrics,
            "explainability": {
                "method": explainability_result.method,
                "top_features": explainability_result.top_features,
                "plot_path": str(explainability_result.plot_path),
                "artifacts": explainability_result.artifacts,
                "summary": explainability_result.summary,
            },
            "critic": {"issues": critic_result.issues, "suggestions": critic_result.suggestions, "severity": critic_result.severity},
            "mentor": [asdict(note) for note in mentor_notes],
            "communication_log": [asdict(message) for message in self.communication.messages],
            "decision_timeline": list(self.communication.decision_timeline),
            "recommendations": self._recommendations(task_type, evaluation_result.metrics, explainability_result.top_features),
            "execution_seconds": execution_seconds,
            "task_type": task_type,
            "target_column": target_column,
        }
        report_artifacts = self.report_agent.generate(final_summary, context.artifacts_path, self.config.docs_dir)
        final_summary["report"] = {
            "markdown": str(report_artifacts.markdown_path),
            "pdf": str(report_artifacts.pdf_path),
        }
        final_summary["complete"] = True
        final_summary["dataset_signature"] = dataset_signature(dataset_path)
        self._save_state(context.state_path, final_summary)
        self._save_json_artifact(context.artifacts_path / "workflow_summary.json", final_summary)
        self.experiment_tracker.finalize(
            experiment,
            {
                "experiment_id": experiment.experiment_id,
                "dataset_signature": final_summary["dataset_signature"],
                "dataset_path": str(dataset_path),
                "target_column": target_column,
                "task_type": task_type,
                "winner": final_winner,
                "primary_metric": self._primary_metric(evaluation_result.metrics, task_type),
                "execution_seconds": execution_seconds,
                "timestamp": now_utc(),
                "model_comparison": model_rows,
                "parameters": tuning_result.best_params,
            },
        )
        self._update_memory(final_summary, tuning_result, dataset_path)
        self._write_enterprise_logs(final_summary)
        self.console.print(Panel.fit("[green]Workflow complete.[/green]", title="Status"))
        self.logger.info("[Supervisor] Workflow complete in %.2f seconds", execution_seconds)
        self.performance_logger.info("Experiment %s completed in %.2f seconds", experiment.experiment_id, execution_seconds)
        return final_summary

    def _build_context(self, dataset_path: Path, experiment_id: str) -> WorkflowContext:
        checkpoint_dir = self.config.checkpoints_dir / f"{dataset_path.stem}_{experiment_id}"
        artifacts_path = self.config.reports_dir / f"{dataset_path.stem}_{experiment_id}"
        docs_path = self.config.docs_dir
        return WorkflowContext(
            dataset_path=dataset_path,
            experiment_id=experiment_id,
            checkpoint_dir=checkpoint_dir,
            state_path=checkpoint_dir / "state.json",
            cleaned_path=checkpoint_dir / "cleaned.csv",
            artifacts_path=artifacts_path,
            docs_path=docs_path,
        )

    def _execute_with_recovery(self, agent_name: str, action: Callable[[], Any], reset_factory: Callable[[], Any], attribute_name: str | None = None) -> Any:
        try:
            return action()
        except Exception as first_error:
            self.error_logger.exception("%s failed once; retrying with a fresh instance", agent_name)
            self.memory_store.record_failure(agent_name, str(first_error))
            self.communication.send("Supervisor", agent_name, f"Retry requested after failure: {first_error}", 0.51, "Self-healing workflow activated.")
            try:
                replacement = reset_factory()
                if attribute_name is not None:
                    setattr(self, attribute_name, replacement)
                result = action()
                self.memory_store.record_success(agent_name, {"status": "recovered"})
                return result
            except Exception as second_error:
                self.error_logger.exception("%s failed after retry", agent_name)
                self.memory_store.record_failure(agent_name, str(second_error))
                raise

    def _run_debate(self, selection_result: ModelSelectionResult) -> dict[str, Any]:
        top_candidates = selection_result.results[:3]
        opinions: list[dict[str, Any]] = []
        labels = ["Model Agent A", "Model Agent B", "Model Agent C"]
        for index, candidate in enumerate(top_candidates):
            vote = 3 - index
            explanation = f"{candidate.name} produced a mean score of {candidate.mean_score:.4f} with stability {candidate.std_score:.4f}."
            opinions.append(
                {
                    "agent_name": labels[index],
                    "model_name": candidate.name,
                    "confidence": max(0.55, min(0.99, candidate.mean_score if candidate.mean_score <= 1 else candidate.mean_score / 100)),
                    "vote": vote,
                    "explanation": explanation,
                }
            )
            self.communication.send(labels[index], "Supervisor", explanation, opinions[-1]["confidence"], "Debate contribution")
        outcome = self.communication.debate(opinions)
        return {
            "winner": outcome.winner,
            "votes": outcome.votes,
            "confidence": outcome.confidence,
            "explanation": outcome.explanation,
            "opinions": opinions,
        }

    def _save_cleaned_dataset(self, cleaned_result: CleaningResult, context: WorkflowContext) -> None:
        context.cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_result.cleaned_data.to_csv(context.cleaned_path, index=False)
        self._save_json_artifact(context.artifacts_path / "cleaning_summary.json", cleaned_result.summary)

    def _write_enterprise_logs(self, summary: dict[str, Any]) -> None:
        self.audit_logger.info("Decision timeline: %s", summary.get("decision_timeline", []))
        self.audit_logger.info("Communication log count: %s", len(summary.get("communication_log", [])))
        self.performance_logger.info("Winner: %s", summary.get("best_model", {}).get("name"))

    def _update_memory(self, summary: dict[str, Any], tuning_result: TuningResult, dataset_path: Path) -> None:
        global_memory = self.memory_store.load_global()
        global_memory.setdefault("successful_models", [])
        global_memory.setdefault("dataset_history", [])
        global_memory.setdefault("best_hyperparameters", [])
        global_memory["successful_models"].append(
            {
                "dataset_signature": summary.get("dataset_signature"),
                "winner": summary.get("best_model", {}).get("name"),
                "metric": summary.get("metrics", {}),
            }
        )
        global_memory["dataset_history"].append(
            {
                "dataset": str(dataset_path),
                "task_type": summary.get("task_type"),
                "quality_score": summary.get("intelligence", {}).get("quality_score"),
            }
        )
        global_memory["best_hyperparameters"].append(tuning_result.best_params)
        self.memory_store.save_global(global_memory)
        self.memory_store.save_agent("Supervisor Agent", {"last_run": summary.get("dataset_signature"), "winner": summary.get("best_model", {}).get("name")})

    def _render_dashboard(self, current_task: str, active_model: str, dataset_summary: dict[str, Any], success_rate: float) -> None:
        snapshot = DashboardSnapshot(
            title="AutoML Multi-Agent Scientist",
            task=current_task,
            active_model=active_model,
            dataset_summary=dataset_summary,
            success_rate=success_rate,
        )
        self.dashboard.render(snapshot, [f"{message.sender} -> {message.recipient}: {message.message}" for message in self.communication.messages])

    def _print_plan(self, plan: Any, risk_profile: Any, dynamic_agents: list[Any], plugins: list[dict[str, Any]]) -> None:
        plan_table = Table(title="AI Planning Engine")
        plan_table.add_column("Field", style="cyan")
        plan_table.add_column("Value", style="magenta")
        plan_table.add_row("Goal", plan.goal)
        plan_table.add_row("Agents", ", ".join(plan.required_agents))
        plan_table.add_row("Estimated Time", f"{plan.estimated_time_minutes} minutes")
        plan_table.add_row("Complexity", plan.complexity)
        plan_table.add_row("Memory Estimate", f"{plan.memory_estimate_mb} MB")
        plan_table.add_row("Risk Level", plan.risk_level)
        plan_table.add_row("Dataset Risk Score", f"{risk_profile.risk_score}")
        plan_table.add_row("Quality Score", f"{risk_profile.quality_score}")
        self.console.print(plan_table)

        if dynamic_agents:
            specialist_table = Table(title="Temporary Specialist Agents")
            specialist_table.add_column("Agent", style="green")
            specialist_table.add_column("Confidence", style="yellow")
            specialist_table.add_column("Explanation", style="white")
            for agent in dynamic_agents:
                specialist_table.add_row(agent.name, f"{agent.confidence:.2f}", agent.explanation)
            self.console.print(specialist_table)

        if plugins:
            plugin_table = Table(title="Loaded Plugins")
            plugin_table.add_column("Plugin", style="green")
            plugin_table.add_column("Description", style="white")
            for plugin in plugins:
                plugin_table.add_row(str(plugin.get("plugin_name")), str(plugin.get("description", "")))
            self.console.print(plugin_table)

    def _save_json_artifact(self, path: Path, data: dict[str, Any]) -> None:
        save_json(data, path)

    def _load_state(self, path: Path) -> dict[str, Any]:
        if path.exists():
            return load_json(path)
        return {}

    def _save_state(self, path: Path, data: dict[str, Any]) -> None:
        save_json(data, path)

    def _write_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def _log_table(self, title: str, values: dict[str, Any]) -> None:
        table = Table(title=title)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="magenta")
        for key, value in values.items():
            table.add_row(str(key), str(value))
        self.console.print(table)

    def _evaluation_summary(self, result: EvaluationResult) -> str:
        if "accuracy" in result.metrics:
            return f"Randomized evaluation finished with accuracy {result.metrics.get('accuracy', 0):.4f}."
        return f"Regression evaluation finished with R2 {result.metrics.get('r2_score', 0):.4f}."

    def _primary_metric(self, metrics: dict[str, Any], task_type: str) -> float | None:
        if task_type == "classification":
            return float(metrics.get("f1_score", metrics.get("accuracy", 0.0)))
        return float(metrics.get("r2_score", 0.0))

    def _recommendations(self, task_type: str, metrics: dict[str, Any], top_features: list[dict[str, Any]]) -> str:
        if task_type == "classification":
            metric = metrics.get("f1_score", metrics.get("accuracy", 0.0))
            if metric < 0.8:
                return "Consider collecting more data, revisiting class balance, and expanding feature engineering for classification performance."
        else:
            if metrics.get("r2_score", 0.0) < 0.75:
                return "The regression fit can likely improve with richer features, stronger tuning, or additional domain-specific inputs."
        if not top_features:
            return "Explainability signals were limited; inspect the data pipeline and feature availability before deployment."
        return "The selected model is suitable for a first production candidate. Monitor drift, retrain periodically, and track the top features in downstream governance."
