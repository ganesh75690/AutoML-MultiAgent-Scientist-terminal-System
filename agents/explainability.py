from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import Ridge

from utils.config import AppConfig

try:
    import shap
except Exception:  # pragma: no cover - optional dependency path
    shap = None


@dataclass(slots=True)
class ExplainabilityResult:
    top_features: list[dict[str, Any]]
    importance_map: dict[str, float]
    plot_path: Path
    method: str
    artifacts: dict[str, str]
    summary: dict[str, Any]
    artifacts: dict[str, str]
    summary: dict[str, Any]


class ExplainabilityAgent:
    """Generate SHAP-based and fallback feature attribution artifacts."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def explain(
        self,
        model_pipeline: Any,
        X_sample: pd.DataFrame,
        feature_columns: list[str],
        output_dir: Path,
        y_reference: pd.Series | None = None,
    ) -> ExplainabilityResult:
        preprocessor = model_pipeline.named_steps["preprocessor"]
        estimator = model_pipeline.named_steps["model"]
        sample = X_sample[feature_columns].copy().head(min(200, len(X_sample)))
        transformed = preprocessor.transform(sample)
        feature_names = list(preprocessor.get_feature_names_out())
        dense_transformed = transformed.toarray() if sparse.issparse(transformed) else np.asarray(transformed)

        artifacts: dict[str, str] = {}
        explanation_summary: dict[str, Any] = {}
        importance_map: dict[str, float] = {}
        method = "combined"

        try:
            importance_map = self._shap_importance(estimator, dense_transformed, feature_names)
            artifacts["shap"] = str(output_dir / "shap_importance.png")
        except Exception:
            importance_map = self._fallback_importance(estimator, feature_names)
            method = "fallback_importance"

        raw_importance: dict[str, float] = {}
        if y_reference is not None:
            try:
                raw_permutation = permutation_importance(
                    model_pipeline,
                    sample,
                    y_reference.loc[sample.index],
                    n_repeats=5,
                    random_state=self.config.random_seed,
                    n_jobs=self.config.n_jobs,
                )
                raw_importance = {column: float(score) for column, score in zip(sample.columns, raw_permutation.importances_mean)}
                artifacts["permutation"] = str(output_dir / "permutation_importance.png")
            except Exception:
                raw_importance = {}

        if raw_importance:
            importance_map = {**importance_map, **raw_importance}

        top_features = [
            {"feature": feature, "importance": float(score)}
            for feature, score in sorted(importance_map.items(), key=lambda item: item[1], reverse=True)[:10]
        ]
        plot_path = output_dir / "feature_importance.png"
        self._save_plot(top_features, plot_path)
        if raw_importance:
            self._save_plot(
                [
                    {"feature": feature, "importance": score}
                    for feature, score in sorted(raw_importance.items(), key=lambda item: item[1], reverse=True)[:10]
                ],
                output_dir / "permutation_importance.png",
                title="Permutation Importance",
                color="#E76F51",
            )

        correlation_path = self._save_correlation_plot(sample, output_dir / "feature_correlation.png")
        artifacts["correlation"] = str(correlation_path)

        partial_dependence_path = self._save_partial_dependence(model_pipeline, sample, output_dir)
        if partial_dependence_path is not None:
            artifacts["partial_dependence"] = str(partial_dependence_path)

        local_explanation = self._save_local_lime_style_explanation(model_pipeline, sample, output_dir)
        artifacts["lime_style"] = str(local_explanation)

        error_analysis = self._save_error_analysis(model_pipeline, sample, y_reference, output_dir)
        if error_analysis is not None:
            artifacts["error_analysis"] = str(error_analysis)

        explanation_summary["top_features"] = top_features
        explanation_summary["raw_permutation_available"] = bool(raw_importance)
        explanation_summary["has_shap"] = method in {"combined", "shap"}
        explanation_summary["has_partial_dependence"] = partial_dependence_path is not None
        explanation_summary["has_error_analysis"] = error_analysis is not None
        explanation_summary["has_correlation_analysis"] = True
        return ExplainabilityResult(
            top_features=top_features,
            importance_map=importance_map,
            plot_path=plot_path,
            method=method,
            artifacts=artifacts,
            summary=explanation_summary,
        )

    def _shap_importance(self, estimator: Any, dense_transformed: np.ndarray, feature_names: list[str]) -> dict[str, float]:
        if shap is None:
            raise RuntimeError("SHAP is not available")
        background = dense_transformed[: min(50, len(dense_transformed))]
        explainer: Any
        try:
            explainer = shap.Explainer(estimator, background)
        except Exception:
            if hasattr(estimator, "coef_"):
                explainer = shap.LinearExplainer(estimator, background)
            else:
                raise
        shap_values = explainer(dense_transformed)
        values = shap_values.values if hasattr(shap_values, "values") else shap_values
        if isinstance(values, list):
            values = np.mean([np.abs(np.asarray(item)) for item in values], axis=0)
        else:
            values = np.abs(np.asarray(values))
        if values.ndim == 3:
            values = values.mean(axis=1)
        scores = np.mean(values, axis=0)
        return {feature: float(score) for feature, score in zip(feature_names, scores)}

    def _fallback_importance(self, estimator: Any, feature_names: list[str]) -> dict[str, float]:
        if hasattr(estimator, "feature_importances_"):
            scores = np.asarray(estimator.feature_importances_)
            return {feature: float(score) for feature, score in zip(feature_names, scores)}
        if hasattr(estimator, "coef_"):
            coefficients = np.asarray(estimator.coef_)
            if coefficients.ndim > 1:
                coefficients = np.mean(np.abs(coefficients), axis=0)
            else:
                coefficients = np.abs(coefficients)
            return {feature: float(score) for feature, score in zip(feature_names, coefficients)}
        uniform_score = 1.0 / max(1, len(feature_names))
        return {feature: uniform_score for feature in feature_names}

    def _save_plot(self, top_features: list[dict[str, Any]], plot_path: Path, title: str = "Top Feature Importance", color: str = "#2A9D8F") -> None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        if not top_features:
            return
        features = [item["feature"] for item in reversed(top_features)]
        scores = [item["importance"] for item in reversed(top_features)]
        plt.figure(figsize=(10, 6))
        plt.barh(features, scores, color=color)
        plt.title(title)
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

    def _save_correlation_plot(self, sample: pd.DataFrame, path: Path) -> Path:
        numeric = sample.select_dtypes(include=[np.number])
        path.parent.mkdir(parents=True, exist_ok=True)
        if numeric.empty:
            return path
        correlation = numeric.corr()
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation, cmap="viridis", aspect="auto")
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
        plt.yticks(range(len(correlation.index)), correlation.index)
        plt.title("Feature Correlation")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        return path

    def _save_partial_dependence(self, model_pipeline: Any, sample: pd.DataFrame, output_dir: Path) -> Path | None:
        numeric_columns = list(sample.select_dtypes(include=[np.number]).columns[:2])
        if not numeric_columns:
            return None
        path = output_dir / "partial_dependence.png"
        try:
            numeric_sample = sample.copy()
            for column in numeric_columns:
                numeric_sample[column] = numeric_sample[column].astype(float)
            display = PartialDependenceDisplay.from_estimator(model_pipeline, numeric_sample, numeric_columns)
            display.figure_.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(display.figure_)
            return path
        except Exception:
            return None

    def _save_local_lime_style_explanation(self, model_pipeline: Any, sample: pd.DataFrame, output_dir: Path) -> Path:
        path = output_dir / "lime_style_explanation.json"
        row = sample.head(1)
        if row.empty:
            path.write_text(json.dumps({"message": "No sample available"}, indent=2), encoding="utf-8")
            return path
        baseline = sample.median(numeric_only=True).fillna(0)
        perturbed = sample.copy().head(min(30, len(sample)))
        for column in baseline.index:
            if column in perturbed.columns:
                perturbed[column] = perturbed[column].fillna(baseline[column])
        try:
            predictions = model_pipeline.predict(perturbed)
            local_model = Ridge(alpha=1.0)
            local_model.fit(pd.get_dummies(perturbed, drop_first=False), predictions)
            coefficients = local_model.coef_.ravel()
            explanation = {
                "type": "lime_style_local_surrogate",
                "features": pd.get_dummies(perturbed, drop_first=False).columns.tolist(),
                "coefficients": [float(value) for value in coefficients],
            }
        except Exception as exc:
            explanation = {"type": "lime_style_local_surrogate", "message": str(exc)}
        path.write_text(json.dumps(explanation, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _save_error_analysis(self, model_pipeline: Any, sample: pd.DataFrame, y_reference: pd.Series | None, output_dir: Path) -> Path | None:
        if y_reference is None:
            return None
        predictions = model_pipeline.predict(sample)
        comparison = pd.DataFrame({"actual": y_reference.loc[sample.index].values, "prediction": predictions})
        path = output_dir / "error_analysis.csv"
        comparison.to_csv(path, index=False)
        return path
