from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    confusion: list[list[int]] | None


class EvaluationAgent:
    """Score the tuned model on the holdout set."""

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> EvaluationResult:
        predictions = model.predict(X_test)
        result_frame = pd.DataFrame({"actual": y_test.reset_index(drop=True), "prediction": pd.Series(predictions)})

        if task_type == "classification":
            metrics = self._classification_metrics(model, X_test, y_test, predictions)
            confusion = confusion_matrix(y_test, predictions).tolist()
            return EvaluationResult(metrics=metrics, predictions=result_frame, confusion=confusion)

        metrics = self._regression_metrics(y_test, predictions)
        return EvaluationResult(metrics=metrics, predictions=result_frame, confusion=None)

    def _classification_metrics(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        }
        roc_auc = self._roc_auc(model, X_test, y_test)
        if roc_auc is not None:
            metrics["roc_auc"] = float(roc_auc)
        return metrics

    def _roc_auc(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
        if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
            return None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    return roc_auc_score(y_test, proba[:, 1])
                return roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
            decision_values = model.decision_function(X_test)
            return roc_auc_score(y_test, decision_values)
        except Exception:
            return None

    def _regression_metrics(self, y_test: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        return {
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2_score": float(r2_score(y_test, predictions)),
        }
