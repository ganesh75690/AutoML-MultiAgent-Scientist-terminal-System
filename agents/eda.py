from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EDAResult:
    summary: dict[str, Any]
    insights: list[str]


class ExploratoryDataAnalysisAgent:
    """Produce compact but useful dataset diagnostics for reporting."""

    def analyze(self, df: pd.DataFrame, target_column: str, task_type: str) -> EDAResult:
        feature_columns = [column for column in df.columns if column != target_column]
        numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])]
        categorical_columns = [column for column in feature_columns if column not in numeric_columns]

        summary: dict[str, Any] = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "target_column": target_column,
            "task_type": task_type,
            "numeric_features": numeric_columns,
            "categorical_features": categorical_columns,
            "numeric_statistics": {},
            "categorical_cardinality": {},
            "missing_values": {column: int(count) for column, count in df.isna().sum().items()},
        }

        if numeric_columns:
            numeric_describe = df[numeric_columns].describe().T
            summary["numeric_statistics"] = numeric_describe[["mean", "std", "min", "max"]].round(4).to_dict(orient="index")

        summary["categorical_cardinality"] = {
            column: int(df[column].nunique(dropna=True)) for column in categorical_columns
        }

        target_series = df[target_column]
        insights: list[str] = [
            f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.",
            f"Detected {len(numeric_columns)} numerical and {len(categorical_columns)} categorical features.",
            f"Target column '{target_column}' has {int(target_series.nunique(dropna=True))} unique values.",
        ]

        if task_type == "classification":
            distribution = target_series.value_counts(dropna=False)
            majority_ratio = float(distribution.iloc[0] / distribution.sum()) if not distribution.empty else 0.0
            summary["target_distribution"] = distribution.to_dict()
            summary["imbalance_ratio"] = round(majority_ratio, 4)
            if majority_ratio > 0.75:
                insights.append("The target distribution is imbalanced and should be monitored during evaluation.")
        else:
            summary["target_statistics"] = target_series.describe().round(4).to_dict()
            insights.append("Regression target detected; metrics will focus on error magnitude and explained variance.")

        if df[numeric_columns].shape[1] >= 2:
            correlation = df[numeric_columns].corr().abs()
            lower_mask = np.tril(np.ones(correlation.shape, dtype=bool))
            upper = correlation.where(~lower_mask)
            highest = upper.max().max()
            if pd.notna(highest) and highest > 0.9:
                insights.append("Some numerical features are strongly correlated, so feature selection may help.")

        return EDAResult(summary=summary, insights=insights)
