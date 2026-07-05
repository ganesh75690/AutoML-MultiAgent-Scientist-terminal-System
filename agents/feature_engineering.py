from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from utils.config import AppConfig


@dataclass(slots=True)
class FeatureEngineeringResult:
    selected_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    preprocessor: ColumnTransformer
    summary: dict[str, Any]


class FeatureEngineeringAgent:
    """Prepare feature schemas and reusable preprocessing pipelines."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def transform(self, X_train: pd.DataFrame, target_column: str, task_type: str) -> FeatureEngineeringResult:
        feature_frame = X_train.copy()
        selected_columns = list(feature_frame.columns)

        low_variance_columns = [
            column for column in selected_columns if feature_frame[column].nunique(dropna=True) <= 1
        ]
        if low_variance_columns:
            selected_columns = [column for column in selected_columns if column not in low_variance_columns]

        numeric_columns = [column for column in selected_columns if pd.api.types.is_numeric_dtype(feature_frame[column])]
        categorical_columns = [column for column in selected_columns if column not in numeric_columns]

        correlated_drop: list[str] = []
        if len(numeric_columns) >= 2:
            correlation_matrix = feature_frame[numeric_columns].corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            for column in upper_triangle.columns:
                if any(upper_triangle[column] > 0.98):
                    correlated_drop.append(column)
            if correlated_drop:
                selected_columns = [column for column in selected_columns if column not in correlated_drop]
                numeric_columns = [column for column in numeric_columns if column not in correlated_drop]

        numeric_pipeline_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if self.config.allow_polynomial_features and 2 <= len(numeric_columns) <= self.config.max_polynomial_numeric_features:
            numeric_pipeline_steps.append(("polynomial", PolynomialFeatures(degree=2, include_bias=False)))
        numeric_pipeline_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(numeric_pipeline_steps)

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )

        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric_columns:
            transformers.append(("numeric", numeric_pipeline, numeric_columns))
        if categorical_columns:
            transformers.append(("categorical", categorical_pipeline, categorical_columns))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

        summary: dict[str, Any] = {
            "selected_columns": selected_columns,
            "removed_low_variance_columns": low_variance_columns,
            "removed_correlated_columns": correlated_drop,
            "numeric_features": numeric_columns,
            "categorical_features": categorical_columns,
            "polynomial_features_enabled": bool(
                self.config.allow_polynomial_features and 2 <= len(numeric_columns) <= self.config.max_polynomial_numeric_features
            ),
            "target_column": target_column,
            "task_type": task_type,
        }

        return FeatureEngineeringResult(
            selected_columns=selected_columns,
            numeric_features=numeric_columns,
            categorical_features=categorical_columns,
            preprocessor=preprocessor,
            summary=summary,
        )
