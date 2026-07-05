from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils.config import AppConfig

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency path
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - optional dependency path
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency path
    CatBoostClassifier = None
    CatBoostRegressor = None


@dataclass(slots=True)
class TuningResult:
    tuned_pipeline: Pipeline
    best_params: dict[str, Any]
    best_score: float
    search_strategy: str


class HyperparameterTuningAgent:
    """Optimize the strongest candidate with a small but meaningful search."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_columns: list[str],
        preprocessor: Any,
        estimator: Any,
        task_type: str,
        scoring: str,
    ) -> TuningResult:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        param_grid = self._parameter_grid(estimator, task_type)
        if not param_grid:
            pipeline.fit(X_train[feature_columns], y_train)
            return TuningResult(tuned_pipeline=pipeline, best_params={}, best_score=float("nan"), search_strategy="none")

        splitter = self._splitter(y_train, task_type)
        total_combinations = self._grid_size(param_grid)
        use_random_search = total_combinations > 20 or hasattr(estimator, "get_params") and estimator.__class__.__name__ in {
            "XGBClassifier",
            "XGBRegressor",
            "LGBMClassifier",
            "LGBMRegressor",
            "CatBoostClassifier",
            "CatBoostRegressor",
        }

        if use_random_search:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=min(self.config.max_tuning_iterations, max(1, total_combinations)),
                scoring=scoring,
                cv=splitter,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_seed,
                refit=True,
                error_score="raise",
            )
            strategy = "random_search"
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=splitter,
                n_jobs=self.config.n_jobs,
                refit=True,
                error_score="raise",
            )
            strategy = "grid_search"

        search.fit(X_train[feature_columns], y_train)
        return TuningResult(
            tuned_pipeline=search.best_estimator_,
            best_params=dict(search.best_params_),
            best_score=float(search.best_score_),
            search_strategy=strategy,
        )

    def _splitter(self, y_train: pd.Series, task_type: str):
        if task_type == "classification":
            class_counts = y_train.value_counts()
            if class_counts.empty or class_counts.min() < 2:
                return KFold(n_splits=min(self.config.cv_folds, max(2, len(y_train) // 4)), shuffle=True, random_state=self.config.random_seed)
            n_splits = min(self.config.cv_folds, int(class_counts.min()))
            return StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=self.config.random_seed)
        return KFold(n_splits=min(self.config.cv_folds, max(2, len(y_train) // 4)), shuffle=True, random_state=self.config.random_seed)

    def _grid_size(self, param_grid: dict[str, list[Any]]) -> int:
        sizes = [len(values) for values in param_grid.values()]
        total = 1
        for size in sizes:
            total *= max(1, size)
        return total

    def _parameter_grid(self, estimator: Any, task_type: str) -> dict[str, list[Any]]:
        name = estimator.__class__.__name__
        if task_type == "classification":
            if name == "LogisticRegression":
                return {"model__C": [0.1, 1.0, 10.0], "model__solver": ["lbfgs", "saga"]}
            if name == "RandomForestClassifier":
                return {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 10, 20],
                    "model__min_samples_split": [2, 5],
                    "model__min_samples_leaf": [1, 2],
                }
            if name == "SVC":
                return {"model__C": [0.5, 1.0, 2.0], "model__gamma": ["scale", "auto"], "model__kernel": ["rbf", "poly"]}
            if name == "DecisionTreeClassifier":
                return {"model__max_depth": [None, 5, 10, 20], "model__min_samples_split": [2, 5, 10]}
            if name == "XGBClassifier":
                return {
                    "model__n_estimators": [150, 250, 350],
                    "model__max_depth": [4, 6, 8],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__subsample": [0.8, 0.9, 1.0],
                    "model__colsample_bytree": [0.8, 0.9, 1.0],
                }
            if name == "LGBMClassifier":
                return {
                    "model__n_estimators": [150, 250, 350],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__num_leaves": [15, 31, 63],
                    "model__max_depth": [-1, 10, 20],
                }
            if name == "CatBoostClassifier":
                return {
                    "model__iterations": [150, 250, 350],
                    "model__depth": [4, 6, 8],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                }
        else:
            if name == "LinearRegression":
                return {}
            if name == "RandomForestRegressor":
                return {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 10, 20],
                    "model__min_samples_split": [2, 5],
                    "model__min_samples_leaf": [1, 2],
                }
            if name == "SVR":
                return {"model__C": [0.5, 1.0, 2.0], "model__gamma": ["scale", "auto"], "model__kernel": ["rbf", "poly"]}
            if name == "DecisionTreeRegressor":
                return {"model__max_depth": [None, 5, 10, 20], "model__min_samples_split": [2, 5, 10]}
            if name == "XGBRegressor":
                return {
                    "model__n_estimators": [150, 250, 350],
                    "model__max_depth": [4, 6, 8],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__subsample": [0.8, 0.9, 1.0],
                    "model__colsample_bytree": [0.8, 0.9, 1.0],
                }
            if name == "LGBMRegressor":
                return {
                    "model__n_estimators": [150, 250, 350],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__num_leaves": [15, 31, 63],
                    "model__max_depth": [-1, 10, 20],
                }
            if name == "CatBoostRegressor":
                return {
                    "model__iterations": [150, 250, 350],
                    "model__depth": [4, 6, 8],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                }
        return {}
