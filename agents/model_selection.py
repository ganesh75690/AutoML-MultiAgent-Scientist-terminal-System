from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.config import AppConfig

try:  # Optional dependencies are allowed but not required for the workflow to start.
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
class ModelCandidateResult:
    name: str
    estimator: Any
    mean_score: float
    std_score: float
    cv_scores: list[float]


@dataclass(slots=True)
class ModelSelectionResult:
    results: list[ModelCandidateResult]
    best_candidate: ModelCandidateResult
    scoring: str


class ModelSelectionAgent:
    """Compare multiple candidate algorithms using cross validation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_columns: list[str],
        preprocessor: Any,
        task_type: str,
        scoring: str,
    ) -> ModelSelectionResult:
        splitter = self._splitter(y_train, task_type)
        candidates = self._build_candidates(task_type)
        results: list[ModelCandidateResult] = []

        for name, estimator in candidates:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )
            try:
                cv_scores = cross_val_score(
                    pipeline,
                    X_train[feature_columns],
                    y_train,
                    cv=splitter,
                    scoring=scoring,
                    n_jobs=self.config.n_jobs,
                    error_score="raise",
                )
                result = ModelCandidateResult(
                    name=name,
                    estimator=estimator,
                    mean_score=float(np.mean(cv_scores)),
                    std_score=float(np.std(cv_scores)),
                    cv_scores=[float(score) for score in cv_scores],
                )
                results.append(result)
            except Exception:
                continue

        if not results:
            fallback = self._fallback_candidate(task_type)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", fallback[1]),
                ]
            )
            cv_scores = cross_val_score(
                pipeline,
                X_train[feature_columns],
                y_train,
                cv=splitter,
                scoring=scoring,
                n_jobs=self.config.n_jobs,
            )
            result = ModelCandidateResult(
                name=fallback[0],
                estimator=fallback[1],
                mean_score=float(np.mean(cv_scores)),
                std_score=float(np.std(cv_scores)),
                cv_scores=[float(score) for score in cv_scores],
            )
            results = [result]

        results.sort(key=lambda item: item.mean_score, reverse=True)
        return ModelSelectionResult(results=results, best_candidate=results[0], scoring=scoring)

    def _splitter(self, y_train: pd.Series, task_type: str):
        if task_type == "classification":
            class_counts = y_train.value_counts()
            if class_counts.empty or class_counts.min() < 2:
                return KFold(n_splits=min(self.config.cv_folds, max(2, len(y_train) // 4)), shuffle=True, random_state=self.config.random_seed)
            n_splits = min(self.config.cv_folds, int(class_counts.min()))
            return StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=self.config.random_seed)
        return KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_seed)

    def _build_candidates(self, task_type: str) -> list[tuple[str, Any]]:
        if task_type == "classification":
            return self._build_classification_candidates()
        return self._build_regression_candidates()

    def _build_classification_candidates(self) -> list[tuple[str, Any]]:
        candidates: list[tuple[str, Any]] = [
            (
                "Logistic Regression",
                LogisticRegression(max_iter=3000, solver="lbfgs", random_state=self.config.random_seed),
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=self.config.random_seed,
                    n_jobs=self.config.n_jobs,
                ),
            ),
            ("SVM", SVC(probability=True, random_state=self.config.random_seed)),
            (
                "Decision Tree",
                DecisionTreeClassifier(random_state=self.config.random_seed),
            ),
        ]
        if XGBClassifier is not None:
            candidates.append(
                (
                    "XGBoost",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=self.config.random_seed,
                        n_jobs=self.config.n_jobs,
                        eval_metric="logloss",
                        tree_method="hist",
                    ),
                )
            )
        if LGBMClassifier is not None:
            candidates.append(
                (
                    "LightGBM",
                    LGBMClassifier(
                        n_estimators=250,
                        learning_rate=0.08,
                        random_state=self.config.random_seed,
                        n_jobs=self.config.n_jobs,
                        verbose=-1,
                    ),
                )
            )
        if CatBoostClassifier is not None:
            candidates.append(
                (
                    "CatBoost",
                    CatBoostClassifier(
                        iterations=250,
                        learning_rate=0.08,
                        depth=6,
                        random_seed=self.config.random_seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                )
            )
        return candidates

    def _build_regression_candidates(self) -> list[tuple[str, Any]]:
        candidates: list[tuple[str, Any]] = [
            ("Linear Regression", LinearRegression()),
            (
                "Random Forest Regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=self.config.random_seed,
                    n_jobs=self.config.n_jobs,
                ),
            ),
            ("SVR", SVR()),
            (
                "Decision Tree Regressor",
                DecisionTreeRegressor(random_state=self.config.random_seed),
            ),
        ]
        if XGBRegressor is not None:
            candidates.append(
                (
                    "XGBoost Regressor",
                    XGBRegressor(
                        n_estimators=250,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=self.config.random_seed,
                        n_jobs=self.config.n_jobs,
                        tree_method="hist",
                    ),
                )
            )
        if LGBMRegressor is not None:
            candidates.append(
                (
                    "LightGBM Regressor",
                    LGBMRegressor(
                        n_estimators=250,
                        learning_rate=0.08,
                        random_state=self.config.random_seed,
                        n_jobs=self.config.n_jobs,
                        verbose=-1,
                    ),
                )
            )
        if CatBoostRegressor is not None:
            candidates.append(
                (
                    "CatBoost Regressor",
                    CatBoostRegressor(
                        iterations=250,
                        learning_rate=0.08,
                        depth=6,
                        random_seed=self.config.random_seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                )
            )
        return candidates

    def _fallback_candidate(self, task_type: str) -> tuple[str, Any]:
        if task_type == "classification":
            return (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=self.config.random_seed,
                    n_jobs=self.config.n_jobs,
                ),
            )
        return (
            "Random Forest Regressor",
            RandomForestRegressor(
                n_estimators=300,
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
            ),
        )
