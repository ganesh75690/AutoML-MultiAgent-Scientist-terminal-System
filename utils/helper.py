from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_NAME_HINTS = (
    "target",
    "label",
    "class",
    "outcome",
    "response",
    "y",
    "churn",
    "attrition",
    "default",
    "purchase",
)


@dataclass(slots=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(slots=True)
class ProblemProfile:
    target_column: str
    task_type: str
    feature_columns: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]


@dataclass(slots=True)
class DatasetIntelligence:
    problem_type: str
    business_domain: str
    industry: str
    quality_score: int
    pii_columns: list[str]
    sensitive_columns: list[str]
    duplicate_patterns: list[str]
    missing_information: list[str]
    notes: list[str]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load a CSV dataset and normalize the column names minimally."""

    df = pd.read_csv(dataset_path)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def infer_target_column(df: pd.DataFrame) -> str:
    """Infer the target column using name hints, then fall back to the last column."""

    lowered = {column.lower().replace(" ", "_"): column for column in df.columns}
    for hint in TARGET_NAME_HINTS:
        if hint in lowered:
            return lowered[hint]
    for column in df.columns:
        normalized = column.lower().replace(" ", "_")
        if any(normalized == hint or normalized.endswith(f"_{hint}") for hint in TARGET_NAME_HINTS):
            return column
    return df.columns[-1]


def detect_task_type(target: pd.Series) -> str:
    """Infer whether the dataset is a classification or regression problem."""

    non_null = target.dropna()
    if non_null.empty:
        return "classification"
    if pd.api.types.is_bool_dtype(non_null) or pd.api.types.is_object_dtype(non_null):
        return "classification"
    unique_count = non_null.nunique(dropna=True)
    if unique_count <= 20 and pd.api.types.is_integer_dtype(non_null):
        return "classification"
    if unique_count <= max(10, int(len(non_null) * 0.1)):
        return "classification"
    if pd.api.types.is_float_dtype(non_null):
        return "regression"
    if unique_count <= 2:
        return "classification"
    return "regression"


def summarize_dataset(df: pd.DataFrame, target_column: str, task_type: str) -> dict[str, Any]:
    """Create a JSON-friendly dataset summary for reporting."""

    feature_columns = [column for column in df.columns if column != target_column]
    numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    missing_counts = df.isna().sum().to_dict()
    target_series = df[target_column]
    target_summary = {
        "unique_values": int(target_series.nunique(dropna=True)),
        "missing_values": int(target_series.isna().sum()),
    }
    if task_type == "classification":
        distribution = target_series.value_counts(dropna=False).head(10).to_dict()
        target_summary["distribution"] = distribution
    else:
        target_summary["mean"] = float(target_series.mean()) if pd.api.types.is_numeric_dtype(target_series) else None
        target_summary["std"] = float(target_series.std()) if pd.api.types.is_numeric_dtype(target_series) else None

    intelligence = analyze_dataset_intelligence(df, target_column)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": int(len(feature_columns)),
        "numeric_feature_count": int(len(numeric_columns)),
        "categorical_feature_count": int(len(categorical_columns)),
        "missing_values": {key: int(value) for key, value in missing_counts.items()},
        "target": target_column,
        "task_type": task_type,
        "target_summary": target_summary,
        "quality_score": intelligence.quality_score,
        "business_domain": intelligence.business_domain,
        "industry": intelligence.industry,
        "problem_type": intelligence.problem_type,
        "pii_columns": intelligence.pii_columns,
        "sensitive_columns": intelligence.sensitive_columns,
        "duplicate_patterns": intelligence.duplicate_patterns,
        "missing_information": intelligence.missing_information,
    }


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    test_size: float,
    random_seed: int,
) -> DataSplit:
    """Split data while preserving class balance when possible."""

    stratify = None
    if task_type == "classification":
        class_counts = y.value_counts(dropna=True)
        if not class_counts.empty and class_counts.min() >= 2:
            stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )
    return DataSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def classification_scoring(y: pd.Series) -> str:
    """Pick a sensible model comparison metric for classification."""

    class_counts = y.value_counts(dropna=True)
    if class_counts.empty:
        return "accuracy"
    minority_ratio = class_counts.min() / class_counts.sum()
    return "f1_weighted" if minority_ratio < 0.35 else "accuracy"


def dataset_signature(dataset_path: Path) -> str:
    """Build a stable dataset signature from path metadata."""

    stat = dataset_path.stat()
    digest_source = f"{dataset_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]


def now_utc() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


def make_json_safe(value: Any) -> Any:
    """Recursively convert common scientific Python objects to JSON-safe types."""

    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return {str(key): make_json_safe(item) for key, item in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Persist a dictionary as pretty-printed JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(data), handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON artifact if it exists."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def feature_profile(df: pd.DataFrame, target_column: str) -> ProblemProfile:
    """Build a lightweight profile of the supervised learning problem."""

    feature_columns = [column for column in df.columns if column != target_column]
    numerical_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])]
    categorical_columns = [column for column in feature_columns if column not in numerical_columns]
    task_type = detect_task_type(df[target_column])
    return ProblemProfile(
        target_column=target_column,
        task_type=task_type,
        feature_columns=feature_columns,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )


def analyze_dataset_intelligence(df: pd.DataFrame, target_column: str) -> DatasetIntelligence:
    feature_columns = [column for column in df.columns if column != target_column]
    normalized_columns = [str(column).lower().replace(" ", "_") for column in df.columns]
    pii_columns = [
        column
        for column in df.columns
        if any(hint in str(column).lower().replace(" ", "_") for hint in ("email", "phone", "ssn", "passport", "address", "dob", "name"))
    ]
    sensitive_columns = [
        column
        for column in df.columns
        if any(hint in str(column).lower().replace(" ", "_") for hint in ("income", "salary", "health", "medical", "credit", "religion", "race", "gender"))
    ]
    missing_information = [column for column in df.columns if df[column].isna().mean() > 0.2]
    duplicate_patterns: list[str] = []
    if df.duplicated().any():
        duplicate_patterns.append("Exact duplicate rows detected")
    if any("id" == column or column.endswith("_id") for column in normalized_columns):
        duplicate_patterns.append("Identifier-like columns present")

    domain = _detect_business_domain(df, feature_columns)
    industry = _detect_industry(df, feature_columns, domain)
    problem_type = "classification" if df[target_column].nunique(dropna=True) <= 20 else "regression"

    missing_ratio = float(df.isna().sum().sum()) / max(1, df.shape[0] * df.shape[1])
    duplicate_ratio = float(df.duplicated().sum()) / max(1, df.shape[0])
    constant_columns = sum(int(df[column].nunique(dropna=True) <= 1) for column in df.columns)
    quality_score = int(
        max(
            0,
            min(
                100,
                100
                - missing_ratio * 55
                - duplicate_ratio * 35
                - constant_columns * 3
                - len(pii_columns) * 4,
            ),
        )
    )

    notes = [
        f"Target column '{target_column}' suggests a {problem_type} problem.",
        f"Dataset quality score estimated at {quality_score}/100.",
    ]
    if pii_columns:
        notes.append("Potential PII should be handled with governance controls.")
    if sensitive_columns:
        notes.append("Sensitive business fields are present.")
    return DatasetIntelligence(
        problem_type=problem_type,
        business_domain=domain,
        industry=industry,
        quality_score=quality_score,
        pii_columns=pii_columns,
        sensitive_columns=sensitive_columns,
        duplicate_patterns=duplicate_patterns,
        missing_information=missing_information,
        notes=notes,
    )


def _detect_business_domain(df: pd.DataFrame, feature_columns: list[str]) -> str:
    columns = " ".join(feature_columns).lower()
    if any(token in columns for token in ("churn", "retention", "customer", "sales", "purchase")):
        return "customer_analytics"
    if any(token in columns for token in ("fraud", "transaction", "payment", "chargeback", "risk")):
        return "financial_risk"
    if any(token in columns for token in ("health", "medical", "patient", "diagnosis")):
        return "healthcare"
    if any(token in columns for token in ("review", "text", "comment", "message", "description")):
        return "nlp"
    if any(token in columns for token in ("image", "pixel", "vision", "camera")):
        return "computer_vision"
    if any(token in columns for token in ("time", "date", "timestamp", "month", "year")):
        return "time_series"
    return "general_tabular"


def _detect_industry(df: pd.DataFrame, feature_columns: list[str], domain: str) -> str:
    columns = " ".join(feature_columns).lower()
    if domain == "financial_risk":
        return "fintech"
    if domain == "healthcare":
        return "healthcare"
    if domain == "time_series":
        return "operations"
    if domain == "nlp":
        return "media"
    if any(token in columns for token in ("retail", "product", "inventory")):
        return "retail"
    if any(token in columns for token in ("hr", "employee", "attrition", "salary")):
        return "human_resources"
    return "analytics"
