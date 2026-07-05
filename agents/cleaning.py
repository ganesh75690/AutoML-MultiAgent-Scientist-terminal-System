from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils.config import AppConfig


@dataclass(slots=True)
class CleaningResult:
    cleaned_data: pd.DataFrame
    summary: dict[str, Any]


class DataCleaningAgent:
    """Prepare raw datasets for supervised learning."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self, df: pd.DataFrame, target_column: str) -> CleaningResult:
        working = df.copy()
        summary: dict[str, Any] = {
            "initial_rows": int(working.shape[0]),
            "initial_columns": int(working.shape[1]),
            "target_column": target_column,
            "duplicates_removed": 0,
            "columns_dropped": [],
            "missing_values_filled": {},
            "outliers_clipped": {},
        }

        working = working.replace([np.inf, -np.inf], np.nan)
        duplicate_count = int(working.duplicated().sum())
        if duplicate_count > 0:
            working = working.drop_duplicates().reset_index(drop=True)
        summary["duplicates_removed"] = duplicate_count

        if target_column in working.columns:
            before_target_drop = int(working.shape[0])
            working = working.dropna(subset=[target_column]).reset_index(drop=True)
            summary["target_rows_removed"] = before_target_drop - int(working.shape[0])

        missing_ratio = working.isna().mean()
        protected_columns = {target_column}
        columns_to_drop = [
            column
            for column, ratio in missing_ratio.items()
            if ratio > self.config.missing_column_threshold and column not in protected_columns
        ]
        if columns_to_drop:
            working = working.drop(columns=columns_to_drop)
        summary["columns_dropped"] = columns_to_drop

        numeric_columns = [
            column
            for column in working.columns
            if column != target_column and pd.api.types.is_numeric_dtype(working[column])
        ]
        categorical_columns = [
            column
            for column in working.columns
            if column != target_column and column not in numeric_columns
        ]

        for column in numeric_columns:
            series = working[column]
            non_null = series.dropna()
            if non_null.empty:
                fill_value = 0.0
                clipped_count = 0
            else:
                q1 = non_null.quantile(0.25)
                q3 = non_null.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.config.outlier_clip_iqr * iqr
                upper = q3 + self.config.outlier_clip_iqr * iqr
                clipped = series.clip(lower=lower, upper=upper)
                clipped_count = int((series.notna() & (series != clipped)).sum())
                working[column] = clipped
                fill_value = float(non_null.median())
            working[column] = working[column].fillna(fill_value)
            summary["missing_values_filled"][column] = int(series.isna().sum())
            summary["outliers_clipped"][column] = clipped_count

        for column in categorical_columns:
            series = working[column].astype("object")
            mode = series.mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            working[column] = series.fillna(fill_value)
            summary["missing_values_filled"][column] = int(series.isna().sum())

        working = working.reset_index(drop=True)
        summary["final_rows"] = int(working.shape[0])
        summary["final_columns"] = int(working.shape[1])
        summary["missing_values_after_cleaning"] = {
            column: int(count) for column, count in working.isna().sum().items()
        }
        return CleaningResult(cleaned_data=working, summary=summary)
