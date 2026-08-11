# src/analyst_agent/ingestion/profiler.py

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: list


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: list[ColumnProfile] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """
        Renders this profile as a compact text summary suitable for
        including in an LLM prompt. Kept separate from __str__ or repr
        so we control exactly what the LLM sees, independent of how
        we might display this in the UI later.
        """
        lines = [
            f"Dataset shape: {self.row_count} rows x {self.column_count} columns",
            f"Duplicate rows: {self.duplicate_row_count}",
            "Columns:",
        ]
        for col in self.columns:
            lines.append(
                f"  - {col.name} ({col.dtype}): "
                f"{col.null_percentage:.1f}% null, "
                f"{col.unique_count} unique values, "
                f"sample: {col.sample_values}"
            )
        return "\n".join(lines)


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """
    Builds a structured profile of a DataFrame: shape, duplicates,
    and per-column statistics (type, nulls, uniqueness, samples).
    """
    columns = []
    for col_name in df.columns:
        series = df[col_name]
        null_count = int(series.isna().sum())
        columns.append(
            ColumnProfile(
                name=col_name,
                dtype=str(series.dtype),
                null_count=null_count,
                null_percentage=round((null_count / len(df)) * 100, 2) if len(df) else 0.0,
                unique_count=int(series.nunique()),
                sample_values=series.dropna().unique()[:3].tolist(),
            )
        )

    profile = DatasetProfile(
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_row_count=int(df.duplicated().sum()),
        columns=columns,
    )

    logger.info(
        "Profiled dataset: rows=%d, columns=%d, duplicates=%d",
        profile.row_count, profile.column_count, profile.duplicate_row_count,
    )
    return profile