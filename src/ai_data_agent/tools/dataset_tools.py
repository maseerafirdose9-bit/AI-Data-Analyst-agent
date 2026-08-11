# src/ai_data_agent/tools/dataset_tools.py

import logging

import pandas as pd
from pydantic import BaseModel, Field

from ai_data_agent.ingestion.profiler import DatasetProfile, profile_dataset

logger = logging.getLogger(__name__)


class InspectDatasetInput(BaseModel):
    """Input schema for the inspect_dataset tool. Takes no real arguments —
    the dataset is already in agent state — but we define this explicitly
    so the tool has a consistent, self-documenting schema."""
    pass


class InspectDatasetOutput(BaseModel):
    row_count: int
    column_count: int
    column_names: list[str]
    column_types: dict[str, str]
    summary_text: str = Field(
        description="Human/LLM-readable summary of the dataset schema."
    )


def inspect_dataset(profile: DatasetProfile) -> InspectDatasetOutput:
    """
    Returns structured schema information about the dataset, formatted
    for the LLM to reason about which columns are relevant to a question.
    """
    return InspectDatasetOutput(
        row_count=profile.row_count,
        column_count=profile.column_count,
        column_names=[c.name for c in profile.columns],
        column_types={c.name: c.dtype for c in profile.columns},
        summary_text=profile.to_prompt_context(),
    )


class DescribeDatasetInput(BaseModel):
    columns: list[str] | None = Field(
        default=None,
        description="Specific columns to describe. If omitted, describes all numeric columns."
    )


class DescribeDatasetOutput(BaseModel):
    stats: dict[str, dict[str, float]] = Field(
        description="Per-column statistics: mean, std, min, max, etc."
    )
    warnings: list[str] = Field(default_factory=list)


def describe_dataset(df: pd.DataFrame, input_data: DescribeDatasetInput) -> DescribeDatasetOutput:
    """
    Returns numeric statistical summaries for the requested columns
    (or all numeric columns if none specified).
    """
    warnings = []

    if input_data.columns:
        missing = [c for c in input_data.columns if c not in df.columns]
        if missing:
            warnings.append(f"Columns not found and skipped: {missing}")
        target_cols = [c for c in input_data.columns if c in df.columns]
    else:
        target_cols = df.select_dtypes(include="number").columns.tolist()

    if not target_cols:
        return DescribeDatasetOutput(stats={}, warnings=warnings + ["No numeric columns to describe."])

    described = df[target_cols].describe().to_dict()
    # Round for readability/token efficiency — the LLM doesn't need
    # 15 decimal places of floating point noise.
    stats = {
        col: {stat: round(val, 4) for stat, val in col_stats.items()}
        for col, col_stats in described.items()
    }

    return DescribeDatasetOutput(stats=stats, warnings=warnings)