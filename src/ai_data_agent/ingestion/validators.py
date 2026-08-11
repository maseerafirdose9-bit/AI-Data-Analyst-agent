# src/analyst_agent/ingestion/validators.py

import logging

from ai_data_agent.exceptions import DataValidationError
from ai_data_agent.ingestion.profiler import DatasetProfile

logger = logging.getLogger(__name__)

MIN_ROWS = 2
MAX_NULL_PERCENTAGE_PER_COLUMN = 95.0


def validate_dataset_quality(profile: DatasetProfile) -> list[str]:
    """
    Checks a dataset profile against basic quality rules.

    Returns a list of warning strings (non-fatal issues the user should
    know about, e.g., "column X is 60% null") and raises DataValidationError
    for fatal issues that make the dataset unusable.
    """
    if profile.row_count < MIN_ROWS:
        raise DataValidationError(
            f"Dataset has only {profile.row_count} row(s); "
            f"at least {MIN_ROWS} are needed for meaningful analysis."
        )

    warnings = []

    for col in profile.columns:
        if col.null_percentage >= MAX_NULL_PERCENTAGE_PER_COLUMN:
            warnings.append(
                f"Column '{col.name}' is {col.null_percentage:.1f}% empty "
                f"and may not be useful for analysis."
            )

    if profile.duplicate_row_count > 0:
        pct = round((profile.duplicate_row_count / profile.row_count) * 100, 1)
        warnings.append(
            f"Found {profile.duplicate_row_count} duplicate rows ({pct}% of the dataset)."
        )

    if warnings:
        logger.warning("Dataset quality warnings: %s", warnings)

    return warnings