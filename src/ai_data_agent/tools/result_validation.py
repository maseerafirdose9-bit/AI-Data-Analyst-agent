# src/ai_data_agent/tools/result_validation.py

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


def validate_columns_referenced(code: str, valid_columns: list[str]) -> tuple[bool, str | None]:
    """
    Checks that every column name referenced in df['...'] or df.column_name
    syntax actually exists in the dataset. Catches hallucinated column names
    before execution would even fail on them naturally.
    """
    bracket_refs = re.findall(r"df\[['\"]([\w\s]+)['\"]\]", code)
    for col in bracket_refs:
        if col not in valid_columns:
            return False, f"Referenced column '{col}' does not exist in the dataset."
    return True, None


def validate_result_shape(result: object) -> tuple[bool, str | None]:
    """
    Sanity-checks the shape of the execution result itself — catches
    cases like an empty Series/DataFrame, which is technically a
    successful execution but a useless answer.
    """
    if result is None:
        return False, "Result is empty (None)."

    if isinstance(result, pd.Series) and result.empty:
        return False, "Result is an empty Series — likely an overly restrictive filter."

    if isinstance(result, pd.DataFrame) and result.empty:
        return False, "Result is an empty DataFrame — likely an overly restrictive filter."

    if isinstance(result, float) and (pd.isna(result)):
        return False, "Result is NaN — calculation likely failed silently."

    return True, None