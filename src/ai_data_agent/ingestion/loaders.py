# src/analyst_agent/ingestion/loaders.py

import io
import logging
from pathlib import Path

import pandas as pd

from ai_data_agent.config import settings
from ai_data_agent.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".parquet"}


def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Loads a dataset from raw file bytes into a Pandas DataFrame.

    Supports CSV, Excel (.xlsx), and Parquet formats, detected from
    the file extension.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename, used only to detect format
                  (never used for disk storage — see security note).

    Returns:
        A Pandas DataFrame containing the loaded data.

    Raises:
        DataIngestionError: If the file is too large, has an unsupported
                             extension, or cannot be parsed.
    """
    _validate_file_size(file_bytes)

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DataIngestionError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    logger.info("Loading dataset: filename=%s, extension=%s, size_kb=%.1f",
                filename, extension, len(file_bytes) / 1024)

    try:
        if extension == ".csv":
            df = _load_csv(file_bytes)
        elif extension == ".xlsx":
            df = _load_excel(file_bytes)
        elif extension == ".parquet":
            df = _load_parquet(file_bytes)
    except DataIngestionError:
        raise
    except Exception as exc:
        # Any unexpected parsing failure gets wrapped in our own
        # exception type, so callers only ever need to catch one thing.
        raise DataIngestionError(f"Failed to parse '{filename}': {exc}") from exc

    if df.empty:
        raise DataIngestionError(f"'{filename}' loaded successfully but contains no rows.")

    logger.info("Dataset loaded: rows=%d, columns=%d", len(df), len(df.columns))
    return df


def _validate_file_size(file_bytes: bytes) -> None:
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise DataIngestionError(
            f"File size {size_mb:.1f}MB exceeds the {settings.max_file_size_mb}MB limit."
        )


def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _load_excel(file_bytes: bytes) -> pd.DataFrame:
    # engine="openpyxl" explicitly: openpyxl does not execute macros,
    # which protects us from malicious macro-enabled Excel files.
    return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")


def _load_parquet(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(file_bytes))