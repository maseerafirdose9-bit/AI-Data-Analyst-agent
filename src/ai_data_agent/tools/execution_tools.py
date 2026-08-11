# src/ai_data_agent/tools/execution_tools.py

import logging
import signal
import platform
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from ai_data_agent.tools.code_validation import validate_code_statically

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT_SECONDS = 10


class ExecuteCodeInput(BaseModel):
    code: str = Field(description="Python/Pandas code to execute against the dataset.")


class ExecuteCodeOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    result_repr: str | None = None
    result_value: Any = None  # the real Python/pandas object, not just its string repr
    error_message: str | None = None


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Code execution exceeded the time limit.")


def execute_python_code(df: pd.DataFrame, input_data: ExecuteCodeInput) -> ExecuteCodeOutput:
    """
    Executes LLM-generated Pandas code inside a restricted namespace,
    after static validation. Only `pd` and `df` are exposed — no
    builtins like open(), no imports, no filesystem/network access.
    """
    code = _strip_markdown_fences(input_data.code)

    is_valid, error = validate_code_statically(code, allowed_columns=list(df.columns))
    if not is_valid:
        logger.warning("Static validation rejected code: %s", error)
        return ExecuteCodeOutput(success=False, error_message=error)

    safe_builtins = {
        "len": len, "range": range, "sum": sum, "min": min, "max": max,
        "sorted": sorted, "list": list, "dict": dict, "str": str,
        "int": int, "float": float, "bool": bool, "round": round,
        "abs": abs, "enumerate": enumerate, "zip": zip,
    }
    local_namespace = {"df": df, "pd": pd}
    global_namespace = {"__builtins__": safe_builtins}

    use_signal_timeout = platform.system() != "Windows"
    if use_signal_timeout:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(EXECUTION_TIMEOUT_SECONDS)

    try:
        exec(code, global_namespace, local_namespace)  # noqa: S102 — intentional, sandboxed
        result = local_namespace.get("result")

        if result is None:
            return ExecuteCodeOutput(
                success=False,
                error_message="Code executed but did not assign a `result` variable.",
            )

        if isinstance(result, str) and result.startswith("UNANSWERABLE"):
            return ExecuteCodeOutput(success=False, error_message=result)

        return ExecuteCodeOutput(success=True, result_repr=repr(result), result_value=result)

    except _TimeoutError as exc:
        logger.error("Code execution timed out.")
        return ExecuteCodeOutput(success=False, error_message=str(exc))
    except Exception as exc:
        logger.error("Code execution failed: %s", exc)
        return ExecuteCodeOutput(success=False, error_message=f"{type(exc).__name__}: {exc}")
    finally:
        if use_signal_timeout:
            signal.alarm(0)


def _strip_markdown_fences(code: str) -> str:
    """Defensively strips ```python / ``` fences the LLM sometimes adds
    despite being told not to — never trust instructions alone."""
    stripped = code.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped