# src/ai_data_agent/tools/code_validation.py

import ast
import logging

logger = logging.getLogger(__name__)

FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "shutil", "socket", "requests",
    "open", "eval", "exec", "compile", "__import__",
    "input", "exit", "quit", "globals", "locals", "vars",
}

FORBIDDEN_ATTRIBUTE_PREFIXES = ("__",)  # blocks dunder attribute access like __class__


def validate_code_statically(code: str, allowed_columns: list[str]) -> tuple[bool, str | None]:
    """
    Parses generated code into an AST and checks for dangerous patterns
    before any execution is attempted.

    Returns (is_valid, error_message).
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return False, f"Generated code has a syntax error: {exc}"

    for node in ast.walk(tree):
        # Block any import statement entirely — the prompt already says
        # not to import anything; this enforces it structurally.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Generated code contains a disallowed import statement."

        # Block calls/references to forbidden names
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"Generated code references a disallowed name: '{node.id}'"

        # Block dunder attribute access (a common sandbox-escape technique,
        # e.g. ().__class__.__bases__ tricks)
        if isinstance(node, ast.Attribute) and node.attr.startswith(FORBIDDEN_ATTRIBUTE_PREFIXES):
            return False, f"Generated code accesses a disallowed attribute: '{node.attr}'"

    if not code.strip().count("result"):
        return False, "Generated code does not assign a `result` variable."

    logger.info("Static code validation passed.")
    return True, None