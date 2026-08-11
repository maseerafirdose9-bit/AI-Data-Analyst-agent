class AnalystAgentError(Exception):
    """Base exception for all application-specific errors."""


class DataIngestionError(AnalystAgentError):
    """Raised when a file cannot be loaded or parsed."""


class DataValidationError(AnalystAgentError):
    """Raised when the dataset fails quality checks."""


class CodeGenerationError(AnalystAgentError):
    """Raised when the LLM fails to produce valid, executable code."""


class CodeExecutionError(AnalystAgentError):
    """Raised when generated code fails during sandboxed execution."""