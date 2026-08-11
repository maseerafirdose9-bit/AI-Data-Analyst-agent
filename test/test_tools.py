# tests/test_tools.py

import pandas as pd
import pytest

from ai_data_agent.ingestion.profiler import profile_dataset
from ai_data_agent.tools.dataset_tools import inspect_dataset, describe_dataset, DescribeDatasetInput
from ai_data_agent.tools.execution_tools import execute_python_code, ExecuteCodeInput
from ai_data_agent.tools.code_validation import validate_code_statically
from ai_data_agent.tools.result_validation import validate_columns_referenced, validate_result_shape
from ai_data_agent.tools.chart_tools import select_chart_type


class TestInspectDataset:
    def test_returns_correct_column_names(self, sample_df):
        profile = profile_dataset(sample_df)
        result = inspect_dataset(profile)
        assert set(result.column_names) == {"region", "product", "revenue", "order_date"}


class TestDescribeDataset:
    def test_describes_numeric_columns_by_default(self, sample_df):
        result = describe_dataset(sample_df, DescribeDatasetInput())
        assert "revenue" in result.stats
        assert result.warnings == []

    def test_warns_on_nonexistent_column_without_crashing(self, sample_df):
        result = describe_dataset(sample_df, DescribeDatasetInput(columns=["not_real"]))
        assert result.stats == {}
        assert "not_real" in result.warnings[0]


class TestStaticCodeValidation:
    """These tests are our security guarantee — they must never regress."""

    def test_allows_safe_pandas_code(self):
        is_valid, error = validate_code_statically(
            "result = df['revenue'].sum()", allowed_columns=["revenue"]
        )
        assert is_valid is True
        assert error is None

    def test_rejects_os_import(self):
        is_valid, error = validate_code_statically(
            "import os\nresult = os.getcwd()", allowed_columns=[]
        )
        assert is_valid is False
        assert "import" in error.lower()

    def test_rejects_dunder_attribute_access(self):
        is_valid, error = validate_code_statically(
            "result = ().__class__.__bases__", allowed_columns=[]
        )
        assert is_valid is False

    def test_rejects_eval_call(self):
        is_valid, error = validate_code_statically(
            "result = eval('1+1')", allowed_columns=[]
        )
        assert is_valid is False

    def test_rejects_code_without_result_assignment(self):
        is_valid, error = validate_code_statically(
            "x = df['revenue'].sum()", allowed_columns=["revenue"]
        )
        assert is_valid is False


class TestExecutePythonCode:
    def test_executes_valid_code_successfully(self, sample_df):
        output = execute_python_code(sample_df, ExecuteCodeInput(code="result = df['revenue'].sum()"))
        assert output.success is True
        assert output.result_value == pytest.approx(330.99)

    def test_blocks_dangerous_code_before_running(self, sample_df):
        output = execute_python_code(sample_df, ExecuteCodeInput(code="import os\nresult = 1"))
        assert output.success is False
        assert "import" in output.error_message.lower()

    def test_handles_runtime_error_gracefully(self, sample_df):
        # References a column that doesn't exist — should fail cleanly,
        # not crash the test process.
        output = execute_python_code(sample_df, ExecuteCodeInput(code="result = df['not_a_column'].sum()"))
        assert output.success is False
        assert output.error_message is not None

    def test_strips_markdown_fences_before_executing(self, sample_df):
        code_with_fences = "```python\nresult = df['revenue'].sum()\n```"
        output = execute_python_code(sample_df, ExecuteCodeInput(code=code_with_fences))
        assert output.success is True


class TestResultValidation:
    def test_flags_hallucinated_column_reference(self, sample_df):
        is_valid, error = validate_columns_referenced(
            "result = df['fake_column'].sum()", valid_columns=list(sample_df.columns)
        )
        assert is_valid is False

    def test_allows_real_column_reference(self, sample_df):
        is_valid, error = validate_columns_referenced(
            "result = df['revenue'].sum()", valid_columns=list(sample_df.columns)
        )
        assert is_valid is True

    def test_flags_empty_series_result(self):
        empty_series = pd.Series([], dtype=float)
        is_valid, error = validate_result_shape(empty_series)
        assert is_valid is False

    def test_flags_nan_result(self):
        is_valid, error = validate_result_shape(float("nan"))
        assert is_valid is False


class TestChartSelection:
    def test_scalar_gets_no_chart(self):
        assert select_chart_type(120.5) == "none"

    def test_small_series_gets_bar_chart(self, sample_df):
        grouped = sample_df.groupby("region")["revenue"].sum()
        assert select_chart_type(grouped) == "bar"

    def test_large_series_gets_histogram(self):
        big_series = pd.Series(range(20))
        assert select_chart_type(big_series) == "histogram"