# src/ai_data_agent/tools/chart_tools.py

import logging

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


def select_chart_type(result: object) -> str:
    """
    Deterministically picks a chart type based on the shape/type of
    the execution result. No LLM involved — this is a rules engine.
    """
    if isinstance(result, (int, float)):
        return "none"  # a single number doesn't need a chart

    if isinstance(result, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(result.index):
            return "line"
        if len(result) > 15:
            return "histogram"
        return "bar"

    if isinstance(result, pd.DataFrame):
        numeric_cols = result.select_dtypes(include="number").columns
        if len(numeric_cols) >= 2:
            return "scatter"
        return "bar"

    return "none"


def build_chart(result: object, chart_type: str, title: str) -> dict | None:
    """
    Builds a Plotly figure spec (as a JSON-serializable dict) for the
    given result and chart type. Returns None if no chart applies.
    """
    if chart_type == "none" or result is None:
        return None

    try:
        if chart_type == "bar" and isinstance(result, pd.Series):
            fig = px.bar(x=result.index, y=result.values, title=title)
        elif chart_type == "line" and isinstance(result, pd.Series):
            fig = px.line(x=result.index, y=result.values, title=title)
        elif chart_type == "histogram" and isinstance(result, pd.Series):
            fig = px.histogram(result, title=title)
        elif chart_type == "scatter" and isinstance(result, pd.DataFrame):
            numeric_cols = result.select_dtypes(include="number").columns[:2]
            fig = px.scatter(result, x=numeric_cols[0], y=numeric_cols[1], title=title)
        else:
            logger.warning("No chart builder matched chart_type=%s, result_type=%s",
                            chart_type, type(result))
            return None

        return fig.to_dict()
    except Exception as exc:
        logger.error("Chart generation failed: %s", exc)
        return None