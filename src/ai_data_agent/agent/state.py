# src/ai_data_agent/agent/state.py

from typing import TypedDict, Optional
import pandas as pd

from ai_data_agent.ingestion.profiler import DatasetProfile


class ConversationTurn(TypedDict):
    """A single past exchange, kept minimal on purpose (see Phase 10 notes
    on bounding context size — we don't store the full chat transcript)."""
    question: str
    code: str
    result_repr: str


class AgentState(TypedDict):
    """
    Shared state passed between every node in the LangGraph workflow.
    """

    # --- Input ---
    user_question: str
    dataframe: pd.DataFrame
    dataset_profile: DatasetProfile

    # --- Conversation memory ---
    conversation_history: list[ConversationTurn]

    # --- Query Planning output ---
    query_plan: Optional[str]

    # --- Code Generation output ---
    generated_code: Optional[str]
    retry_count: int

    # --- Execution output ---
    execution_result: Optional[object]
    execution_result_obj: Optional[object]
    execution_error: Optional[str]

    # --- Validation output ---
    is_valid: bool
    validation_message: Optional[str]

    # --- Chart & Insight output ---
    chart_type: Optional[str]
    chart_spec: Optional[dict]
    insight_text: Optional[str]

    # --- Final ---
    final_response: Optional[dict]