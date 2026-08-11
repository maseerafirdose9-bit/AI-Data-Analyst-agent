# src/ai_data_agent/agent/graph.py

import logging

from langgraph.graph import StateGraph, END

from ai_data_agent.agent.state import AgentState
from ai_data_agent.agent.llm import get_llm
from ai_data_agent.agent.prompts import (
    build_code_generation_prompt,
    build_insight_prompt,
    build_semantic_check_prompt,
)
from ai_data_agent.tools.dataset_tools import inspect_dataset, describe_dataset, DescribeDatasetInput
from ai_data_agent.tools.execution_tools import execute_python_code, ExecuteCodeInput, _strip_markdown_fences
from ai_data_agent.tools.code_validation import validate_code_statically
from ai_data_agent.tools.chart_tools import select_chart_type, build_chart
from ai_data_agent.tools.result_validation import validate_columns_referenced, validate_result_shape

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def dataset_understanding_node(state: AgentState) -> dict:
    logger.info("Node: dataset_understanding")
    return {}


def query_planning_node(state: AgentState) -> dict:
    logger.info("Node: query_planning")

    profile = state["dataset_profile"]
    df = state["dataframe"]

    inspection = inspect_dataset(profile)
    description = describe_dataset(df, DescribeDatasetInput())

    logger.info("Inspected dataset: %d columns", inspection.column_count)
    logger.info("Described dataset stats for: %s", list(description.stats.keys()))

    plan = f"Question: {state['user_question']} | Schema known: {inspection.column_names}"
    return {"query_plan": plan}


def code_generation_node(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0)
    logger.info("Node: code_generation (retry_count=%d)", retry_count)

    profile = state["dataset_profile"]
    llm = get_llm()

    system_prompt, user_prompt = build_code_generation_prompt(
        schema_text=profile.to_prompt_context(),
        question=state["user_question"],
        history=state.get("conversation_history"),
    )

    if retry_count > 0 and state.get("validation_message"):
        user_prompt += (
            f"\n\nYour previous attempt failed with this error: "
            f"{state['validation_message']}\nPlease fix it and try again."
        )

    response = llm.invoke([
        ("system", system_prompt),
        ("human", user_prompt),
    ])

    return {
        "generated_code": _extract_text_content(response.content),
        "retry_count": retry_count + 1,
    }


def code_validation_node(state: AgentState) -> dict:
    logger.info("Node: code_validation")

    code = _strip_markdown_fences(state["generated_code"])
    columns = [c.name for c in state["dataset_profile"].columns]

    is_valid, error = validate_code_statically(code, allowed_columns=columns)
    if not is_valid:
        return {"is_valid": False, "validation_message": error}

    is_valid, error = validate_columns_referenced(code, columns)
    return {"is_valid": is_valid, "validation_message": error}


def safe_execution_node(state: AgentState) -> dict:
    logger.info("Node: safe_execution")

    df = state["dataframe"]
    output = execute_python_code(df, ExecuteCodeInput(code=state["generated_code"]))

    if not output.success:
        return {"execution_result": None, "execution_result_obj": None, "execution_error": output.error_message}

    return {
        "execution_result": output.result_repr,
        "execution_result_obj": output.result_value,
        "execution_error": None,
    }


def result_validation_node(state: AgentState) -> dict:
    logger.info("Node: result_validation")

    if state.get("execution_error"):
        return {"is_valid": False, "validation_message": state["execution_error"]}

    result_obj = state.get("execution_result_obj")

    is_valid, error = validate_result_shape(result_obj)
    if not is_valid:
        return {"is_valid": False, "validation_message": error}

    llm = get_llm()
    system_prompt, user_prompt = build_semantic_check_prompt(
        question=state["user_question"],
        code=state["generated_code"],
    )
    response = llm.invoke([
        ("system", system_prompt),
        ("human", user_prompt),
    ])
    verdict = _extract_text_content(response.content).strip().upper()
    logger.info("Semantic check verdict: %s", verdict)

    if "NO" in verdict:
        return {
            "is_valid": False,
            "validation_message": "The generated code does not fully answer the question as asked.",
        }

    return {"is_valid": True, "validation_message": None}


def chart_selection_node(state: AgentState) -> dict:
    logger.info("Node: chart_selection")

    result_obj = state.get("execution_result_obj")
    chart_type = select_chart_type(result_obj)
    chart_spec = build_chart(result_obj, chart_type, title=state["user_question"])

    logger.info("Selected chart_type=%s (chart_built=%s)", chart_type, chart_spec is not None)
    return {"chart_type": chart_type, "chart_spec": chart_spec}


def insight_generation_node(state: AgentState) -> dict:
    logger.info("Node: insight_generation")

    llm = get_llm()
    system_prompt, user_prompt = build_insight_prompt(
        question=state["user_question"],
        result_repr=state.get("execution_result", "No result available."),
    )

    response = llm.invoke([
        ("system", system_prompt),
        ("human", user_prompt),
    ])

    return {"insight_text": _extract_text_content(response.content)}


def final_response_node(state: AgentState) -> dict:
    logger.info("Node: final_response")

    # Append this turn to conversation history for the NEXT invocation
    # to pick up (the caller is responsible for passing this forward —
    # see test script below).
    new_turn = {
        "question": state["user_question"],
        "code": state.get("generated_code", ""),
        "result_repr": state.get("execution_result", ""),
    }
    updated_history = list(state.get("conversation_history", [])) + [new_turn]

    return {
        "final_response": {
            "code": state.get("generated_code"),
            "chart_type": state.get("chart_type"),
            "chart_spec": state.get("chart_spec"),
            "insight": state.get("insight_text"),
            "success": state.get("is_valid", False),
            "error": state.get("validation_message") if not state.get("is_valid", True) else None,
        },
        "conversation_history": updated_history,
    }


def route_after_validation(state: AgentState) -> str:
    if state.get("is_valid"):
        return "proceed"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        logger.warning("Max retries exceeded — routing to final_response with failure")
        return "give_up"
    return "retry"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("dataset_understanding", dataset_understanding_node)
    graph.add_node("query_planning", query_planning_node)
    graph.add_node("code_generation", code_generation_node)
    graph.add_node("code_validation", code_validation_node)
    graph.add_node("safe_execution", safe_execution_node)
    graph.add_node("result_validation", result_validation_node)
    graph.add_node("chart_selection", chart_selection_node)
    graph.add_node("insight_generation", insight_generation_node)
    graph.add_node("final_response", final_response_node)

    graph.set_entry_point("dataset_understanding")

    graph.add_edge("dataset_understanding", "query_planning")
    graph.add_edge("query_planning", "code_generation")
    graph.add_edge("code_generation", "code_validation")

    graph.add_conditional_edges(
        "code_validation",
        route_after_validation,
        {"proceed": "safe_execution", "retry": "code_generation", "give_up": "final_response"},
    )

    graph.add_edge("safe_execution", "result_validation")

    graph.add_conditional_edges(
        "result_validation",
        route_after_validation,
        {"proceed": "chart_selection", "retry": "code_generation", "give_up": "final_response"},
    )

    graph.add_edge("chart_selection", "insight_generation")
    graph.add_edge("insight_generation", "final_response")
    graph.add_edge("final_response", END)

    return graph.compile()