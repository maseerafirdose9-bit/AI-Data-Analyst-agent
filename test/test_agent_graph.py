# tests/test_agent_graph.py

from unittest.mock import MagicMock, patch

import pandas as pd

from ai_data_agent.agent.graph import build_graph
from ai_data_agent.ingestion.profiler import profile_dataset


def _make_fake_llm_response(text: str):
    """Builds a fake LangChain-style response object with a .content attribute."""
    fake_response = MagicMock()
    fake_response.content = text
    return fake_response


class TestAgentGraphHappyPath:
    @patch("ai_data_agent.agent.graph.get_llm")
    def test_full_pipeline_with_valid_code(self, mock_get_llm, sample_df):
        """
        Simulates a full successful run: the LLM 'returns' pre-written
        code, a 'YES' semantic verdict, and an insight sentence — all
        without any real network call.
        """
        mock_llm = MagicMock()
        # invoke() gets called 3 times in a successful run: code_gen,
        # semantic_check, insight_gen — we return a different canned
        # response each time, in that order.
        mock_llm.invoke.side_effect = [
            _make_fake_llm_response("result = df['revenue'].sum()"),
            _make_fake_llm_response("YES"),
            _make_fake_llm_response("Total revenue is $330.99."),
        ]
        mock_get_llm.return_value = mock_llm

        profile = profile_dataset(sample_df)
        app = build_graph()

        state = {
            "user_question": "What is the total revenue?",
            "dataframe": sample_df,
            "dataset_profile": profile,
            "conversation_history": [],
            "retry_count": 0,
        }
        result = app.invoke(state)

        assert result["final_response"]["success"] is True
        assert result["final_response"]["code"] == "result = df['revenue'].sum()"
        assert result["final_response"]["insight"] == "Total revenue is $330.99."


class TestAgentGraphRetryLogic:
    @patch("ai_data_agent.agent.graph.get_llm")
    def test_retries_on_dangerous_code_then_succeeds(self, mock_get_llm, sample_df):
        """
        Simulates the LLM producing dangerous code on the first attempt
        (rejected by static validation), then valid code on retry —
        proving our retry loop from Phase 5 actually works.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            _make_fake_llm_response("import os\nresult = 1"),       # attempt 1: rejected
            _make_fake_llm_response("result = df['revenue'].sum()"),  # attempt 2: valid
            _make_fake_llm_response("YES"),
            _make_fake_llm_response("Total revenue is $330.99."),
        ]
        mock_get_llm.return_value = mock_llm

        profile = profile_dataset(sample_df)
        app = build_graph()

        state = {
            "user_question": "What is the total revenue?",
            "dataframe": sample_df,
            "dataset_profile": profile,
            "conversation_history": [],
            "retry_count": 0,
        }
        result = app.invoke(state)

        assert result["final_response"]["success"] is True
        assert mock_llm.invoke.call_count == 4  # confirms a retry actually happened

    @patch("ai_data_agent.agent.graph.get_llm")
    def test_gives_up_gracefully_after_max_retries(self, mock_get_llm, sample_df):
        """
        Simulates the LLM producing dangerous code every single time —
        proving we don't loop forever and DO fail gracefully.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_fake_llm_response("import os\nresult = 1")
        mock_get_llm.return_value = mock_llm

        profile = profile_dataset(sample_df)
        app = build_graph()

        state = {
            "user_question": "What is the total revenue?",
            "dataframe": sample_df,
            "dataset_profile": profile,
            "conversation_history": [],
            "retry_count": 0,
        }
        result = app.invoke(state)

        assert result["final_response"]["success"] is False
        assert result["final_response"]["error"] is not None