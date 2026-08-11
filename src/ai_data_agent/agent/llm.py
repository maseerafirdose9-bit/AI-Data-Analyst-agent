# src/ai_data_agent/agent/llm.py

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from ai_data_agent.config import settings

logger = logging.getLogger(__name__)

_llm_instance = None


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Returns a shared, lazily-initialized LLM client.
    Reused across calls rather than re-instantiated every time,
    since creating a new client per-call has unnecessary overhead.
    """
    global _llm_instance
    if _llm_instance is None:
        logger.info("Initializing LLM client: model=%s", settings.llm_model_name)
        _llm_instance = ChatGoogleGenerativeAI(
            model=settings.llm_model_name,
            google_api_key=settings.google_api_key,
            temperature=0,
            max_tokens=1024,
        )
    return _llm_instance