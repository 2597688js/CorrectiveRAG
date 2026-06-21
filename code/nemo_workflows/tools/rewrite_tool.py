"""
Query rewrite tool for NEMO workflows.
Rewrites user question for web search when retrieval is insufficient.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import logging

logger = logging.getLogger(__name__)


class WebQuery(BaseModel):
    """LLM output for query rewriting."""

    query: str = Field(description="Rewritten query for web search")


class RewriteInput(BaseModel):
    """Input schema for rewrite tool."""

    question: str = Field(description="The original user question")


class RewriteOutput(BaseModel):
    """Output schema for rewrite tool."""

    web_query: str = Field(description="Rewritten query for web search")


# Global LLM instance (set by workflow builder)
_llm = None
_rewrite_chain = None


def set_llm(llm: ChatNVIDIA):
    """Set the LLM instance."""
    global _llm, _rewrite_chain
    _llm = llm

    # Build rewrite chain
    from prompts.rewrite_prompt import rewrite_prompt

    _rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


def get_llm():
    """Get the LLM instance."""
    return _llm


def get_rewrite_chain():
    """Get the rewrite chain."""
    return _rewrite_chain


async def rewrite_for_web_search(
    context: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Rewrite the user question for web search.

    Args:
        context: Workflow context dict
        **kwargs: Additional arguments

    Returns:
        Dict with 'web_query' key containing rewritten query
    """
    rewrite_chain = get_rewrite_chain()
    if not rewrite_chain:
        raise ValueError(
            "Rewrite chain not initialized. Call set_llm() before invoking workflow."
        )

    question = context.get("question")
    if not question:
        raise ValueError("No question in context")

    logger.debug(f"Rewriting question for web search")

    try:
        result = rewrite_chain.invoke({"question": question})
        logger.debug(f"Rewritten query: {result.query}")

        return {
            "web_query": result.query
        }

    except Exception as e:
        logger.error(f"Query rewriting failed: {str(e)}")
        raise
