"""
Knowledge supplement tool for NEMO workflows.
Uses NVIDIA NIM knowledge to supplement insufficient retrieval.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import logging

logger = logging.getLogger(__name__)


class SupplementInput(BaseModel):
    """Input schema for supplement tool."""

    query: str = Field(description="Query for knowledge supplementation")


class SupplementOutput(BaseModel):
    """Output schema for supplement tool."""

    web_docs: List[Dict[str, Any]] = Field(
        description="Knowledge-based documents (serialized)"
    )


# Global LLM instance (set by workflow builder)
_llm = None


def set_llm(llm: ChatNVIDIA):
    """Set the LLM instance."""
    global _llm
    _llm = llm


def get_llm():
    """Get the LLM instance."""
    return _llm


async def supplement_with_knowledge(
    context: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Supplement retrieval with NVIDIA NIM knowledge.

    Uses the NVIDIA-hosted model's knowledge base to provide additional
    context when retrieval is insufficient.

    Args:
        context: Workflow context dict
        **kwargs: Additional arguments

    Returns:
        Dict with 'web_docs' key containing knowledge-based documents
    """
    llm = get_llm()
    if not llm:
        raise ValueError(
            "LLM not set. Call set_llm() before invoking workflow."
        )

    query = context.get("web_query") or context.get("question")
    if not query:
        raise ValueError("No query in context")

    logger.debug(f"Supplementing knowledge for query: {query[:50]}...")

    try:
        result = llm.invoke(
            (
                "Use your NVIDIA-hosted model knowledge to provide a concise "
                "retrieval supplement for this query. Include only facts that are "
                "useful for answering the user's question. If you are not "
                f"confident, say so explicitly.\n\nQuery: {query}"
            )
        )

        logger.debug(f"Knowledge supplement retrieved")

        web_docs = [
            Document(
                page_content=result.content,
                metadata={
                    "source": "nvidia_nim_supplement",
                    "query": query,
                },
            )
        ]

        return {
            "web_docs": web_docs
        }

    except Exception as e:
        logger.error(f"Knowledge supplementation failed: {str(e)}")
        raise
