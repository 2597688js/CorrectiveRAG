"""
Retrieval tool for NEMO workflows.
Retrieves documents from vector store based on question.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class RetrievalInput(BaseModel):
    """Input schema for retrieval tool."""

    question: str = Field(description="The user question")


class RetrievalOutput(BaseModel):
    """Output schema for retrieval tool."""

    docs: List[Dict[str, Any]] = Field(
        description="Retrieved documents (serialized)"
    )


# Global retriever instance (set by workflow builder)
_retriever = None


def set_retriever(retriever):
    """Set the retriever instance."""
    global _retriever
    _retriever = retriever


def get_retriever():
    """Get the retriever instance."""
    return _retriever


async def retrieve_documents(
    context: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Retrieve documents from vector store based on question.

    Args:
        context: Workflow context dict
        **kwargs: Additional arguments

    Returns:
        Dict with 'docs' key containing list of Document objects
    """
    retriever = get_retriever()
    if not retriever:
        raise ValueError(
            "Retriever not set. Call set_retriever() before invoking workflow."
        )

    question = context.get("question")
    if not question:
        raise ValueError("No question in context")

    logger.debug(f"Retrieving documents for question: {question[:50]}...")

    try:
        docs = retriever.invoke(question)
        logger.debug(f"Retrieved {len(docs)} documents")

        return {
            "docs": docs
        }

    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        raise
