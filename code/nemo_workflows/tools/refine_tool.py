"""
Context refinement tool for NEMO workflows.
Filters and combines documents to create refined context.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import re
import logging

logger = logging.getLogger(__name__)


class KeepOrDrop(BaseModel):
    """LLM output for sentence filtering."""

    keep: bool = Field(description="Whether to keep this sentence")


class RefineInput(BaseModel):
    """Input schema for refine tool."""

    question: str = Field(description="The user question")
    good_docs: List[Document] = Field(description="Good documents from evaluation")
    web_docs: List[Document] = Field(description="Knowledge-supplemented documents")
    verdict: str = Field(description="Evaluation verdict")


class RefineOutput(BaseModel):
    """Output schema for refine tool."""

    strips: List[str] = Field(description="Decomposed sentences from documents")
    kept_strips: List[str] = Field(description="Sentences that passed filtering")
    refined_context: str = Field(description="Final refined context")


# Global LLM instance (set by workflow builder)
_llm = None
_filter_chain = None


def set_llm(llm: ChatNVIDIA):
    """Set the LLM instance."""
    global _llm, _filter_chain
    _llm = llm

    # Build filter chain
    from prompts.filter_prompt import filter_prompt

    _filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)


def get_llm():
    """Get the LLM instance."""
    return _llm


def get_filter_chain():
    """Get the filter chain."""
    return _filter_chain


def decompose_to_sentences(text: str) -> List[str]:
    """Decompose text into sentences."""
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]


async def refine_context(
    context: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Refine context by filtering and combining documents.

    Selects documents based on verdict and filters to relevant sentences.

    Args:
        context: Workflow context dict
        **kwargs: Additional arguments

    Returns:
        Dict with 'refined_context', 'strips', and 'kept_strips' keys
    """
    filter_chain = get_filter_chain()
    if not filter_chain:
        raise ValueError(
            "Filter chain not initialized. Call set_llm() before invoking workflow."
        )

    question = context.get("question")
    good_docs = context.get("good_docs", [])
    web_docs = context.get("web_docs", [])
    verdict = context.get("verdict", "AMBIGUOUS")

    if not question:
        raise ValueError("No question in context")

    logger.debug(f"Refining context (verdict: {verdict})")

    # Select documents based on verdict
    if verdict == "CORRECT":
        docs_to_use = good_docs
    elif verdict == "INCORRECT":
        docs_to_use = web_docs
    else:  # AMBIGUOUS
        docs_to_use = good_docs + web_docs

    if not docs_to_use:
        logger.warning("No documents to refine")
        return {
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
        }

    # Combine document content
    context_text = "\n\n".join(
        doc.page_content for doc in docs_to_use
    )

    # Decompose into sentences
    strips = decompose_to_sentences(context_text)
    logger.debug(f"Decomposed into {len(strips)} sentences")

    if not strips:
        logger.warning("No sentences to filter")
        return {
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
        }

    # Filter sentences using LLM
    kept_strips = []

    try:
        for sentence in strips:
            result = filter_chain.invoke(
                {
                    "question": question,
                    "sentence": sentence,
                }
            )

            if result.keep:
                kept_strips.append(sentence)

        logger.debug(f"Kept {len(kept_strips)}/{len(strips)} sentences")

        refined_context = "\n".join(kept_strips)

        return {
            "strips": strips,
            "kept_strips": kept_strips,
            "refined_context": refined_context,
        }

    except Exception as e:
        logger.error(f"Context refinement failed: {str(e)}")
        raise
