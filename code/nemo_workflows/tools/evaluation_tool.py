"""
Evaluation tool for NEMO workflows.
Evaluates retrieved documents for relevance using LLM.
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import logging

logger = logging.getLogger(__name__)

# Configuration
UPPER_THRESHOLD = 0.7
LOWER_THRESHOLD = 0.3


class DocEvalScore(BaseModel):
    """LLM output for document evaluation."""

    score: float = Field(
        description="Relevance score between 0 and 1"
    )


class EvaluationInput(BaseModel):
    """Input schema for evaluation tool."""

    question: str = Field(description="The user question")
    docs: List[Document] = Field(description="Documents to evaluate")


class EvaluationOutput(BaseModel):
    """Output schema for evaluation tool."""

    good_docs: List[Dict[str, Any]] = Field(
        description="Documents that passed relevance threshold"
    )
    verdict: Literal["CORRECT", "INCORRECT", "AMBIGUOUS"] = Field(
        description="Evaluation verdict"
    )
    reason: str = Field(description="Reason for the verdict")


# Global LLM instance (set by workflow builder)
_llm = None
_eval_chain = None


def set_llm(llm: ChatNVIDIA):
    """Set the LLM instance."""
    global _llm, _eval_chain
    _llm = llm

    # Build evaluation chain
    from prompts.doc_eval_prompt import doc_eval_prompt

    _eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)


def get_llm():
    """Get the LLM instance."""
    return _llm


def get_eval_chain():
    """Get the evaluation chain."""
    return _eval_chain


async def evaluate_document_relevance(
    context: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate retrieved documents for relevance to the question.

    Args:
        context: Workflow context dict
        **kwargs: Additional arguments

    Returns:
        Dict with 'good_docs', 'verdict', and 'reason' keys
    """
    eval_chain = get_eval_chain()
    if not eval_chain:
        raise ValueError(
            "Evaluation chain not initialized. Call set_llm() before invoking workflow."
        )

    question = context.get("question")
    docs = context.get("docs", [])

    if not question:
        raise ValueError("No question in context")

    if not docs:
        logger.warning("No documents to evaluate")
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": "No documents retrieved",
        }

    logger.debug(f"Evaluating {len(docs)} documents")

    scores = []
    good_docs = []

    try:
        for doc in docs:
            result = eval_chain.invoke(
                {
                    "question": question,
                    "chunk": doc.page_content,
                }
            )

            scores.append(result.score)
            logger.debug(f"Document score: {result.score:.2f}")

            if result.score > LOWER_THRESHOLD:
                good_docs.append(doc)

        # Determine verdict based on scores
        if any(score > UPPER_THRESHOLD for score in scores):
            verdict = "CORRECT"
            reason = f"At least one chunk scored > {UPPER_THRESHOLD}"

        elif scores and all(score < LOWER_THRESHOLD for score in scores):
            verdict = "INCORRECT"
            reason = f"All chunks scored < {LOWER_THRESHOLD}"

        else:
            verdict = "AMBIGUOUS"
            reason = (
                f"No chunk scored > {UPPER_THRESHOLD}, "
                f"but not all scored < {LOWER_THRESHOLD}"
            )

        logger.debug(f"Evaluation verdict: {verdict}")

        return {
            "good_docs": good_docs,
            "verdict": verdict,
            "reason": reason,
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
