"""
Answer generation tool for NEMO workflows.
Generates the final answer based on refined context.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import logging

logger = logging.getLogger(__name__)


class GenerateInput(BaseModel):
    """Input schema for generate tool."""

    question: str = Field(description="The user question")
    refined_context: str = Field(description="Refined context for answer generation")


class GenerateOutput(BaseModel):
    """Output schema for generate tool."""

    answer: str = Field(description="The generated answer")


# Global LLM instance (set by workflow builder)
_llm = None
_generate_chain = None


def set_llm(llm: ChatNVIDIA):
    """Set the LLM instance."""
    global _llm, _generate_chain
    _llm = llm

    # Build generation chain
    from prompts.answer_prompt import answer_prompt

    _generate_chain = answer_prompt | llm


def get_llm():
    """Get the LLM instance."""
    return _llm


def get_generate_chain():
    """Get the generation chain."""
    return _generate_chain


async def generate_answer(
    context: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Generate the final answer based on refined context.

    Args:
        context: Workflow context dict
        **kwargs: Additional arguments

    Returns:
        Dict with 'answer' key containing the generated answer
    """
    generate_chain = get_generate_chain()
    if not generate_chain:
        raise ValueError(
            "Generation chain not initialized. Call set_llm() before invoking workflow."
        )

    question = context.get("question")
    refined_context = context.get("refined_context", "")

    if not question:
        raise ValueError("No question in context")

    logger.debug("Generating answer")

    try:
        result = generate_chain.invoke(
            {
                "question": question,
                "context": refined_context,
            }
        )

        answer = result.content
        logger.debug(f"Answer generated ({len(answer)} characters)")

        return {
            "answer": answer
        }

    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        raise
