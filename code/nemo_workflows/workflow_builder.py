"""
Workflow builder for NEMO CRAG workflow.
Constructs and configures the complete Corrective RAG workflow.
"""

from typing import Optional, Any, Dict
from .base_workflow import NemoWorkflow
from .tools import (
    retrieve_documents,
    evaluate_document_relevance,
    rewrite_for_web_search,
    supplement_with_knowledge,
    refine_context,
    generate_answer,
)
from . import tools as tools_module
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import logging

logger = logging.getLogger(__name__)


class NemoCragWorkflowBuilder:
    """Builder for NEMO-based Corrective RAG workflow."""

    def __init__(self):
        self.workflow: Optional[NemoWorkflow] = None
        self.llm: Optional[ChatNVIDIA] = None
        self.retriever: Optional[Any] = None

    def with_llm(self, llm: ChatNVIDIA) -> "NemoCragWorkflowBuilder":
        """Set the LLM for the workflow."""
        self.llm = llm
        return self

    def with_retriever(self, retriever: Any) -> "NemoCragWorkflowBuilder":
        """Set the retriever for the workflow."""
        self.retriever = retriever
        return self

    def build(self) -> NemoWorkflow:
        """Build and return the configured workflow."""
        if not self.llm:
            raise ValueError("LLM not set. Call with_llm() first.")
        if not self.retriever:
            raise ValueError("Retriever not set. Call with_retriever() first.")

        logger.info("Building NEMO CRAG Workflow")

        # Initialize workflow
        self.workflow = NemoWorkflow(name="corrective_rag_nemo")

        # Initialize all tool modules with LLM and retriever
        tools_module.retrieval_tool.set_retriever(self.retriever)
        tools_module.evaluation_tool.set_llm(self.llm)
        tools_module.rewrite_tool.set_llm(self.llm)
        tools_module.supplement_tool.set_llm(self.llm)
        tools_module.refine_tool.set_llm(self.llm)
        tools_module.generate_tool.set_llm(self.llm)

        # Register tools with workflow
        self._register_tools()

        # Set execution order and compile
        self._set_execution_order()
        self.workflow.compile()

        logger.info("NEMO CRAG Workflow built successfully")
        return self.workflow

    def _register_tools(self) -> None:
        """Register all tools with the workflow."""
        logger.debug("Registering tools")

        # Retrieval tool
        self.workflow.register_tool(
            name="retrieve",
            description="Retrieve documents from vector store",
            func=retrieve_documents,
        )

        # Evaluation tool
        self.workflow.register_tool(
            name="evaluate",
            description="Evaluate documents for relevance",
            func=evaluate_document_relevance,
        )

        # Rewrite tool
        self.workflow.register_tool(
            name="rewrite_query",
            description="Rewrite query for web search",
            func=rewrite_for_web_search,
        )

        # Supplement tool
        self.workflow.register_tool(
            name="web_search",
            description="Supplement with knowledge",
            func=supplement_with_knowledge,
        )

        # Refine tool
        self.workflow.register_tool(
            name="refine",
            description="Refine context by filtering sentences",
            func=refine_context,
        )

        # Generate tool
        self.workflow.register_tool(
            name="generate",
            description="Generate final answer",
            func=generate_answer,
        )

        logger.debug("All tools registered")

    def _set_execution_order(self) -> None:
        """Set the execution order of tools."""
        logger.debug("Setting execution order")

        # Main execution order (with conditional routing point)
        execution_order = [
            "retrieve",
            "evaluate",
            "conditional_route",  # Routes based on verdict
            "refine",
            "generate",
        ]

        # Also need to add rewrite and web_search to the order
        # They'll be inserted dynamically by the conditional routing
        self.workflow.set_execution_order(execution_order)

        logger.debug(f"Execution order: {execution_order}")


def build_nemo_workflow(
    llm: ChatNVIDIA,
    retriever: Any,
) -> NemoWorkflow:
    """
    Convenience function to build a NEMO CRAG workflow.

    Args:
        llm: NVIDIA LLM instance
        retriever: Document retriever instance

    Returns:
        Compiled NEMO workflow
    """
    builder = NemoCragWorkflowBuilder()
    return (
        builder
        .with_llm(llm)
        .with_retriever(retriever)
        .build()
    )
