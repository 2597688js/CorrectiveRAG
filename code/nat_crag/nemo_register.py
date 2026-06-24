"""
NEMO Agent Toolkit registration for Corrective RAG workflow.
Enables running the NEMO CRAG workflow via nat CLI.
"""

import asyncio
from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class CorrectiveRagNemoConfig(FunctionBaseConfig, name="corrective_rag_nemo"):
    """Configuration for NEMO-based Corrective RAG workflow."""

    pdf_paths: list[str] = Field(
        default_factory=list,
        description="PDF files to index before running the CRAG workflow.",
    )


@register_function(config_type=CorrectiveRagNemoConfig)
async def corrective_rag_nemo(config: CorrectiveRagNemoConfig, builder: Builder):
    """
    Corrective RAG workflow using NEMO Agentic Toolkit.

    Args:
        config: Workflow configuration
        builder: NAT Builder instance

    Yields:
        Async function that answers questions based on configured PDFs
    """
    from ingestion.document_loader import DocumentLoader
    from ingestion.document_chunker import DocumentChunker
    from retrieval.vector_store_builder import VectorStoreBuilder
    from retrieval.retriever_builder import RetrieverBuilder
    from nemo_workflows.workflow_builder import build_nemo_workflow
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from config.nvidia import llm_model, nvidia_base_url

    if not config.pdf_paths:
        raise ValueError("corrective_rag_nemo requires at least one PDF path.")

    # Build retriever
    loader = DocumentLoader(config.pdf_paths)
    docs = loader.load()

    chunker = DocumentChunker(
        chunk_size=400,
        chunk_overlap=100,
    )
    chunks = chunker.chunk(docs)

    vector_store = VectorStoreBuilder().build(chunks)
    retriever = RetrieverBuilder(k=4).build(vector_store)

    # Create LLM
    llm_kwargs = {
        "model": llm_model(),
        "temperature": 0,
    }

    base_url = nvidia_base_url()
    if base_url:
        llm_kwargs["base_url"] = base_url

    llm = ChatNVIDIA(**llm_kwargs)

    # Build workflow
    workflow = build_nemo_workflow(llm=llm, retriever=retriever)

    async def _run(question: str) -> str:
        """
        Answer a question using the Corrective RAG workflow.

        Args:
            question: User question

        Returns:
            Generated answer
        """
        initial_state = {
            "question": question,
            "docs": [],
            "good_docs": [],
            "verdict": "",
            "reason": "",
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
            "web_query": "",
            "web_docs": [],
            "answer": "",
        }

        result = await workflow.invoke(initial_state)
        return result["answer"]

    yield _run
