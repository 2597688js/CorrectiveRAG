"""
NeMO Agent Toolkit registration for the corrective RAG workflow.
"""

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class CorrectiveRagConfig(FunctionBaseConfig, name="corrective_rag"):
    pdf_paths: list[str] = Field(
        default_factory=list,
        description="PDF files to index before running the CRAG workflow.",
    )


@register_function(config_type=CorrectiveRagConfig)
async def corrective_rag(config: CorrectiveRagConfig, builder: Builder):
    from graph.agent_builder import AgentBuilder
    from main import build_retriever
    import graph.nodes as nodes

    if not config.pdf_paths:
        raise ValueError("corrective_rag requires at least one PDF path.")

    retriever = build_retriever(config.pdf_paths)
    nodes.retriever = retriever
    app = AgentBuilder.build()

    async def _run(question: str) -> str:
        """
        Answer a question using the corrective RAG workflow over configured PDFs.
        """
        result = app.invoke(
            {
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
        )

        return result["answer"]

    yield _run
