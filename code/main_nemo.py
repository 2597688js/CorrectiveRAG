"""
NEMO-based CLI entry point for Corrective RAG.
Demonstrates async workflow execution with NEMO agentic toolkit.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add code to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.document_loader import DocumentLoader
from ingestion.document_chunker import DocumentChunker
from retrieval.vector_store_builder import VectorStoreBuilder
from retrieval.retriever_builder import RetrieverBuilder
from nemo_workflows.workflow_builder import build_nemo_workflow
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from config.nvidia import llm_model, embedding_model, nvidia_base_url
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def build_retriever(pdf_paths: list) -> object:
    """Build vector store retriever from PDF files."""
    loader = DocumentLoader(pdf_paths)
    docs = loader.load()

    chunker = DocumentChunker(
        chunk_size=400,
        chunk_overlap=100,
    )
    chunks = chunker.chunk(docs)

    vector_store = VectorStoreBuilder().build(chunks)
    retriever = RetrieverBuilder(k=4).build(vector_store)

    return retriever


def create_llm() -> ChatNVIDIA:
    """Create NVIDIA LLM instance."""
    llm_kwargs = {
        "model": llm_model(),
        "temperature": 0,
    }

    base_url = nvidia_base_url()
    if base_url:
        llm_kwargs["base_url"] = base_url

    return ChatNVIDIA(**llm_kwargs)


def create_initial_state(question: str) -> dict:
    """Create initial workflow state."""
    return {
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


async def async_main(pdf_paths: list, question: str) -> None:
    """Main async entry point."""
    print("\n" + "=" * 70)
    print("CORRECTIVE RAG - NEMO AGENTIC TOOLKIT")
    print("=" * 70)

    # Build components
    print("\n[1/3] Building retriever...")
    retriever = build_retriever(pdf_paths)
    print(f"      ✓ Retriever built")

    print("[2/3] Creating LLM...")
    llm = create_llm()
    print(f"      ✓ LLM initialized: {llm_model()}")

    print("[3/3] Building NEMO workflow...")
    workflow = build_nemo_workflow(llm=llm, retriever=retriever)
    print(f"      ✓ Workflow compiled")

    # Execute workflow
    print("\n" + "-" * 70)
    print(f"QUESTION: {question}")
    print("-" * 70)
    print("\n[WORKFLOW EXECUTION]\n")

    initial_state = create_initial_state(question)

    try:
        result = await workflow.invoke(initial_state)

        # Display results
        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(f"\n{result.get('answer', 'No answer generated')}\n")

        # Display debug info
        print("=" * 70)
        print("DEBUG INFORMATION")
        print("=" * 70)
        print(f"Verdict: {result.get('verdict', 'N/A')}")
        print(f"Reason: {result.get('reason', 'N/A')}")
        print(f"Web Query: {result.get('web_query', 'N/A')}")
        print(f"Documents Retrieved: {len(result.get('docs', []))}")
        print(f"Good Documents: {len(result.get('good_docs', []))}")
        print(f"Context Sentences: {len(result.get('strips', []))}")
        print(f"Kept Sentences: {len(result.get('kept_strips', []))}")

        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n", file=sys.stderr)
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Corrective RAG using NEMO Agentic Toolkit"
    )

    parser.add_argument(
        "--pdf",
        nargs="+",
        required=True,
        help="One or more PDF paths",
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask",
    )

    args = parser.parse_args()

    # Run async main
    try:
        asyncio.run(async_main(args.pdf, args.question))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
