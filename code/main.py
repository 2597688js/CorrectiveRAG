"""
Author : Janarddan Sarkar
File_name = main.py
Date : 21-06-2026
Description : Application entry point
"""
import argparse
from ingestion.document_loader import DocumentLoader
from ingestion.document_chunker import DocumentChunker

from retrieval.vector_store_builder import VectorStoreBuilder
from retrieval.retriever_builder import RetrieverBuilder

from graph.agent_builder import AgentBuilder

import graph.nodes as nodes

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def build_retriever(pdf_paths):

    loader = DocumentLoader(pdf_paths)

    docs = loader.load()

    chunker = DocumentChunker(
        chunk_size=900,
        chunk_overlap=150,
    )

    chunks = chunker.chunk(docs)

    vector_store = (
        VectorStoreBuilder()
        .build(chunks)
    )

    retriever = (
        RetrieverBuilder(k=4)
        .build(vector_store)
    )

    return retriever

def parse_args():

    parser = argparse.ArgumentParser()

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

    return parser.parse_args()

def main():

    args = parse_args()

    retriever = build_retriever(
        args.pdf
    )

    nodes.retriever = retriever

    app = AgentBuilder.build()

    result = app.invoke(
        {
            "question": args.question,
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

    print("\nANSWER:\n")
    print(result["answer"])


if __name__ == "__main__":
    main()
