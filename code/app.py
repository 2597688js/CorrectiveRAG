"""
Author : Janarddan Sarkar
File_name : streamlit_app.py
Date : 21-06-2026
Description : Streamlit UI for Corrective RAG
"""

import tempfile
from pathlib import Path

import streamlit as st

from graph.agent_builder import AgentBuilder

from ingestion.document_loader import DocumentLoader
from ingestion.document_chunker import DocumentChunker

from retrieval.vector_store_builder import VectorStoreBuilder
from retrieval.retriever_builder import RetrieverBuilder

import graph.nodes as nodes


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Corrective RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Corrective RAG")
st.caption(
    "Upload one or more PDF documents and ask questions."
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

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


def create_initial_state(question):
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


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "app" not in st.session_state:
    st.session_state.app = None

if "retriever_ready" not in st.session_state:
    st.session_state.retriever_ready = False


# --------------------------------------------------
# Upload PDFs
# --------------------------------------------------

uploaded_files = st.file_uploader(
    label="Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} file(s) uploaded."
    )

    if st.button(
        "Build Knowledge Base",
        type="primary",
    ):

        with st.spinner(
            "Processing documents..."
        ):

            temp_dir = tempfile.mkdtemp()

            pdf_paths = []

            for uploaded_file in uploaded_files:

                pdf_path = (
                    Path(temp_dir)
                    / uploaded_file.name
                )

                with open(
                    pdf_path,
                    "wb",
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                pdf_paths.append(
                    str(pdf_path)
                )

            retriever = build_retriever(
                pdf_paths
            )

            nodes.retriever = retriever

            st.session_state.app = (
                AgentBuilder.build()
            )

            st.session_state.retriever_ready = True

        st.success(
            "Knowledge Base Created Successfully!"
        )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if st.session_state.retriever_ready:

    st.divider()

    question = st.text_input(
        label="Ask a Question",
        placeholder="What is the publication about?",
    )

    if st.button(
        "Submit Question"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Generating answer..."
            ):

                result = (
                    st.session_state.app.invoke(
                        create_initial_state(
                            question
                        )
                    )
                )

            st.subheader(
                "Answer"
            )

            st.write(
                result["answer"]
            )

            with st.expander(
                "Debug Information"
            ):

                st.write(
                    {
                        "verdict": result["verdict"],
                        "reason": result["reason"],
                        "web_query": result["web_query"],
                    }
                )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Corrective RAG powered by LangGraph, FAISS, OpenAI and Tavily."
)
