"""
Streamlit UI for Corrective RAG using NEMO Agentic Toolkit.
"""

import tempfile
import asyncio
from pathlib import Path

import streamlit as st

from nemo_workflows.workflow_builder import build_nemo_workflow
from ingestion.document_loader import DocumentLoader
from ingestion.document_chunker import DocumentChunker
from retrieval.vector_store_builder import VectorStoreBuilder
from retrieval.retriever_builder import RetrieverBuilder
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from config.nvidia import llm_model, nvidia_base_url
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# ==================================================
# Page Config
# ==================================================

st.set_page_config(
    page_title="Corrective RAG - NEMO",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Corrective RAG (NEMO Agentic Toolkit)")

st.caption(
    "Upload PDF documents and ask questions using NVIDIA NEMO agentic toolkit."
)


# ==================================================
# Helpers
# ==================================================

def build_retriever(pdf_paths):
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


def create_llm():
    """Create NVIDIA LLM instance."""
    llm_kwargs = {
        "model": llm_model(),
        "temperature": 0,
    }

    base_url = nvidia_base_url()
    if base_url:
        llm_kwargs["base_url"] = base_url

    return ChatNVIDIA(**llm_kwargs)


def create_initial_state(question):
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


# ==================================================
# Session State
# ==================================================

if "workflow" not in st.session_state:
    st.session_state.workflow = None

if "retriever_ready" not in st.session_state:
    st.session_state.retriever_ready = False


# ==================================================
# Upload PDFs
# ==================================================

uploaded_files = st.file_uploader(
    label="Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")

    if st.button(
        "Build Knowledge Base",
        type="primary",
    ):
        with st.spinner("Processing documents..."):
            try:
                temp_dir = tempfile.mkdtemp()
                pdf_paths = []

                for uploaded_file in uploaded_files:
                    pdf_path = Path(temp_dir) / uploaded_file.name

                    with open(pdf_path, "wb") as file:
                        file.write(uploaded_file.getbuffer())

                    pdf_paths.append(str(pdf_path))

                # Build retriever and LLM
                retriever = build_retriever(pdf_paths)
                llm = create_llm()

                # Build NEMO workflow
                workflow = build_nemo_workflow(llm=llm, retriever=retriever)

                st.session_state.workflow = workflow
                st.session_state.retriever_ready = True

                st.success("Knowledge Base Created Successfully!")

            except Exception as e:
                st.error(f"Error building knowledge base: {str(e)}")


# ==================================================
# Question Answering
# ==================================================

if st.session_state.retriever_ready:
    st.divider()

    question = st.text_input(
        label="Ask a Question",
        placeholder="What is LoRA?",
    )

    if st.button("Submit Question"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    workflow = st.session_state.workflow
                    initial_state = create_initial_state(question)

                    # Run async workflow
                    result = asyncio.run(
                        workflow.invoke(initial_state)
                    )

                    # Display answer
                    st.subheader("Answer")
                    st.write(result["answer"])

                    # Display debug info
                    with st.expander("Debug Information"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Evaluation**")
                            st.write(f"Verdict: {result['verdict']}")
                            st.write(f"Reason: {result['reason']}")

                        with col2:
                            st.write("**Documents**")
                            st.write(f"Retrieved: {len(result['docs'])}")
                            st.write(f"Good: {len(result['good_docs'])}")
                            st.write(f"Web Docs: {len(result['web_docs'])}")

                        st.write("**Query Rewriting**")
                        st.write(f"Web Query: {result['web_query']}")

                        st.write("**Context Refinement**")
                        st.write(f"Total Sentences: {len(result['strips'])}")
                        st.write(f"Kept Sentences: {len(result['kept_strips'])}")

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")


# ==================================================
# Footer
# ==================================================

st.divider()

st.caption(
    "Corrective RAG powered by NVIDIA NIM, NEMO Agentic Toolkit, and FAISS."
)
