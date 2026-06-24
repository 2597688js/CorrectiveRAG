"""
Author : janarddan
File_name = nodes.py
Date : 21/06/26
Description :

"""

import re

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config.nvidia import llm_model, nvidia_base_url
from models.state import State
from models.doc_eval import DocEvalScore
from models.filter import KeepOrDrop
from models.web_query import WebQuery

from prompts.doc_eval_prompt import doc_eval_prompt
from prompts.filter_prompt import filter_prompt
from prompts.rewrite_prompt import rewrite_prompt
from prompts.answer_prompt import answer_prompt

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


UPPER_TH = 0.7
LOWER_TH = 0.3


llm_kwargs = {
    "model": llm_model(),
    "temperature": 0,
}

base_url = nvidia_base_url()
if base_url:
    llm_kwargs["base_url"] = base_url

llm = ChatNVIDIA(**llm_kwargs)

retriever = None


doc_eval_chain = (
    doc_eval_prompt
    | llm.with_structured_output(DocEvalScore)
)

filter_chain = (
    filter_prompt
    | llm.with_structured_output(KeepOrDrop)
)

rewrite_chain = (
    rewrite_prompt
    | llm.with_structured_output(WebQuery)
)


def retrieve_node(state: State):

    question = state["question"]

    return {
        "docs": retriever.invoke(question)
    }


def eval_each_doc_node(state: State):

    question = state["question"]

    scores = []
    good_docs = []

    for doc in state["docs"]:

        result = doc_eval_chain.invoke(
            {
                "question": question,
                "chunk": doc.page_content,
            }
        )

        scores.append(result.score)

        if result.score > LOWER_TH:
            good_docs.append(doc)

    if any(score > UPPER_TH for score in scores):

        return {
            "good_docs": good_docs,
            "verdict": "CORRECT",
            "reason": f"At least one chunk scored > {UPPER_TH}",
        }

    if scores and all(score < LOWER_TH for score in scores):

        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All chunks scored < {LOWER_TH}",
        }

    return {
        "good_docs": good_docs,
        "verdict": "AMBIGUOUS",
        "reason": (
            f"No chunk scored > {UPPER_TH}, "
            f"but not all scored < {LOWER_TH}"
        ),
    }


def decompose_to_sentences(text: str):

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


def rewrite_query_node(state: State):

    result = rewrite_chain.invoke(
        {
            "question": state["question"]
        }
    )

    return {
        "web_query": result.query
    }


def web_search_node(state: State):

    query = (
        state.get("web_query")
        or state["question"]
    )

    result = llm.invoke(
        (
            "Use your NVIDIA-hosted model knowledge to provide a concise "
            "retrieval supplement for this query. Include only facts that are "
            "useful for answering the user's question. If you are not "
            f"confident, say so explicitly.\n\nQuery: {query}"
        )
    )

    return {
        "web_docs": [
            Document(
                page_content=result.content,
                metadata={
                    "source": "nvidia_nim_supplement",
                    "query": query,
                },
            )
        ]
    }


def refine(state: State):

    question = state["question"]

    if state["verdict"] == "CORRECT":
        docs_to_use = state["good_docs"]

    elif state["verdict"] == "INCORRECT":
        docs_to_use = state["web_docs"]

    else:
        docs_to_use = (
            state["good_docs"]
            + state["web_docs"]
        )

    context = "\n\n".join(
        doc.page_content
        for doc in docs_to_use
    )

    strips = decompose_to_sentences(
        context
    )

    kept_strips = []

    for sentence in strips:

        result = filter_chain.invoke(
            {
                "question": question,
                "sentence": sentence,
            }
        )

        if result.keep:
            kept_strips.append(sentence)

    refined_context = "\n".join(
        kept_strips
    )

    return {
        "strips": strips,
        "kept_strips": kept_strips,
        "refined_context": refined_context,
    }


def generate(state: State):

    result = (
        answer_prompt
        | llm
    ).invoke(
        {
            "question": state["question"],
            "context": state["refined_context"],
        }
    )

    return {
        "answer": result.content
    }


def route_after_eval(state: State):

    if state["verdict"] == "CORRECT":
        return "refine"

    return "rewrite_query"
