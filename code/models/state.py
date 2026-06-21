"""
Author : janarddan
File_name = state.py.py
Date : 21/06/26
Description :

"""

from typing import List, TypedDict, Literal

from langchain_core.documents import Document

# -----------------------------
# State
# -----------------------------
class State(TypedDict):
    question: str

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_query: str

    web_docs: List[Document]

    answer: str
