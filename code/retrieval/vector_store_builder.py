"""
Author : janarddan
File_name = vector_store.py.py
Date : 21/06/26
Description :

"""
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStoreBuilder:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
    ):
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model
        )

    def build(self, chunks: List[Document]) -> FAISS:
        return FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )