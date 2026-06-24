"""
Author : janarddan
File_name = vector_store.py.py
Date : 21/06/26
Description :

"""
from typing import List

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

from config.nvidia import embedding_model, nvidia_base_url


class VectorStoreBuilder:
    def __init__(
        self,
        model: str | None = None,
    ):
        kwargs = {
            "model": model or embedding_model(),
        }

        base_url = nvidia_base_url()
        if base_url:
            kwargs["base_url"] = base_url

        self.embeddings = NVIDIAEmbeddings(**kwargs)

    def build(self, chunks: List[Document]) -> FAISS:
        return FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
