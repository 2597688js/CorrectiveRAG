"""
Author : janarddan
File_name = retriver.py.py
Date : 21/06/26
Description :

"""
from langchain_core.vectorstores import VectorStoreRetriever


class RetrieverBuilder:
    def __init__(
        self,
        search_type: str = "similarity",
        k: int = 4,
    ):
        self.search_type = search_type
        self.k = k

    def build(self, vector_store) -> VectorStoreRetriever:
        return vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs={
                "k": self.k
            },
        )