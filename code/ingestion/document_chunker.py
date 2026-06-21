"""
Author : janarddan
File_name = document_chunker.py
Date : 21/06/26
Description :

"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        clean_text: bool = True,
    ):
        self.clean_text = clean_text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, docs: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(docs)

        if self.clean_text:
            for chunk in chunks:
                chunk.page_content = (
                    chunk.page_content
                    .encode("utf-8", "ignore")
                    .decode("utf-8", "ignore")
                )

        return chunks
