"""
Author : janarddan
File_name = document_loader.py
Date : 21/06/26
Description :

"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

class DocumentLoader:
    def __init__(self, document_paths: List[str]):
        self.document_paths = document_paths

    def load(self) -> List[Document]:
        docs = []

        for path in self.document_paths:
            docs.extend(PyPDFLoader(path).load())

        return docs


if __name__ == "__main__":
    document_loader = DocumentLoader(
        [
            "/Users/janarddan/jana files/1.All Notes/11.GenAi_note_pdfs/Build_a_Multimodal_AI_Agent_in_30_Minutes.pdf",
            "/Users/janarddan/jana files/1.All Notes/11.GenAi_note_pdfs/Fine-tune_Your_First_LLM_with_LoRA_in_30_Minutes.pdf"
        ]
    )

    docs = document_loader.load()
    for id, doc in enumerate(docs):
        print(f"{id}. {doc}")
        print()
        print()
        print()