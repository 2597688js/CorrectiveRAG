"""
Author : janarddan
File_name = doc_eval_prompt.py
Date : 21/06/26
Description :

"""
from langchain_core.prompts import ChatPromptTemplate


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        (
            "human",
            "Question: {question}\n\nChunk:\n{chunk}",
        ),
    ]
)