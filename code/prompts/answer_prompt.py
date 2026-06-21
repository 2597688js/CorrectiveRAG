"""
Author : janarddan
File_name = answer_prompt.py
Date : 21/06/26
Description :

"""
from langchain_core.prompts import ChatPromptTemplate


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI Assistant.\n"
            "Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, say: "
            "'I don't know.'",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}",
        ),
    ]
)