"""
Author : janarddan
File_name = filter_prompt.py
Date : 21/06/26
Description :

"""
from langchain_core.prompts import ChatPromptTemplate


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence.\n"
            "Output JSON only.",
        ),
        (
            "human",
            "Question: {question}\n\nSentence:\n{sentence}",
        ),
    ]
)