"""
Author : janarddan
File_name = rewrite_prompt.py
Date : 21/06/26
Description :

"""
from langchain_core.prompts import ChatPromptTemplate


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6-15 words).\n"
            "- If the question implies recency (recent/latest/last week/last month), "
            "add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        (
            "human",
            "Question: {question}",
        ),
    ]
)