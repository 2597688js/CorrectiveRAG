"""
Author : janarddan
File_name = web_query.py.py
Date : 21/06/26
Description :

"""
from pydantic import BaseModel

# -----------------------------
# Query rewrite for web search
# -----------------------------
class WebQuery(BaseModel):
    query: str