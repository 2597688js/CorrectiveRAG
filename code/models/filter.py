"""
Author : janarddan
File_name = filter.py.py
Date : 21/06/26
Description :

"""

from pydantic import BaseModel

# -----------------------------
# FILTER (LLM judge)
# -----------------------------
class KeepOrDrop(BaseModel):
    keep: bool