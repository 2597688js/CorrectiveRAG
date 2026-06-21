"""
Author : janarddan
File_name = doc_eval.py.py
Date : 21/06/26
Description :

"""
from pydantic import BaseModel

# -----------------------------
# Score-based doc evaluator
# -----------------------------
class DocEvalScore(BaseModel):
    score: float
    reason: str