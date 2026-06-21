"""
Shared NVIDIA model configuration.
"""

import os


DEFAULT_LLM_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"


def llm_model() -> str:
    return os.getenv("NVIDIA_LLM_MODEL", DEFAULT_LLM_MODEL)


def embedding_model() -> str:
    return os.getenv("NVIDIA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def nvidia_base_url() -> str | None:
    return os.getenv("NVIDIA_BASE_URL") or None
