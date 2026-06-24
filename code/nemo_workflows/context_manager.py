"""
Context manager for NEMO workflows.
Handles state threading and context management across tool invocations.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from langchain_core.documents import Document


@dataclass
class CRAGContext:
    """State container for Corrective RAG workflow."""

    # Input
    question: str = ""

    # Retrieval
    docs: List[Document] = field(default_factory=list)

    # Evaluation
    good_docs: List[Document] = field(default_factory=list)
    verdict: str = ""
    reason: str = ""

    # Query rewriting
    web_query: str = ""

    # Web search / knowledge supplement
    web_docs: List[Document] = field(default_factory=list)

    # Context refinement
    strips: List[str] = field(default_factory=list)
    kept_strips: List[str] = field(default_factory=list)
    refined_context: str = ""

    # Answer generation
    answer: str = ""

    # Metadata
    _metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Manages context throughout NEMO workflow execution."""

    def __init__(self):
        self.context = CRAGContext()
        self._history: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary (compatible with LangGraph State)."""
        return asdict(self.context)

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load context from dictionary."""
        for key, value in data.items():
            if key != "_metadata" and hasattr(self.context, key):
                setattr(self.context, key, value)

    def update(self, **kwargs) -> None:
        """Update context with new values."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
                self._record_update(key, value)
            else:
                raise ValueError(f"Unknown context field: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context."""
        return getattr(self.context, key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata about execution."""
        self.context._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata."""
        return self.context._metadata.get(key, default)

    def _record_update(self, field: str, value: Any) -> None:
        """Record an update for debugging/logging."""
        self._history.append({
            "field": field,
            "value_type": type(value).__name__,
            "value_preview": (
                str(value)[:50] if not isinstance(value, (list, dict))
                else f"{type(value).__name__}({len(value)})"
            )
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Get update history for debugging."""
        return self._history

    def reset(self) -> None:
        """Reset context to initial state."""
        self.context = CRAGContext()
        self._history = []
