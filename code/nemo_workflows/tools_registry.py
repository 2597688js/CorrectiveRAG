"""
Tool registry for NEMO workflows.
Manages registration, lookup, and invocation of workflow tools.
"""

from typing import Callable, Dict, Any, Optional, List, Type
from dataclasses import dataclass
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a workflow tool."""

    name: str
    description: str
    input_schema: Optional[Type[BaseModel]] = None
    output_schema: Optional[Type[BaseModel]] = None
    func: Optional[Callable] = None
    requires: List[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.requires is None:
            self.requires = []
        if self.tags is None:
            self.tags = []


class ToolRegistry:
    """Registry for workflow tools."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._execution_order: List[str] = []

    def register(
        self,
        name: str,
        description: str,
        func: Callable,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        requires: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> ToolDefinition:
        """Register a new tool."""
        if name in self._tools:
            logger.warning(f"Overwriting existing tool: {name}")

        definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            func=func,
            requires=requires or [],
            tags=tags or [],
        )

        self._tools[name] = definition
        logger.debug(f"Registered tool: {name}")
        return definition

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """Get all registered tools."""
        return list(self._tools.values())

    def set_execution_order(self, order: List[str]) -> None:
        """Set the execution order of tools."""
        for tool_name in order:
            if tool_name not in self._tools and tool_name != "conditional_route":
                logger.warning(f"Tool not registered: {tool_name}")
        self._execution_order = order

    def get_execution_order(self) -> List[str]:
        """Get the execution order."""
        return self._execution_order

    async def invoke(
        self,
        tool_name: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Invoke a tool with context.

        Args:
            tool_name: Name of the tool to invoke
            context: Current workflow context
            **kwargs: Additional arguments to pass to tool

        Returns:
            Tool output
        """
        tool_def = self.get(tool_name)
        if not tool_def:
            raise ValueError(f"Tool not found: {tool_name}")

        if not tool_def.func:
            raise ValueError(f"Tool has no implementation: {tool_name}")

        logger.debug(f"Invoking tool: {tool_name}")

        # Check if tool function is async
        import inspect
        if inspect.iscoroutinefunction(tool_def.func):
            result = await tool_def.func(context, **kwargs)
        else:
            result = tool_def.func(context, **kwargs)

        logger.debug(f"Tool completed: {tool_name}")
        return result

    def validate_dependencies(self) -> bool:
        """Validate that all tool dependencies are registered."""
        for tool in self._tools.values():
            for dep in tool.requires:
                if dep not in self._tools:
                    logger.error(
                        f"Tool {tool.name} requires unregistered tool: {dep}"
                    )
                    return False
        return True

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._execution_order.clear()
