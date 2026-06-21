"""
Base NEMO workflow abstraction.
Orchestrates tool execution with context threading.
"""

from typing import Any, Dict, Optional, Callable, List
from .context_manager import ContextManager
from .tools_registry import ToolRegistry
import logging

logger = logging.getLogger(__name__)


class NemoWorkflow:
    """
    Base class for NEMO agentic workflows.
    Manages tool execution, context threading, and routing.
    """

    def __init__(self, name: str = "nemo_workflow"):
        self.name = name
        self.context_manager = ContextManager()
        self.tool_registry = ToolRegistry()
        self._routing_rules: Dict[str, Callable] = {}
        self._compiled = False

    def register_tool(
        self,
        name: str,
        description: str,
        func: Callable,
        **kwargs
    ) -> None:
        """Register a tool with the workflow."""
        self.tool_registry.register(
            name=name,
            description=description,
            func=func,
            **kwargs
        )

    def add_routing_rule(
        self,
        condition_name: str,
        rule_func: Callable[[Dict[str, Any]], str]
    ) -> None:
        """
        Add a conditional routing rule.

        Args:
            condition_name: Name of the conditional routing point
            rule_func: Function that takes context dict and returns next tool name
        """
        self._routing_rules[condition_name] = rule_func

    def set_execution_order(self, order: List[str]) -> None:
        """Set the order in which tools should be executed."""
        self.tool_registry.set_execution_order(order)

    def compile(self) -> "NemoWorkflow":
        """Compile the workflow (validate configuration)."""
        if not self.tool_registry.get_execution_order():
            raise ValueError("Execution order not set. Call set_execution_order().")

        if not self.tool_registry.validate_dependencies():
            raise ValueError("Tool dependencies validation failed.")

        self._compiled = True
        logger.info(f"Workflow compiled: {self.name}")
        return self

    async def invoke(
        self,
        initial_state: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the workflow with initial state.

        Args:
            initial_state: Initial workflow state (compatible with LangGraph State)
            **kwargs: Additional arguments

        Returns:
            Final workflow state
        """
        if not self._compiled:
            raise ValueError(
                "Workflow not compiled. Call compile() before invoke()."
            )

        # Initialize context
        self.context_manager.from_dict(initial_state)
        logger.info(f"Starting workflow: {self.name}")

        execution_order = self.tool_registry.get_execution_order()
        current_tool_idx = 0

        while current_tool_idx < len(execution_order):
            tool_name = execution_order[current_tool_idx]
            context_dict = self.context_manager.to_dict()

            # Check if this is a conditional routing point
            if tool_name == "conditional_route":
                next_tool = await self._execute_routing(
                    context_dict, **kwargs
                )
                # Add the routed tool to execution
                if next_tool:
                    # Update execution order dynamically
                    remaining = execution_order[current_tool_idx + 1:]
                    execution_order = execution_order[:current_tool_idx]
                    execution_order.append(next_tool)
                    execution_order.extend(remaining)
                current_tool_idx += 1
                continue

            # Execute tool
            try:
                logger.debug(f"Executing tool: {tool_name}")
                result = await self.tool_registry.invoke(
                    tool_name, context_dict, **kwargs
                )

                # Update context with tool output
                if isinstance(result, dict):
                    self.context_manager.update(**result)
                else:
                    logger.warning(
                        f"Tool {tool_name} returned non-dict result: "
                        f"{type(result).__name__}"
                    )

                logger.debug(f"Tool completed: {tool_name}")

            except Exception as e:
                logger.error(f"Tool execution failed: {tool_name}: {str(e)}")
                raise

            current_tool_idx += 1

        logger.info(f"Workflow completed: {self.name}")
        return self.context_manager.to_dict()

    def invoke_sync(
        self,
        initial_state: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for invoke (for compatibility with LangGraph).
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.invoke(initial_state, **kwargs)
        )

    async def _execute_routing(
        self,
        context_dict: Dict[str, Any],
        **kwargs
    ) -> Optional[str]:
        """Execute conditional routing logic."""
        logger.debug("Executing conditional routing")

        # Default routing based on verdict (CRAG-specific)
        verdict = context_dict.get("verdict", "")

        if verdict == "CORRECT":
            next_tool = "refine"
        elif verdict in ("INCORRECT", "AMBIGUOUS"):
            next_tool = "rewrite_query"
        else:
            logger.warning(f"Unknown verdict for routing: {verdict}")
            next_tool = None

        if next_tool:
            logger.debug(f"Routing to: {next_tool}")

        return next_tool

    def reset(self) -> None:
        """Reset workflow state."""
        self.context_manager.reset()
        logger.debug(f"Workflow state reset: {self.name}")

    def get_context_history(self) -> List[Dict[str, Any]]:
        """Get the history of context updates (for debugging)."""
        return self.context_manager.get_history()
