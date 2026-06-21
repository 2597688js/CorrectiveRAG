"""
Author : Janarddan Sarkar
File_name = agent_builder.py
Date : 21-06-2026
Description : Graph builder
"""

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from models.state import State

from graph.nodes import (
    retrieve_node,
    eval_each_doc_node,
    rewrite_query_node,
    web_search_node,
    refine,
    generate,
    route_after_eval,
)


class AgentBuilder:

    @staticmethod
    def build():

        builder = StateGraph(State)

        builder.add_node(
            "retrieve",
            retrieve_node,
        )

        builder.add_node(
            "evaluate",
            eval_each_doc_node,
        )

        builder.add_node(
            "rewrite_query",
            rewrite_query_node,
        )

        builder.add_node(
            "web_search",
            web_search_node,
        )

        builder.add_node(
            "refine",
            refine,
        )

        builder.add_node(
            "generate",
            generate,
        )

        builder.add_edge(
            START,
            "retrieve",
        )

        builder.add_edge(
            "retrieve",
            "evaluate",
        )

        builder.add_conditional_edges(
            "evaluate",
            route_after_eval,
            {
                "refine": "refine",
                "rewrite_query": "rewrite_query",
            },
        )

        builder.add_edge(
            "rewrite_query",
            "web_search",
        )

        builder.add_edge(
            "web_search",
            "refine",
        )

        builder.add_edge(
            "refine",
            "generate",
        )

        builder.add_edge(
            "generate",
            END,
        )

        return builder.compile()