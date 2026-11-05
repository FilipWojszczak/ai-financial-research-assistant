"""
START → prepare → agent ─tool calls→ tools → grade ─insufficient, round < max→ agent
                         └─no calls→ generate ←─sufficient or round cap─┘
"""

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from .context import ChatContext
from .models import ChatModels
from .nodes import ChatNodes, prepare, route_after_agent, route_after_grade
from .state import ChatInput, ChatState
from .tools import CHAT_TOOLS

type ChatGraph = CompiledStateGraph[ChatState, ChatContext, ChatInput, ChatState]


def build_chat_graph(
    models: ChatModels,
    *,
    checkpointer: BaseCheckpointSaver,
    tools: Sequence[BaseTool] = CHAT_TOOLS,
) -> ChatGraph:
    nodes = ChatNodes(models, tools)
    builder = StateGraph(ChatState, context_schema=ChatContext, input_schema=ChatInput)
    builder.add_node("prepare", prepare)
    builder.add_node("agent", nodes.agent)
    builder.add_node("tools", ToolNode(tools, messages_key="scratchpad"))
    builder.add_node("grade", nodes.grade)
    builder.add_node("generate", nodes.generate)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "generate"])
    builder.add_edge("tools", "grade")
    builder.add_conditional_edges("grade", route_after_grade, ["agent", "generate"])
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=checkpointer)
