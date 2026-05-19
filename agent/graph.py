"""
Custom StateGraph for the optimization agent.

Replaces ``create_react_agent`` with a hand-wired graph that adds:
  - **Two-layer state**: serialisable OptimizationState (checkpointed by
    LangGraph) + mutable SharedState (torch models, numpy arrays, COM
    connections) accessed by tools via closures.
  - **sync_state bridge**: after each tool execution, scalar metrics are
    copied from SharedState into OptimizationState so that conditional
    edges can make routing decisions from pure state alone.
  - **Deterministic context compression**: when the message list grows
    past a threshold, the middle section is replaced with a metadata
    summary built from SharedState — zero LLM tokens.
  - **Conditional hint injection**: a pre-emptive SystemMessage nudge
    when the agent approaches the 20-CST threshold without a surrogate.

Graph topology::

    START ─> agent ─> should_continue ─┬─> tools ─> sync_state
                                        │               │
                                        │     post_tool_router
                                        │      ╱     │     ╲
                                        │  compress  hint  agent
                                        │     ╲       │     ╱
                                        │      ╲      │    ╱
                                        │       ──> agent <──
                                        │
                                        └─> END
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agent.config import (
    LLM_MODEL, LLM_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    MAX_CST_CALLS, GRAPH_RECURSION_LIMIT, get_deepseek_api_key,
)
from agent.prompts import SYSTEM_PROMPT
from agent.state import OptimizationState
from agent.tools import SharedState, build_tools
from agent.nodes import (
    CompressionStrategy,
    make_agent_node, make_sync_node, make_compress_node, inject_hint,
    should_continue, make_post_tool_router,
)


def build_agent(
    shared: SharedState | None = None,
    max_cst_calls: int = MAX_CST_CALLS,
    api_key: str | None = None,
    compression: CompressionStrategy = CompressionStrategy.DUAL,
    compress_trigger_tokens: int | None = None,
    compress_target_tokens: int | None = None,
):
    """Build the custom StateGraph agent.

    Parameters
    ----------
    shared : SharedState, optional
        Pre-configured shared state.  If None, creates a default one.
    max_cst_calls : int
        CST simulation budget.
    api_key : str, optional
        DeepSeek API key.  If None, reads from ``DEEPSEEK_API_KEY`` env var.

    Returns
    -------
    agent : CompiledGraph
        Compiled LangGraph StateGraph ready for ``.stream()`` / ``.invoke()``.
    shared : SharedState
        Reference to the mutable shared state (for post-run inspection).
    """
    if shared is None:
        shared = SharedState(max_cst_calls=max_cst_calls)

    if api_key is None:
        api_key = get_deepseek_api_key()

    # ── LLM ──────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    # ── Tools ────────────────────────────────────────────────────
    tools = build_tools(shared)

    # ── System prompt template ───────────────────────────────────
    # First pass: fill {max_cst_calls}.
    # Leaves {cst_call_count} for the agent node to fill at runtime.
    system_prompt_template = SYSTEM_PROMPT.format(max_cst_calls=max_cst_calls)

    # ── Assemble StateGraph ──────────────────────────────────────
    graph = StateGraph(OptimizationState)

    # — Nodes —
    graph.add_node("agent", make_agent_node(llm, tools, system_prompt_template))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("sync_state", make_sync_node(shared))
    graph.add_node("compress", make_compress_node(
        shared, llm, compression,
        trigger_tokens=compress_trigger_tokens,
        target_tokens=compress_target_tokens,
    ))
    graph.add_node("inject_hint", inject_hint)

    # — Edges —
    graph.add_edge(START, "agent")

    # After agent: call tools or finish
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "end": END,
    })

    # After tools: always sync state
    graph.add_edge("tools", "sync_state")

    # After sync: route based on state conditions
    post_tool_router = make_post_tool_router(compress_trigger_tokens)
    graph.add_conditional_edges("sync_state", post_tool_router, {
        "agent": "agent",
        "compress": "compress",
        "inject_hint": "inject_hint",
    })

    # After compress / hint: return to agent
    graph.add_edge("compress", "agent")
    graph.add_edge("inject_hint", "agent")

    # ── Compile ──────────────────────────────────────────────────
    checkpointer = MemorySaver()
    agent = graph.compile(checkpointer=checkpointer)

    return agent, shared
