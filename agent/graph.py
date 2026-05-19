"""
LangGraph agent assembly.

Uses create_react_agent — no hardcoded flow control.
The LLM decides which tool to call at each step.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from agent.config import (
    LLM_MODEL, LLM_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    MAX_CST_CALLS, get_deepseek_api_key,
)
from agent.prompts import SYSTEM_PROMPT
from agent.tools import SharedState, build_tools


def build_agent(
    shared: SharedState | None = None,
    max_cst_calls: int = MAX_CST_CALLS,
    api_key: str | None = None,
):
    """Build the LangGraph ReAct agent.

    Parameters
    ----------
    shared : SharedState, optional
        Pre-configured shared state. If None, creates a default one.
    max_cst_calls : int
        CST simulation budget.
    api_key : str, optional
        DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var.

    Returns
    -------
    agent : compiled LangGraph agent
    shared : SharedState reference (for inspection after runs)
    """
    if shared is None:
        shared = SharedState(max_cst_calls=max_cst_calls)

    if api_key is None:
        api_key = get_deepseek_api_key()

    # LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    # Tools
    tools = build_tools(shared)

    # Format system prompt with budget
    system_prompt = SYSTEM_PROMPT.format(max_cst_calls=max_cst_calls)

    # Build agent
    # No custom state_schema — we use SharedState (mutable closure) for
    # experiment data, not LangGraph's message-based state.
    checkpointer = MemorySaver()
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        prompt=system_prompt,
    )

    return agent, shared
