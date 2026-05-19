"""
Graph nodes and routing logic for the custom StateGraph.

Architecture:
  START -> agent -> should_continue -> {tools, END}
  tools -> sync_state -> post_tool_router -> {agent, compress, inject_hint}
  compress -> agent
  inject_hint -> agent

Node responsibilities:
  - agent_node:  Call LLM with tool bindings; return AIMessage.
  - sync_state:  Bridge SharedState -> OptimizationState after tool execution.
                 Conditional edges read OptimizationState for routing decisions,
                 so they never touch the non-serialisable SharedState directly.
  - compress:    Dual-track context compression — see docstring below.
  - inject_hint: Inject a SystemMessage nudge when the agent approaches the
                 CST budget without having trained the surrogate.

Routing:
  - should_continue:  agent -> tools (if tool_calls) | END
  - post_tool_router: sync_state -> compress | inject_hint | agent

Context compression design (the interesting part):
  Agent messages carry two fundamentally different kinds of information:

  1. **Experiment facts** — CST results, surrogate accuracy, best candidate.
     These live in SharedState already; a deterministic function can
     reconstruct them without touching messages.  Zero cost.

  2. **Reasoning trace** — *why* the agent chose a region, what patterns it
     observed, what strategies it abandoned.  These only exist in message
     text.  Losing them causes the agent to re-explore dead ends.

  The compress node separates these two concerns:
    Track 1 (deterministic): serialise experiment state from SharedState.
    Track 2 (semantic): one focused LLM call extracts strategic insights
      and negative results into a structured format.

  Compression is triggered by *estimated token count*, not message count,
  so it adapts to conversation density.  The tail window is also
  token-budget-aware: it keeps as many recent messages as fit within the
  target budget, rather than a fixed count.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage, RemoveMessage,
)


class CompressionStrategy(str, Enum):
    """Selectable compression strategies for ablation experiments.

    NONE:       No compression — context grows unbounded until the LLM's
                window is exhausted.  Baseline for measuring information
                loss from compression.
    TRUNCATE:   Keep head + adaptive tail, discard middle without any
                summary.  Cheapest but loses ALL historical context.
    STRUCTURED: Track 1 only — deterministic experiment-state summary
                from SharedState.  Preserves facts, loses reasoning.
    DUAL:       Track 1 + Track 2 — adds one focused LLM call to extract
                strategic reasoning from the removed messages.  Preserves
                both facts and implicit reasoning at the cost of ~1 API call.
    """
    NONE = "none"
    TRUNCATE = "truncate"
    STRUCTURED = "structured"
    DUAL = "dual"

if TYPE_CHECKING:
    from agent.tools import SharedState

logger = logging.getLogger(__name__)


# ── Token budget constants ────────────────────────────────────────────
# DeepSeek-V3 context: 64K tokens.  Rough estimate: 4 chars ≈ 1 token
# for the English-heavy tool output this agent produces.

CHARS_PER_TOKEN_EST = 4
COMPRESS_TRIGGER_TOKENS = 32_000   # trigger compression above this
COMPRESS_TARGET_TOKENS = 16_000    # aim for this much tail context
COMPRESS_KEEP_HEAD = 2             # always keep first N (user kick-off)
COMPRESS_MIN_TAIL = 6              # never drop below this many recent msgs

# LLM extraction limits
EXTRACTION_MAX_CHARS = 8_000       # cap input to extraction LLM call
EXTRACTION_MSG_TRUNCATE = 300      # truncate each message for extraction

# Workflow hint
HINT_CST_THRESHOLD = 18            # inject hint when CST calls >= this


# ── Extraction prompt ─────────────────────────────────────────────────

EXTRACTION_PROMPT = """\
You are analyzing an optical-metasurface optimization agent's conversation \
that is about to be compressed.  Extract the key reasoning that should be \
preserved so the agent does not lose its strategic memory.

Output EXACTLY four numbered sections.  Each bullet should be ONE concise \
sentence.  If a section has no relevant content, write "None observed."

1. PARAMETER INSIGHTS
   Patterns about how pillar geometry (L, W, h) affects optical performance \
(phase difference dphi toward 180 deg, transmittance T_TE/T_TM).

2. PROMISING REGIONS
   Parameter sub-ranges that showed good results, with evidence.

3. ABANDONED STRATEGIES
   Regions or approaches that were tried, found poor, and why.

4. CURRENT STRATEGY
   What the agent was planning to do next, in one sentence.

Conversation to analyze (may be truncated):
"""


# ── Node: agent ───────────────────────────────────────────────────────

def make_agent_node(llm, tools: list, system_prompt_template: str):
    """Create the agent node.

    The system prompt is rebuilt every invocation so that the live
    ``cst_call_count`` is always visible to the LLM.  The prompt is
    prepended at call time and never stored in LangGraph state, keeping
    the message list clean for compression and checkpointing.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state):
        messages = list(state["messages"])
        cst_count = state.get("cst_call_count", 0)

        # Inject live budget counter into the system prompt
        prompt = system_prompt_template.replace(
            "{cst_call_count}", str(cst_count)
        )
        full_messages = [SystemMessage(content=prompt)] + messages

        response = llm_with_tools.invoke(full_messages)
        return {"messages": [response]}

    return agent_node


# ── Node: sync_state ──────────────────────────────────────────────────

def make_sync_node(shared: "SharedState"):
    """Create the sync node (SharedState -> OptimizationState bridge).

    Runs after every ``ToolNode`` execution.  Reads from the mutable
    ``SharedState`` closure and writes scalar metrics into the
    serialisable ``OptimizationState``.  This lets conditional edges
    make routing decisions from pure state, without importing or
    inspecting SharedState.
    """

    def sync_state(state):
        r2 = -1.0
        if shared.surrogate_r2 is not None:
            r2 = shared.surrogate_r2.get("dphi", -1.0)

        best_fom = -999.0
        if shared.best_result is not None:
            best_fom = shared.best_result.get("fom", -999.0)

        # Cast to native Python types — MemorySaver uses msgpack which
        # cannot serialise numpy.float64 / numpy.int64.
        return {
            "cst_call_count": int(shared.cst_call_count),
            "surrogate_trained": bool(shared.surrogate_trained),
            "surrogate_r2_dphi": float(r2),
            "best_fom": float(best_fom),
        }

    return sync_state


# ── Node: compress ────────────────────────────────────────────────────

def make_compress_node(
    shared: "SharedState",
    llm,
    strategy: CompressionStrategy = CompressionStrategy.DUAL,
    trigger_tokens: int | None = None,
    target_tokens: int | None = None,
):
    """Create a configurable context-compression node.

    The ``strategy`` parameter selects what replaces the compressed
    messages — see ``CompressionStrategy`` for the four options.

    Regardless of strategy, the tail window is always **token-budget-
    aware**: it keeps as many recent messages as fit within the target,
    with a floor of ``COMPRESS_MIN_TAIL``.

    Parameters
    ----------
    shared : SharedState
        For Track 1 deterministic summary.
    llm : BaseChatModel
        For Track 2 reasoning extraction (only used by DUAL strategy).
    strategy : CompressionStrategy
        Which replacement summary to generate.
    trigger_tokens : int, optional
        Override ``COMPRESS_TRIGGER_TOKENS`` (useful for evaluation).
    target_tokens : int, optional
        Override ``COMPRESS_TARGET_TOKENS``.
    """
    _trigger = trigger_tokens or COMPRESS_TRIGGER_TOKENS
    _target = target_tokens or COMPRESS_TARGET_TOKENS

    def compress(state):
        messages = state["messages"]
        est_tokens = _estimate_tokens(messages)

        if strategy == CompressionStrategy.NONE:
            return {}

        if est_tokens < _trigger:
            return {}

        # ── Adaptive tail: keep as many recent msgs as fit the budget ──
        keep_tail = _compute_adaptive_tail(messages, _target)

        if keep_tail >= len(messages) - COMPRESS_KEEP_HEAD:
            return {}

        # ── Align to safe boundary (don't break tool-call cycles) ──
        proposed_split = len(messages) - keep_tail
        safe_split = _align_to_safe_boundary(messages, proposed_split)

        to_compress = messages[COMPRESS_KEEP_HEAD:safe_split]
        if not to_compress:
            return {}

        # ── Build replacement summary based on strategy ──
        if strategy == CompressionStrategy.TRUNCATE:
            combined = (
                f"[Context compressed: {len(to_compress)} messages removed. "
                f"No summary available — rely on recent messages only.]"
            )
        elif strategy == CompressionStrategy.STRUCTURED:
            combined = _build_structured_summary(shared)
        elif strategy == CompressionStrategy.DUAL:
            structured = _build_structured_summary(shared)
            semantic = _extract_reasoning_memory(llm, to_compress)
            combined = f"{structured}\n\n{semantic}"
        else:
            combined = ""

        # ── Reconstruct entire message list (safest approach) ──
        # Instead of surgical RemoveMessage (which can leave orphaned
        # tool_calls), we remove ALL messages and rebuild a validated
        # sequence: head + summary + sanitised tail.
        head = list(messages[:COMPRESS_KEEP_HEAD])
        tail = list(messages[safe_split:])
        summary_msg = SystemMessage(content=combined)

        # Sanitize the ENTIRE reconstructed list — not just the tail.
        # The head can also contain an AIMessage with tool_calls whose
        # ToolMessages were in the compressed middle section.
        new_msgs = _sanitize_tool_sequences(head + [summary_msg] + tail)

        remove_all = [
            RemoveMessage(id=m.id)
            for m in messages
            if getattr(m, "id", None)
        ]

        new_est = _estimate_tokens(new_msgs)
        logger.info(
            f"Compression [{strategy.value}]: "
            f"~{est_tokens} -> ~{new_est} est. tokens, "
            f"removed {len(to_compress)} messages from middle, "
            f"kept head={COMPRESS_KEEP_HEAD} tail={len(tail)}"
        )

        return {"messages": remove_all + new_msgs}

    return compress


# ── Node: inject_hint ─────────────────────────────────────────────────

def inject_hint(state):
    """Inject a workflow nudge before the hard CST block kicks in.

    Fires once when ``cst_call_count >= 18`` and the surrogate has not
    been trained.  The cst_tool hard block at 20 is the safety net;
    this hint gives the LLM advance warning to plan ahead.
    """
    cst_count = state.get("cst_call_count", 0)

    hint = (
        f"WORKFLOW ALERT: You have used {cst_count} of your first 20 "
        f"CST calls. After 20 calls, run_cst will be BLOCKED until you "
        f"call train_surrogate. You have enough data to train now. "
        f"Consider calling train_surrogate in your next step to unlock "
        f"more CST budget."
    )

    return {"messages": [SystemMessage(content=hint)]}


# ── Routing: should_continue ──────────────────────────────────────────

def should_continue(state) -> Literal["tools", "end"]:
    """Route after agent_node: execute tool calls or finish."""
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# ── Routing: post_tool_router ─────────────────────────────────────────

def make_post_tool_router(
    trigger_tokens: int | None = None,
):
    """Create the post-tool routing function.

    Wraps the trigger threshold so evaluation code can lower it.
    """
    _trigger = trigger_tokens or COMPRESS_TRIGGER_TOKENS

    def post_tool_router(state) -> Literal["agent", "compress", "inject_hint"]:
        """Route after sync_state based on token budget and workflow state.

        Priority order:
          1. **compress** — token budget exceeded.
          2. **inject_hint** — approaching CST limit without surrogate.
          3. **agent** — default.
        """
        messages = state.get("messages", [])
        est_tokens = _estimate_tokens(messages)
        cst_count = state.get("cst_call_count", 0)
        trained = state.get("surrogate_trained", False)

        if est_tokens >= _trigger:
            return "compress"

        if (cst_count >= HINT_CST_THRESHOLD
                and not trained
                and not _hint_already_injected(messages)):
            return "inject_hint"

        return "agent"

    return post_tool_router


# ══════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════

def _estimate_tokens(messages) -> int:
    """Estimate token count from message character lengths."""
    total_chars = sum(
        len(getattr(m, "content", "") or "") for m in messages
    )
    return total_chars // CHARS_PER_TOKEN_EST


def _sanitize_tool_sequences(messages: list) -> list:
    """Ensure every AIMessage.tool_calls has its ToolMessage responses.

    Two-pass approach:
      Pass 1 — identify complete (AIMessage + all ToolMessages) cycles.
      Pass 2 — keep complete cycles intact; strip tool_calls from
               orphaned AIMessages; drop orphaned ToolMessages.

    This is the last line of defence before messages are sent to the
    LLM API, which hard-rejects malformed sequences.
    """
    # Pass 1: tag messages that belong to complete cycles
    in_complete_cycle: set[int] = set()  # indices
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tc_ids = {tc["id"] for tc in msg.tool_calls}
            n = len(tc_ids)
            following = messages[i + 1 : i + 1 + n]
            if (
                len(following) == n
                and all(isinstance(f, ToolMessage) for f in following)
                and all(f.tool_call_id in tc_ids for f in following)
            ):
                for j in range(i, i + 1 + n):
                    in_complete_cycle.add(j)
                i += 1 + n
                continue
        i += 1

    # Pass 2: build sanitised list
    result = []
    for i, msg in enumerate(messages):
        if i in in_complete_cycle:
            result.append(msg)
        elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            # Orphaned AI with tool_calls → keep text, strip calls
            result.append(AIMessage(
                content=(msg.content or "") + "\n[Prior tool calls compressed]",
                id=msg.id,
            ))
        elif isinstance(msg, ToolMessage):
            # Orphaned ToolMessage → drop silently
            pass
        else:
            result.append(msg)

    return result


def _align_to_safe_boundary(messages, proposed_split: int) -> int:
    """Move the split point so the tail doesn't start mid-tool-call-cycle.

    The DeepSeek (and OpenAI) API requires every ``AIMessage`` with
    ``tool_calls`` to be immediately followed by the corresponding
    ``ToolMessage`` responses.  If compression removes one half of a
    cycle, the API rejects the request.

    This function walks the split point **backward** (keeping more
    messages in the tail) until the first tail message is safe:
      - NOT a ``ToolMessage`` (which needs a preceding AIMessage)
      - If it's an ``AIMessage`` with ``tool_calls``, all the
        responding ``ToolMessage``\\ s must also be in the tail
    """
    idx = proposed_split

    while idx > COMPRESS_KEEP_HEAD:
        msg = messages[idx]

        # Case 1: ToolMessage at boundary — its AIMessage was removed
        if isinstance(msg, ToolMessage):
            idx -= 1
            continue

        # Case 2: AIMessage with tool_calls — check responses are intact
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            n_calls = len(msg.tool_calls)
            following = messages[idx + 1 : idx + 1 + n_calls]
            if len(following) == n_calls and all(
                isinstance(m, ToolMessage) for m in following
            ):
                break  # complete cycle starts here — safe
            idx -= 1
            continue

        # Case 3: HumanMessage / SystemMessage / plain AIMessage — safe
        break

    return idx


def _compute_adaptive_tail(messages, target_tokens: int = COMPRESS_TARGET_TOKENS) -> int:
    """Compute how many tail messages to keep within the token budget.

    Walks backward from the most recent message, accumulating characters
    until the target budget is hit.  Always keeps at least
    ``COMPRESS_MIN_TAIL`` messages.
    """
    keep = 0
    chars = 0
    target_chars = target_tokens * CHARS_PER_TOKEN_EST

    for m in reversed(messages[COMPRESS_KEEP_HEAD:]):
        msg_chars = len(getattr(m, "content", "") or "")
        if chars + msg_chars > target_chars:
            break
        chars += msg_chars
        keep += 1

    return max(keep, COMPRESS_MIN_TAIL)


def _build_structured_summary(shared: "SharedState") -> str:
    """Track 1: deterministic summary from SharedState metadata.

    Perfectly faithful, zero cost.  Covers everything that's captured
    in the structured experiment database.
    """
    lines = [
        "=== STRUCTURED EXPERIMENT STATE ===",
        f"CST calls: {shared.cst_call_count}/{shared.max_cst_calls}",
        f"Database: {len(shared.cst_database)} records",
    ]

    if shared.best_result:
        b = shared.best_result
        lines.append(
            f"Best result: L={b['L']:.0f}, W={b['W']:.0f}, h={b['h']:.0f} | "
            f"dphi={b['dphi_deg']:.1f} deg, T_avg={b['T_avg_pct']:.1f}%, "
            f"FOM={b['fom']:.4f}"
        )

    if shared.surrogate_trained and shared.surrogate_r2:
        r2 = shared.surrogate_r2
        lines.append(
            f"Surrogate: trained | R2 dphi={r2.get('dphi', 0):.3f}, "
            f"T_TE={r2.get('T_TE', 0):.3f}, T_TM={r2.get('T_TM', 0):.3f}"
        )
    else:
        lines.append("Surrogate: NOT yet trained")

    # Top 5 by FOM
    if shared.cst_database:
        ranked = sorted(
            shared.cst_database, key=lambda x: x["fom"], reverse=True
        )
        lines.append("Top 5 CST results:")
        for r in ranked[:5]:
            lines.append(
                f"  L={r['L']:.0f} W={r['W']:.0f} h={r['h']:.0f} | "
                f"dphi={r['dphi_deg']:.1f} T_avg={r['T_avg_pct']:.1f}% "
                f"FOM={r['fom']:.4f}"
            )

        # Coverage stats
        Ls = [r["L"] for r in shared.cst_database]
        Ws = [r["W"] for r in shared.cst_database]
        Hs = [r["h"] for r in shared.cst_database]
        lines.append(
            f"Coverage: L=[{min(Ls):.0f},{max(Ls):.0f}], "
            f"W=[{min(Ws):.0f},{max(Ws):.0f}], "
            f"h=[{min(Hs):.0f},{max(Hs):.0f}]"
        )

    lines.append("=== END STRUCTURED STATE ===")
    return "\n".join(lines)


def _extract_reasoning_memory(llm, messages) -> str:
    """Track 2: LLM-based extraction of strategic reasoning.

    Takes the messages about to be removed and runs one focused LLM call
    to extract parameter insights, promising/abandoned regions, and the
    current strategy.  This preserves the implicit reasoning that only
    exists in the conversation text and cannot be reconstructed from
    the experiment database.

    Input is capped at EXTRACTION_MAX_CHARS and each message is truncated
    to EXTRACTION_MSG_TRUNCATE chars to keep the extraction call cheap.

    Falls back gracefully if the LLM call fails.
    """
    # Pre-process messages into compact text
    msg_lines = []
    total_chars = 0
    for m in messages:
        role = getattr(m, "type", "unknown")
        content = (getattr(m, "content", "") or "")[:EXTRACTION_MSG_TRUNCATE]
        if not content.strip():
            continue
        line = f"[{role}] {content}"
        if total_chars + len(line) > EXTRACTION_MAX_CHARS:
            msg_lines.append("[... earlier messages truncated ...]")
            break
        msg_lines.append(line)
        total_chars += len(line)

    if not msg_lines:
        return (
            "=== STRATEGIC MEMORY ===\n"
            "No reasoning content to extract.\n"
            "=== END STRATEGIC MEMORY ==="
        )

    conversation_text = "\n".join(msg_lines)

    try:
        response = llm.invoke([
            SystemMessage(content=EXTRACTION_PROMPT + conversation_text)
        ])
        extracted = response.content
        return (
            "=== STRATEGIC MEMORY (extracted from compressed messages) ===\n"
            f"{extracted}\n"
            "=== END STRATEGIC MEMORY ==="
        )
    except Exception as e:
        logger.warning(f"Reasoning memory extraction failed: {e}")
        return (
            "=== STRATEGIC MEMORY ===\n"
            "Extraction failed. Rely on structured experiment state above.\n"
            "=== END STRATEGIC MEMORY ==="
        )


def _hint_already_injected(messages) -> bool:
    """Check whether the workflow hint has already been added."""
    for m in messages:
        if isinstance(m, SystemMessage) and "WORKFLOW ALERT" in (m.content or ""):
            return True
    return False
