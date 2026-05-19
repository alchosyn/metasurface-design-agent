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
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import AIMessage, SystemMessage, RemoveMessage

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

        return {
            "cst_call_count": shared.cst_call_count,
            "surrogate_trained": shared.surrogate_trained,
            "surrogate_r2_dphi": r2,
            "best_fom": best_fom,
        }

    return sync_state


# ── Node: compress (dual-track) ──────────────────────────────────────

def make_compress_node(shared: "SharedState", llm):
    """Create the dual-track context-compression node.

    Two independent tracks produce the replacement summary:

    **Track 1 — Deterministic (structured facts)**
      Built from SharedState: database stats, top-5 results, surrogate
      accuracy, best candidate.  Zero LLM cost, perfectly faithful.

    **Track 2 — Semantic (reasoning memory)**
      One focused LLM call extracts parameter insights, promising regions,
      abandoned strategies, and current plan from the messages about to be
      removed.  This preserves the *reasoning* that only exists in text.

    The tail window is token-budget-aware: it walks backward from the most
    recent message, accumulating characters until hitting the target budget,
    then keeps at least ``COMPRESS_MIN_TAIL`` messages.

    Falls back to deterministic-only if the LLM extraction call fails.
    """

    def compress(state):
        messages = state["messages"]
        est_tokens = _estimate_tokens(messages)

        if est_tokens < COMPRESS_TRIGGER_TOKENS:  # router already checked, but safety
            return {}  # safety no-op

        # ── Adaptive tail: keep as many recent msgs as fit the budget ──
        keep_tail = _compute_adaptive_tail(messages)

        if keep_tail >= len(messages) - COMPRESS_KEEP_HEAD:
            return {}  # nothing to compress

        to_compress = (
            messages[COMPRESS_KEEP_HEAD:-keep_tail]
            if keep_tail > 0
            else messages[COMPRESS_KEEP_HEAD:]
        )
        if not to_compress:
            return {}

        # ── Track 1: deterministic structured summary ──
        structured = _build_structured_summary(shared)

        # ── Track 2: LLM-based reasoning memory extraction ──
        semantic = _extract_reasoning_memory(llm, to_compress)

        combined = f"{structured}\n\n{semantic}"

        # Remove compressed messages, inject combined summary
        removals = [
            RemoveMessage(id=m.id)
            for m in to_compress
            if getattr(m, "id", None)
        ]
        summary_msg = SystemMessage(content=combined)

        new_est = _estimate_tokens(
            list(messages[:COMPRESS_KEEP_HEAD])
            + [summary_msg]
            + list(messages[-keep_tail:] if keep_tail > 0 else [])
        )
        logger.info(
            f"Context compression: ~{est_tokens} -> ~{new_est} est. tokens, "
            f"removed {len(removals)} messages, "
            f"kept head={COMPRESS_KEEP_HEAD} tail={keep_tail}"
        )

        return {"messages": removals + [summary_msg]}

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

def post_tool_router(state) -> Literal["agent", "compress", "inject_hint"]:
    """Route after sync_state based on token budget and workflow state.

    Evaluated purely from ``OptimizationState`` fields (+ messages for
    token estimation).  Priority order:

      1. **compress** — token budget exceeded.
      2. **inject_hint** — approaching CST limit without surrogate.
      3. **agent** — default.
    """
    messages = state.get("messages", [])
    est_tokens = _estimate_tokens(messages)
    cst_count = state.get("cst_call_count", 0)
    trained = state.get("surrogate_trained", False)

    # Priority 1: context compression (token-budget-driven)
    if est_tokens >= COMPRESS_TRIGGER_TOKENS:
        return "compress"

    # Priority 2: pre-emptive workflow hint (fire only once)
    if (cst_count >= HINT_CST_THRESHOLD
            and not trained
            and not _hint_already_injected(messages)):
        return "inject_hint"

    # Default
    return "agent"


# ══════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════

def _estimate_tokens(messages) -> int:
    """Estimate token count from message character lengths."""
    total_chars = sum(
        len(getattr(m, "content", "") or "") for m in messages
    )
    return total_chars // CHARS_PER_TOKEN_EST


def _compute_adaptive_tail(messages) -> int:
    """Compute how many tail messages to keep within the token budget.

    Walks backward from the most recent message, accumulating characters
    until the target budget is hit.  Always keeps at least
    ``COMPRESS_MIN_TAIL`` messages.
    """
    keep = 0
    chars = 0
    target_chars = COMPRESS_TARGET_TOKENS * CHARS_PER_TOKEN_EST

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
