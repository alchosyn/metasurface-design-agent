"""
Agent state schema and data classes.

Two-layer state design:
  - OptimizationState (LangGraph state): lightweight, serialisable metadata
    that drives graph routing decisions. Managed by the checkpointer.
  - SharedState (tools/__init__.py): heavy/non-serialisable objects (torch
    models, numpy arrays, COM connections). Accessed by tools via closures.

The sync_state node bridges the two: after each tool call, it reads from
SharedState and writes key metrics into OptimizationState so the graph's
conditional edges can make routing decisions without touching SharedState.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import TypedDict, Annotated, Optional

from langgraph.graph.message import add_messages


# ── LangGraph State (serialisable, checkpointable) ──────────────────

class OptimizationState(TypedDict):
    """LangGraph state for the optimization agent.

    Fields beyond 'messages' are synced from SharedState after each
    tool execution. They enable:
      - Conditional routing (e.g., compress when messages > threshold)
      - Workflow enforcement (e.g., inject hint when CST > 18 without surrogate)
      - Convergence detection
    """

    # Standard message history (accumulated by LangGraph's add_messages reducer)
    messages: Annotated[list, add_messages]

    # ── Synced from SharedState (updated by sync_state node) ──
    cst_call_count: int         # how many CST simulations run so far
    surrogate_trained: bool     # whether train_surrogate has been called
    surrogate_r2_dphi: float    # surrogate accuracy for phase (-1 = untrained)
    best_fom: float             # best FOM found so far (-999 = none)


# ── Data Classes ────────────────────────────────────────────────────

@dataclass
class SimResult:
    """One CST or surrogate evaluation."""

    L: float
    W: float
    h: float
    dphi_deg: float
    T_TE_pct: float
    T_TM_pct: float
    source: str            # "cst" | "surrogate"
    fom: float = 0.0

    @property
    def T_avg(self) -> float:
        return (self.T_TE_pct + self.T_TM_pct) / 2.0

    @property
    def phase_err(self) -> float:
        return abs(abs(self.dphi_deg) - 180.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["T_avg_pct"] = self.T_avg
        d["phase_err_deg"] = self.phase_err
        return d

    def summary(self) -> str:
        return (
            f"L={self.L:.0f}, W={self.W:.0f}, h={self.h:.0f}: "
            f"dphi={self.dphi_deg:.1f} deg, "
            f"T_TE={self.T_TE_pct:.1f}%, T_TM={self.T_TM_pct:.1f}%, "
            f"T_avg={self.T_avg:.1f}%, "
            f"phase_err={self.phase_err:.1f} deg, "
            f"FOM={self.fom:.4f} [{self.source}]"
        )
