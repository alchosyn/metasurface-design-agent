"""
Agent state schema and data classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import TypedDict, Annotated, Optional

from langgraph.graph.message import add_messages


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


class AgentState(TypedDict):
    """LangGraph state for the optimization agent."""

    # Standard message history (accumulated by LangGraph)
    messages: Annotated[list, add_messages]

    # ── Experiment database (managed by tools) ──
    # These are NOT in the message flow; tools read/write them via closures.
    # Kept as module-level singletons (see tools/__init__.py).
