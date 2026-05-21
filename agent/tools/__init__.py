"""
Tool registry and shared state for the optimization agent.

SharedState is a mutable singleton that all tools access via closures.
It holds the CST experiment database, surrogate model instance, and
runtime configuration — kept outside of LangGraph's message-based state
to avoid serialising large objects (model weights, numpy arrays).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from agent.config import MAX_CST_CALLS


@dataclass
class SharedState:
    """Mutable shared state accessible by all tools via closure."""

    # ── Serialise CST access (only one simulation at a time) ──
    # LangGraph ToolNode runs parallel tool calls in a thread pool —
    # without this lock concurrent run_cst calls clobber each other's
    # parameters in the single shared CST project.
    cst_lock: Any = field(default_factory=threading.Lock)

    # ── CST experiment database ──
    cst_database: list[dict] = field(default_factory=list)
    cst_call_count: int = 0
    max_cst_calls: int = MAX_CST_CALLS
    best_result: Optional[dict] = None

    # ── CST connection ──
    mock_cst: bool = True
    cst_connection: Any = None           # CSTConnection instance
    cst_project_path: Optional[str] = None

    # ── Ground truth model (for mock CST mode) ──
    ground_truth_model: Any = None       # (model, x_scaler, y_scaler) tuple

    # ── Agent's self-built surrogate ──
    surrogate_model: Any = None          # ForwardSurrogate instance
    surrogate_x_scaler: Any = None       # StandardScaler
    surrogate_y_scaler: Any = None       # StandardScaler
    surrogate_trained: bool = False
    surrogate_r2: Optional[dict] = None

    # ── Surrogate training history (for evaluation) ──
    # Each entry: {"n_points": int, "r2_dphi": float, "r2_T_TE": float,
    #              "r2_T_TM": float, "cst_call_count": int}
    surrogate_train_history: list[dict] = field(default_factory=list)


def build_tools(shared: SharedState) -> list:
    """Construct all tools with access to the shared state."""

    from agent.tools.cst_tool import make_cst_tool
    from agent.tools.surrogate_tool import (
        make_train_tool, make_predict_tool, make_uncertainty_tool,
    )
    from agent.tools.fresnel_tool import make_fresnel_tool
    from agent.tools.fom_tool import make_fom_tool
    from agent.tools.database_tool import make_database_tool

    return [
        make_cst_tool(shared),           # Tier 3
        make_train_tool(shared),         # Tier 2
        make_fresnel_tool(),             # Tier 2
        make_predict_tool(shared),       # Tier 1
        make_uncertainty_tool(shared),   # Tier 1
        make_fom_tool(shared),           # Tier 1
        make_database_tool(shared),      # Tier 1
    ]
