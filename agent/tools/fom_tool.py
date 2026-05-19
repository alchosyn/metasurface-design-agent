"""
Figure of Merit computation tool — Tier 1 (cheap, <1s).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from agent.config import FOM_PHASE_WEIGHT, FOM_TRANS_WEIGHT

if TYPE_CHECKING:
    from agent.tools import SharedState


def make_fom_tool(shared: SharedState):

    @tool
    def compute_fom(results: str) -> str:
        """Compute Figure of Merit and rank a set of simulation results.

        FOM = -10 * |abs(dphi) - 180| + (T_TE + T_TM) / 200
        Higher is better. Perfect HWP with 100% transmission gives FOM ~ 1.0.

        Parameters:
            results: JSON array of objects with keys:
                dphi_deg, T_TE_pct, T_TM_pct, and optionally L, W, h.
                Example: '[{"dphi_deg": -179.6, "T_TE_pct": 95, "T_TM_pct": 98}]'

        Returns:
            Results sorted by FOM (best first) with computed metrics.
        """
        try:
            data = json.loads(results)
        except json.JSONDecodeError:
            return "ERROR: Invalid JSON."

        if not data:
            return "ERROR: Empty results list."

        scored = []
        for r in data:
            dphi = r.get("dphi_deg", 0)
            T_TE = r.get("T_TE_pct", 0)
            T_TM = r.get("T_TM_pct", 0)
            phase_err = abs(abs(dphi) - 180.0)
            T_avg = (T_TE + T_TM) / 2.0
            fom = -FOM_PHASE_WEIGHT * phase_err + FOM_TRANS_WEIGHT * T_avg / 100.0
            scored.append({**r, "phase_err": phase_err, "T_avg": T_avg, "fom": fom})

        scored.sort(key=lambda x: x["fom"], reverse=True)

        lines = [f"FOM ranking ({len(scored)} results):"]
        for i, s in enumerate(scored[:15]):
            geo = ""
            if "L" in s:
                geo = f"L={s['L']:.0f}, W={s['W']:.0f}, h={s['h']:.0f}: "
            lines.append(
                f"  {i+1}. {geo}"
                f"dphi={s['dphi_deg']:.1f} deg, "
                f"T_avg={s['T_avg']:.1f}%, "
                f"phase_err={s['phase_err']:.1f} deg, "
                f"FOM={s['fom']:.4f}"
            )

        return "\n".join(lines)

    return compute_fom
