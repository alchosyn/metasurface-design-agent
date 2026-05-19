"""
Database query tool — Tier 1 (cheap, <1s).

Lets the agent inspect its accumulated CST simulation history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from agent.tools import SharedState


def make_database_tool(shared: SharedState):

    @tool
    def query_database(
        L_min: float = 60, L_max: float = 340,
        W_min: float = 60, W_max: float = 340,
        h_min: float = 600, h_max: float = 860,
    ) -> str:
        """Query existing CST simulation results within a parameter region.

        Use this to check what data already exists before running CST,
        and to review results in a specific region of parameter space.

        Parameters:
            L_min: Minimum pillar length (nm), default 60
            L_max: Maximum pillar length (nm), default 340
            W_min: Minimum pillar width (nm), default 60
            W_max: Maximum pillar width (nm), default 340
            h_min: Minimum pillar height (nm), default 600
            h_max: Maximum pillar height (nm), default 860

        Returns:
            List of matching CST results with their metrics, plus summary
            statistics. Also shows overall database status and best result.
        """
        matches = [
            r for r in shared.cst_database
            if (L_min <= r["L"] <= L_max
                and W_min <= r["W"] <= W_max
                and h_min <= r["h"] <= h_max)
        ]

        total = len(shared.cst_database)
        lines = [
            f"Database: {total} total CST results, "
            f"{shared.cst_call_count}/{shared.max_cst_calls} budget used."
        ]

        if shared.best_result:
            b = shared.best_result
            lines.append(
                f"Current best: L={b['L']:.0f}, W={b['W']:.0f}, h={b['h']:.0f}, "
                f"dphi={b['dphi_deg']:.1f} deg, T_avg={b['T_avg_pct']:.1f}%, "
                f"FOM={b['fom']:.4f}"
            )

        lines.append(
            f"\nQuery region: L=[{L_min:.0f},{L_max:.0f}], "
            f"W=[{W_min:.0f},{W_max:.0f}], h=[{h_min:.0f},{h_max:.0f}]"
        )
        lines.append(f"Matches: {len(matches)}")

        if matches:
            matches.sort(key=lambda r: r["fom"], reverse=True)
            for r in matches[:15]:
                lines.append(
                    f"  L={r['L']:.0f}, W={r['W']:.0f}, h={r['h']:.0f}: "
                    f"dphi={r['dphi_deg']:.1f} deg, "
                    f"T_avg={r['T_avg_pct']:.1f}%, "
                    f"FOM={r['fom']:.4f}"
                )
            if len(matches) > 15:
                lines.append(f"  ... ({len(matches) - 15} more)")

            # Region statistics
            import numpy as np
            foms = [r["fom"] for r in matches]
            lines.append(
                f"\nRegion stats: FOM min={min(foms):.4f}, "
                f"max={max(foms):.4f}, mean={np.mean(foms):.4f}"
            )
        else:
            lines.append("No data in this region yet.")

        return "\n".join(lines)

    return query_database
