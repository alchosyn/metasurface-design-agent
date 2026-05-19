"""
CST simulation tool — Tier 3 (expensive, ~1 min per call).

Wraps the CST COM interface for use as a LangChain tool.
Supports mock mode for development/testing without CST.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from agent.config import L_MIN, L_MAX, W_MIN, W_MAX, H_MIN, H_MAX, FOM_PHASE_WEIGHT, FOM_TRANS_WEIGHT
from agent.state import SimResult

if TYPE_CHECKING:
    from agent.tools import SharedState

logger = logging.getLogger(__name__)


def make_cst_tool(shared: SharedState):
    """Create the run_cst tool with access to shared state."""

    @tool
    def run_cst(L: float, W: float, h: float) -> str:
        """Run a full-wave electromagnetic simulation in CST Studio Suite.

        Simulates a single TiO2 pillar on SiO2 substrate at 632 nm and
        returns its TE/TM transmittance and phase difference.

        Parameters:
            L: Pillar length in nm (60-340)
            W: Pillar width in nm (60-340)
            h: Pillar height in nm (600-860)

        Returns:
            Text summary of simulation results including phase difference,
            transmittance, and figure of merit.

        Note: Each call takes 30s-2min. Use predict_surrogate for fast
        exploration and reserve run_cst for validation. Budget is limited.
        """
        # ── Validate inputs ──
        errors = []
        if not (L_MIN <= L <= L_MAX):
            errors.append(f"L={L} out of range [{L_MIN}, {L_MAX}]")
        if not (W_MIN <= W <= W_MAX):
            errors.append(f"W={W} out of range [{W_MIN}, {W_MAX}]")
        if not (H_MIN <= h <= H_MAX):
            errors.append(f"h={h} out of range [{H_MIN}, {H_MAX}]")
        if L < W:
            errors.append(f"L={L} < W={W}: L must be >= W (PB convention)")
        if errors:
            return "ERROR: " + "; ".join(errors)

        # ── Check budget ──
        if shared.cst_call_count >= shared.max_cst_calls:
            return (
                f"ERROR: CST budget exhausted ({shared.cst_call_count}/"
                f"{shared.max_cst_calls}). Use predict_surrogate instead, "
                f"or declare convergence."
            )

        # ── Enforce surrogate training ──
        if shared.cst_call_count >= 20 and not shared.surrogate_trained:
            return (
                f"BLOCKED: You have used {shared.cst_call_count} CST calls "
                f"without training the surrogate model. You MUST call "
                f"train_surrogate NOW before running any more CST simulations. "
                f"You have enough data ({len(shared.cst_database)} points)."
            )

        # ── Check duplicates ──
        for r in shared.cst_database:
            if (abs(r["L"] - L) < 0.5 and abs(r["W"] - W) < 0.5
                    and abs(r["h"] - h) < 0.5):
                return (
                    f"DUPLICATE: Already simulated L={r['L']:.0f}, "
                    f"W={r['W']:.0f}, h={r['h']:.0f}. "
                    f"Result: dphi={r['dphi_deg']:.1f} deg, "
                    f"T_avg={r['T_avg_pct']:.1f}%, FOM={r['fom']:.4f}. "
                    f"Use query_database to check existing data."
                )

        # ── Run simulation (real or mock) ──
        if shared.mock_cst:
            result = _mock_simulate(L, W, h, shared)
        else:
            result = _real_simulate(L, W, h, shared)

        # ── Compute FOM ──
        phase_err = abs(abs(result["dphi_deg"]) - 180.0)
        T_avg = (result["T_TE_pct"] + result["T_TM_pct"]) / 2.0
        fom = (-FOM_PHASE_WEIGHT * phase_err + FOM_TRANS_WEIGHT * T_avg / 100.0)

        record = {
            "L": L, "W": W, "h": h,
            "dphi_deg": result["dphi_deg"],
            "T_TE_pct": result["T_TE_pct"],
            "T_TM_pct": result["T_TM_pct"],
            "T_avg_pct": T_avg,
            "phase_err_deg": phase_err,
            "fom": fom,
        }
        shared.cst_database.append(record)
        shared.cst_call_count += 1

        # Update best
        if shared.best_result is None or fom > shared.best_result["fom"]:
            shared.best_result = record.copy()

        # ── Format response ──
        budget_msg = f"[CST {shared.cst_call_count}/{shared.max_cst_calls}]"
        return (
            f"{budget_msg} L={L:.0f}, W={W:.0f}, h={h:.0f}: "
            f"dphi={result['dphi_deg']:.1f} deg, "
            f"T_TE={result['T_TE_pct']:.1f}%, "
            f"T_TM={result['T_TM_pct']:.1f}%, "
            f"T_avg={T_avg:.1f}%, "
            f"phase_err={phase_err:.1f} deg, "
            f"FOM={fom:.4f}"
        )

    return run_cst


def _real_simulate(L: float, W: float, h: float, shared: SharedState) -> dict:
    """Run actual CST simulation via COM."""
    from agent.cst.geometry import set_pillar_geometry
    from agent.cst.solver import run_frequency_solver
    from agent.cst.results import extract_s_parameters, convert_s_params

    if shared.cst_connection is None:
        raise RuntimeError("CST connection not initialised. Set cst_project_path.")

    project = shared.cst_connection.project
    set_pillar_geometry(project, L, W, h)
    run_frequency_solver(project)
    sparam = extract_s_parameters(project)
    return convert_s_params(sparam)


def _mock_simulate(L: float, W: float, h: float, shared: SharedState) -> dict:
    """Mock CST using the pre-trained surrogate from ml/ as ground truth.

    Adds small Gaussian noise to simulate CST variability.
    Falls back to a physics-inspired model if no checkpoint is available.
    """
    if shared.ground_truth_model is not None:
        model, x_scaler, y_scaler = shared.ground_truth_model
        device = next(model.parameters()).device
        g = np.array([[L, W, h]], dtype=np.float32)
        gn = (g - x_scaler.mean) / x_scaler.std
        import torch
        with torch.no_grad():
            y = model(torch.tensor(gn, device=device))
            y = y.cpu().numpy()[0] * y_scaler.std + y_scaler.mean
        dphi = float(np.rad2deg(np.arctan2(y[0], y[1])))
        T_TE = float(np.clip(y[2], 0, 100))
        T_TM = float(np.clip(y[3], 0, 100))
    else:
        # Simple physics-inspired model as fallback
        aspect = L / max(W, 1)
        dphi = -180.0 * np.tanh((aspect - 1.5) * 0.8) * (h / 700) ** 0.5
        T_TE = 95.0 - 0.01 * (L - 200) ** 2 / 100
        T_TM = 90.0 - 0.02 * (W - 100) ** 2 / 100

    # Add noise
    dphi += np.random.normal(0, 1.5)
    T_TE = np.clip(T_TE + np.random.normal(0, 0.8), 0, 100)
    T_TM = np.clip(T_TM + np.random.normal(0, 0.8), 0, 100)

    return {"dphi_deg": dphi, "T_TE_pct": T_TE, "T_TM_pct": T_TM}
