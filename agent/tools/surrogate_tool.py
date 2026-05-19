"""
Surrogate model tools -Tier 1 (predict, uncertainty) and Tier 2 (train).

The agent builds its own surrogate model from CST data it collects.
These tools operate on the shared state's surrogate model instance.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.tools import tool

from agent.config import (
    SURROGATE_MIN_POINTS, L_MIN, L_MAX, W_MIN, W_MAX, H_MIN, H_MAX,
    FOM_PHASE_WEIGHT, FOM_TRANS_WEIGHT,
)

if TYPE_CHECKING:
    from agent.tools import SharedState

logger = logging.getLogger(__name__)


def make_train_tool(shared: SharedState):

    @tool
    def train_surrogate() -> str:
        """Train or retrain the surrogate neural network using accumulated CST data.

        Uses all CST simulation results collected so far to train a lightweight
        MLP model. After training, you can use predict_surrogate for fast
        predictions and estimate_uncertainty for confidence assessment.

        Requires at least 20 CST data points. Training takes a few seconds.

        Returns:
            Training summary including number of data points used and
            R-squared accuracy metrics for phase and transmittance.
        """
        n = len(shared.cst_database)
        if n < SURROGATE_MIN_POINTS:
            return (
                f"ERROR: Need at least {SURROGATE_MIN_POINTS} CST data points "
                f"to train. Currently have {n}. Run more CST simulations first."
            )

        from agent.surrogate.trainer import train_surrogate as _train

        model, x_scaler, y_scaler, r2 = _train(shared.cst_database)
        shared.surrogate_model = model
        shared.surrogate_x_scaler = x_scaler
        shared.surrogate_y_scaler = y_scaler
        shared.surrogate_trained = True
        shared.surrogate_r2 = r2

        return (
            f"Surrogate trained on {n} CST data points. "
            f"Validation R-squared: "
            f"dphi={r2['dphi']:.3f}, "
            f"T_TE={r2['T_TE']:.3f}, "
            f"T_TM={r2['T_TM']:.3f}. "
            f"You can now use predict_surrogate and estimate_uncertainty."
        )

    return train_surrogate


def make_predict_tool(shared: SharedState):

    @tool
    def predict_surrogate(geometries: str) -> str:
        """Predict optical properties for multiple pillar geometries using the surrogate model.

        Parameters:
            geometries: JSON array of [L, W, h] triplets, e.g.
                "[[310,115,600],[300,120,650],[280,100,700]]"
                Maximum 500 geometries per call.

        Returns:
            Predictions sorted by FOM (best first), showing predicted
            phase difference, transmittance, and figure of merit.

        Note: Must call train_surrogate first. Predictions may be inaccurate
        in regions with sparse CST data -use estimate_uncertainty to check.
        """
        if not shared.surrogate_trained:
            return (
                "ERROR: Surrogate model not trained yet. "
                "Call train_surrogate first (need >= 20 CST data points)."
            )

        try:
            geom_list = json.loads(geometries)
        except json.JSONDecodeError:
            return "ERROR: Invalid JSON. Use format: [[L1,W1,h1],[L2,W2,h2],...]"

        if len(geom_list) > 500:
            return "ERROR: Maximum 500 geometries per call."
        if not geom_list:
            return "ERROR: Empty geometry list."

        import torch

        model = shared.surrogate_model
        x_scaler = shared.surrogate_x_scaler
        y_scaler = shared.surrogate_y_scaler
        device = next(model.parameters()).device

        # Filter valid geometries
        valid = []
        for g in geom_list:
            if len(g) != 3:
                continue
            L, W, h = g
            if (L_MIN <= L <= L_MAX and W_MIN <= W <= W_MAX
                    and H_MIN <= h <= H_MAX and L >= W):
                valid.append(g)

        if not valid:
            return "ERROR: No valid geometries (check ranges and L >= W)."

        X = np.array(valid, dtype=np.float32)
        X_norm = x_scaler.transform(X)

        model.eval()
        with torch.no_grad():
            Y_norm = model(torch.tensor(X_norm, device=device)).cpu().numpy()
        Y = y_scaler.inverse(Y_norm)

        # Format results
        results = []
        for i, (L, W, h) in enumerate(valid):
            dphi = float(np.rad2deg(np.arctan2(Y[i, 0], Y[i, 1])))
            T_TE = float(np.clip(Y[i, 2], 0, 100))
            T_TM = float(np.clip(Y[i, 3], 0, 100))
            phase_err = abs(abs(dphi) - 180.0)
            T_avg = (T_TE + T_TM) / 2.0
            fom = -FOM_PHASE_WEIGHT * phase_err + FOM_TRANS_WEIGHT * T_avg / 100.0
            results.append({
                "L": L, "W": W, "h": h,
                "dphi": dphi, "T_TE": T_TE, "T_TM": T_TM,
                "T_avg": T_avg, "phase_err": phase_err, "fom": fom,
            })

        results.sort(key=lambda r: r["fom"], reverse=True)

        lines = [f"Surrogate predictions ({len(results)} valid geometries, sorted by FOM):"]
        for r in results[:20]:  # show top 20
            lines.append(
                f"  L={r['L']:.0f}, W={r['W']:.0f}, h={r['h']:.0f}: "
                f"dphi={r['dphi']:.1f} deg, T_avg={r['T_avg']:.1f}%, "
                f"phase_err={r['phase_err']:.1f} deg, FOM={r['fom']:.4f}"
            )
        if len(results) > 20:
            lines.append(f"  ... ({len(results) - 20} more omitted)")

        return "\n".join(lines)

    return predict_surrogate


def make_uncertainty_tool(shared: SharedState):

    @tool
    def estimate_uncertainty(geometries: str) -> str:
        """Estimate prediction uncertainty at specified geometries using MC Dropout.

        Runs 50 stochastic forward passes through the surrogate model to
        estimate the variance of predictions. High uncertainty means the
        model has insufficient training data in that region.

        Parameters:
            geometries: JSON array of [L, W, h] triplets, e.g.
                "[[310,115,600],[280,130,700]]"
                Maximum 100 geometries per call.

        Returns:
            Uncertainty estimates for each geometry, with recommendations
            on whether CST validation is needed.

        Note: Must call train_surrogate first.
        """
        if not shared.surrogate_trained:
            return (
                "ERROR: Surrogate model not trained yet. "
                "Call train_surrogate first."
            )

        try:
            geom_list = json.loads(geometries)
        except json.JSONDecodeError:
            return "ERROR: Invalid JSON."

        if len(geom_list) > 100:
            return "ERROR: Maximum 100 geometries per call."

        from agent.surrogate.uncertainty import estimate_uncertainty as _estimate

        X = np.array(geom_list, dtype=np.float32)
        unc = _estimate(
            shared.surrogate_model,
            shared.surrogate_x_scaler,
            shared.surrogate_y_scaler,
            X,
        )

        lines = ["Uncertainty estimates (MC Dropout, 50 samples):"]
        for i, (L, W, h) in enumerate(geom_list):
            dphi_std = unc["std_dphi"][i]
            T_TE_std = unc["std_T_TE"][i]
            T_TM_std = unc["std_T_TM"][i]

            # Classification
            if dphi_std > 10:
                level = "HIGH"
                advice = "CST validation strongly recommended"
            elif dphi_std > 5:
                level = "MEDIUM"
                advice = "consider CST validation"
            else:
                level = "LOW"
                advice = "surrogate prediction is reliable"

            lines.append(
                f"  L={L:.0f}, W={W:.0f}, h={h:.0f}: "
                f"dphi_std=+/-{dphi_std:.1f} deg ({level}), "
                f"T_TE_std=+/-{T_TE_std:.1f}%, "
                f"T_TM_std=+/-{T_TM_std:.1f}% -{advice}"
            )

        return "\n".join(lines)

    return estimate_uncertainty
