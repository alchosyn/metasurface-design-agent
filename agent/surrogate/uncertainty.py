"""
MC Dropout uncertainty estimation.

With dropout enabled during inference, multiple forward passes give a
distribution of predictions. The variance of this distribution is an
estimate of model uncertainty (epistemic uncertainty).
"""

from __future__ import annotations

import numpy as np
import torch

from agent.config import MC_DROPOUT_SAMPLES
from agent.surrogate.model import ForwardSurrogate, StandardScaler


def enable_mc_dropout(model: ForwardSurrogate):
    """Set dropout layers to training mode (active) while keeping
    everything else in eval mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def estimate_uncertainty(
    model: ForwardSurrogate,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    geometries: np.ndarray,
    n_samples: int = MC_DROPOUT_SAMPLES,
    device: torch.device | None = None,
) -> dict:
    """Estimate prediction uncertainty via MC Dropout.

    Parameters
    ----------
    geometries : (N, 3) array of [L, W, h]
    n_samples : number of stochastic forward passes

    Returns
    -------
    dict with keys:
        mean_dphi, std_dphi : (N,) arrays, degrees
        mean_T_TE, std_T_TE : (N,) arrays, %
        mean_T_TM, std_T_TM : (N,) arrays, %
    """
    device = device or next(model.parameters()).device

    X_norm = x_scaler.transform(geometries)
    X_t = torch.tensor(X_norm, device=device)

    # Collect stochastic predictions
    enable_mc_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            y_norm = model(X_t).cpu().numpy()
            y_raw = y_scaler.inverse(y_norm)
            preds.append(y_raw)
    model.eval()  # restore full eval mode

    preds = np.stack(preds, axis=0)  # (n_samples, N, 4)

    # Convert sin/cos to degrees for each sample
    dphi_samples = np.rad2deg(np.arctan2(preds[:, :, 0], preds[:, :, 1]))
    T_TE_samples = preds[:, :, 2]
    T_TM_samples = preds[:, :, 3]

    return {
        "mean_dphi": dphi_samples.mean(axis=0),
        "std_dphi": dphi_samples.std(axis=0),
        "mean_T_TE": T_TE_samples.mean(axis=0),
        "std_T_TE": T_TE_samples.std(axis=0),
        "mean_T_TM": T_TM_samples.mean(axis=0),
        "std_T_TM": T_TM_samples.std(axis=0),
    }
