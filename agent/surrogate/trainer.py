"""
Online training / retraining logic for the agent's surrogate model.

The agent builds its own surrogate from scratch using CST data it collects.
Training is fast (~seconds for <200 points) and can be called multiple times
as new data arrives.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from agent.config import (
    SURROGATE_HIDDEN, SURROGATE_N_LAYERS, SURROGATE_DROPOUT,
    SURROGATE_TRAIN_EPOCHS, SURROGATE_TRAIN_LR, SURROGATE_TRAIN_BATCH,
)
from agent.surrogate.model import ForwardSurrogate, StandardScaler


def prepare_training_data(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert CST experiment records to (X, Y) arrays.

    Parameters
    ----------
    records : list of dicts with keys L, W, h, dphi_deg, T_TE_pct, T_TM_pct

    Returns
    -------
    X : (N, 3) float32 — [L, W, h]
    Y : (N, 4) float32 — [sin(dphi), cos(dphi), T_TE, T_TM]
    """
    X = np.array([[r["L"], r["W"], r["h"]] for r in records], dtype=np.float32)
    dphi_rad = np.deg2rad([r["dphi_deg"] for r in records]).astype(np.float32)
    sin_phi = np.sin(dphi_rad)
    cos_phi = np.cos(dphi_rad)
    T_TE = np.array([r["T_TE_pct"] for r in records], dtype=np.float32)
    T_TM = np.array([r["T_TM_pct"] for r in records], dtype=np.float32)
    Y = np.column_stack([sin_phi, cos_phi, T_TE, T_TM])
    return X, Y


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R-squared per output channel (on denormalised values)."""
    pred_phase = np.rad2deg(np.arctan2(y_pred[:, 0], y_pred[:, 1]))
    true_phase = np.rad2deg(np.arctan2(y_true[:, 0], y_true[:, 1]))

    result = {}
    for label, yt, yp in [
        ("dphi", true_phase, pred_phase),
        ("T_TE", y_true[:, 2], y_pred[:, 2]),
        ("T_TM", y_true[:, 3], y_pred[:, 3]),
    ]:
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        result[label] = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    return result


def train_surrogate(
    records: list[dict],
    device: torch.device | None = None,
) -> tuple[ForwardSurrogate, StandardScaler, StandardScaler, dict[str, float]]:
    """Train a new surrogate model from scratch on the given CST data.

    Returns
    -------
    model : trained ForwardSurrogate in eval mode
    x_scaler, y_scaler : fitted StandardScalers
    r2 : dict of R-squared per output
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y = prepare_training_data(records)
    n = len(X)

    # Fit scalers
    x_scaler = StandardScaler(X)
    y_scaler = StandardScaler(Y)
    X_norm = x_scaler.transform(X)
    Y_norm = y_scaler.transform(Y)

    # Train/val split (80/20)
    n_val = max(1, int(n * 0.2))
    idx = np.random.permutation(n)
    X_train = torch.tensor(X_norm[idx[n_val:]], device=device)
    Y_train = torch.tensor(Y_norm[idx[n_val:]], device=device)
    X_val = torch.tensor(X_norm[idx[:n_val]], device=device)
    Y_val = torch.tensor(Y_norm[idx[:n_val]], device=device)

    train_ds = TensorDataset(X_train, Y_train)
    train_dl = DataLoader(train_ds, batch_size=SURROGATE_TRAIN_BATCH, shuffle=True)

    # Build model
    model = ForwardSurrogate(
        hidden=SURROGATE_HIDDEN,
        n_layers=SURROGATE_N_LAYERS,
        dropout=SURROGATE_DROPOUT,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=SURROGATE_TRAIN_LR,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=50, factor=0.5, min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = min(200, SURROGATE_TRAIN_EPOCHS // 3)

    for epoch in range(SURROGATE_TRAIN_EPOCHS):
        # Training
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.functional.mse_loss(val_pred, Y_val).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping (only after reasonable warmup)
        if epoch > max(50, SURROGATE_TRAIN_EPOCHS // 5) and patience_counter >= patience:
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Compute R² on full dataset (denormalised)
    with torch.no_grad():
        Y_pred_norm = model(torch.tensor(X_norm, device=device)).cpu().numpy()
    Y_pred = y_scaler.inverse(Y_pred_norm)
    r2 = compute_r2(Y, Y_pred)

    return model, x_scaler, y_scaler, r2
