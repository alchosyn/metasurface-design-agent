"""
Random sampling baseline for comparison with the LLM agent.

Core question: can 45 random CST calls find a design as good as the agent?
If yes, the agent adds no value. If no, its domain-knowledge-driven
exploration is genuinely more efficient.

Three baselines:
  1. **Uniform random**: sample (L, W, h) uniformly in the full parameter
     space.  The dumbest possible strategy.
  2. **Latin Hypercube**: space-filling design — better coverage per sample
     than pure random.  Standard DoE baseline.
  3. **Grid baseline**: regular grid at matching sample count, as a
     reference for structured-but-uninformed exploration.

All use the same mock CST ground truth model as the agent, with the same
noise level, so results are directly comparable.

Usage:
    python -m agent.evaluation.random_baseline --budget 45 --repeats 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import (
    AGENT_OUTPUT_DIR, L_MIN, L_MAX, W_MIN, W_MAX, H_MIN, H_MAX,
    FOM_PHASE_WEIGHT, FOM_TRANS_WEIGHT,
)

logger = logging.getLogger(__name__)


# ── Mock simulation (same as cst_tool._mock_simulate) ────────────────

def _load_ground_truth():
    """Load the pre-trained surrogate from ml/checkpoints."""
    from agent.run import load_ground_truth_model
    return load_ground_truth_model()


def mock_simulate(L, W, h, gt_model, rng: np.random.RandomState):
    """Evaluate one (L, W, h) with mock CST — identical to agent's mock."""
    if gt_model is not None:
        import torch
        model, x_scaler, y_scaler = gt_model
        device = next(model.parameters()).device
        g = np.array([[L, W, h]], dtype=np.float32)
        gn = (g - x_scaler.mean) / x_scaler.std
        with torch.no_grad():
            y = model(torch.tensor(gn, device=device))
            y = y.cpu().numpy()[0] * y_scaler.std + y_scaler.mean
        dphi = float(np.rad2deg(np.arctan2(y[0], y[1])))
        T_TE = float(np.clip(y[2], 0, 100))
        T_TM = float(np.clip(y[3], 0, 100))
    else:
        aspect = L / max(W, 1)
        dphi = -180.0 * np.tanh((aspect - 1.5) * 0.8) * (h / 700) ** 0.5
        T_TE = 95.0 - 0.01 * (L - 200) ** 2 / 100
        T_TM = 90.0 - 0.02 * (W - 100) ** 2 / 100

    # Same noise as agent
    dphi = float(dphi + rng.normal(0, 1.5))
    T_TE = float(np.clip(T_TE + rng.normal(0, 0.8), 0, 100))
    T_TM = float(np.clip(T_TM + rng.normal(0, 0.8), 0, 100))

    phase_err = abs(abs(dphi) - 180.0)
    T_avg = (T_TE + T_TM) / 2.0
    fom = -FOM_PHASE_WEIGHT * phase_err + FOM_TRANS_WEIGHT * T_avg / 100.0

    return {
        "L": L, "W": W, "h": h,
        "dphi_deg": dphi, "T_TE_pct": T_TE, "T_TM_pct": T_TM,
        "T_avg_pct": T_avg, "phase_err_deg": phase_err, "fom": fom,
    }


# ── Sampling strategies ──────────────────────────────────────────────

def sample_uniform(n: int, rng: np.random.RandomState):
    """Uniform random (L, W, h) with L >= W constraint."""
    points = []
    while len(points) < n:
        L = rng.uniform(L_MIN, L_MAX)
        W = rng.uniform(W_MIN, W_MAX)
        h = rng.uniform(H_MIN, H_MAX)
        if L >= W:
            points.append((L, W, h))
    return points


def sample_lhs(n: int, rng: np.random.RandomState):
    """Latin Hypercube Sampling (L, W, h) with L >= W constraint.

    Generates 2× points in LHS, filters L>=W, takes first n.
    Falls back to uniform if LHS doesn't yield enough valid points.
    """
    n_gen = n * 3  # oversample to account for L>=W rejection
    dims = 3

    # Build LHS grid
    intervals = np.arange(n_gen) / n_gen
    perms = np.column_stack([rng.permutation(n_gen) for _ in range(dims)])
    samples = (perms + rng.uniform(size=(n_gen, dims))) / n_gen

    # Map to parameter ranges
    L = L_MIN + samples[:, 0] * (L_MAX - L_MIN)
    W = W_MIN + samples[:, 1] * (W_MAX - W_MIN)
    h = H_MIN + samples[:, 2] * (H_MAX - H_MIN)

    # Filter L >= W
    valid = L >= W
    L, W, h = L[valid], W[valid], h[valid]

    if len(L) < n:
        # Fall back to uniform fill
        extra = sample_uniform(n - len(L), rng)
        points = list(zip(L, W, h)) + extra
    else:
        points = list(zip(L[:n], W[:n], h[:n]))

    return [(float(l), float(w), float(hh)) for l, w, hh in points]


def sample_grid(n: int):
    """Regular grid with ~n points (cube root spacing), L >= W only."""
    k = max(2, int(round(n ** (1/3))))
    Ls = np.linspace(L_MIN, L_MAX, k)
    Ws = np.linspace(W_MIN, W_MAX, k)
    Hs = np.linspace(H_MIN, H_MAX, k)

    points = [
        (float(L), float(W), float(h))
        for L in Ls for W in Ws for h in Hs
        if L >= W
    ]
    return points[:n]  # trim to budget


# ── Run one baseline experiment ──────────────────────────────────────

def run_baseline(
    strategy: str,
    budget: int,
    gt_model,
    seed: int,
) -> dict:
    """Run one random baseline trial and return metrics."""
    rng = np.random.RandomState(seed)

    if strategy == "uniform":
        points = sample_uniform(budget, rng)
    elif strategy == "lhs":
        points = sample_lhs(budget, rng)
    elif strategy == "grid":
        points = sample_grid(budget)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Evaluate all points
    results = []
    for L, W, h in points:
        r = mock_simulate(L, W, h, gt_model, rng)
        results.append(r)

    # Metrics
    foms = [r["fom"] for r in results]
    best = max(results, key=lambda r: r["fom"])

    # FOM convergence: best FOM seen after k calls
    fom_curve = []
    running_best = -999
    for r in results:
        running_best = max(running_best, r["fom"])
        fom_curve.append(running_best)

    # Calls to reach threshold
    threshold = -5.0
    cst_to_threshold = None
    for i, f in enumerate(fom_curve):
        if f >= threshold:
            cst_to_threshold = i + 1
            break

    return {
        "strategy": strategy,
        "seed": seed,
        "budget": budget,
        "cst_calls": len(results),
        "final_fom": best["fom"],
        "phase_err_final": best["phase_err_deg"],
        "T_avg_final": best["T_avg_pct"],
        "best_L": best["L"],
        "best_W": best["W"],
        "best_h": best["h"],
        "cst_to_threshold": cst_to_threshold,
        "fom_curve": fom_curve,
    }


# ── Report ───────────────────────────────────────────────────────────

def format_baseline_report(
    all_results: dict[str, list[dict]],
    agent_results: dict | None = None,
) -> str:
    """Format comparison table."""
    lines = []
    lines.append("=" * 72)
    lines.append("RANDOM BASELINE vs AGENT COMPARISON")
    lines.append("=" * 72)

    header = (
        f"{'Strategy':<12} {'FOM (mean)':>12} {'FOM (best)':>12} "
        f"{'Phase err':>10} {'T_avg':>8} {'>Thr':>6}"
    )
    lines.append(header)
    lines.append("-" * 72)

    for name, runs in all_results.items():
        foms = [r["final_fom"] for r in runs]
        phase_errs = [r["phase_err_final"] for r in runs]
        t_avgs = [r["T_avg_final"] for r in runs]
        thrs = [r["cst_to_threshold"] for r in runs
                if r["cst_to_threshold"] is not None]

        n = len(foms)
        fom_mean = np.mean(foms)
        fom_std = np.std(foms) if n > 1 else 0
        fom_best = np.max(foms)
        phase_mean = np.mean(phase_errs)
        t_mean = np.mean(t_avgs)
        thr_frac = f"{len(thrs)}/{n}"

        lines.append(
            f"{name:<12} "
            f"{fom_mean:>7.3f}±{fom_std:<4.3f} "
            f"{fom_best:>12.3f} "
            f"{phase_mean:>8.1f}° "
            f"{t_mean:>7.1f}% "
            f"{thr_frac:>6}"
        )

    # Agent results (if provided)
    if agent_results:
        lines.append("-" * 72)
        for name, runs in agent_results.items():
            foms = [r["final_fom"] for r in runs]
            phase_errs = [r["phase_err_final"] for r in runs]
            t_avgs = [r["T_avg_final"] for r in runs]
            n = len(foms)
            fom_mean = np.mean(foms)
            fom_std = np.std(foms) if n > 1 else 0
            fom_best = np.max(foms)
            lines.append(
                f"{'agent/'+name:<12} "
                f"{fom_mean:>7.3f}±{fom_std:<4.3f} "
                f"{fom_best:>12.3f} "
                f"{np.mean(phase_errs):>8.1f}° "
                f"{np.mean(t_avgs):>7.1f}% "
                f"{'n/a':>6}"
            )

    lines.append("-" * 72)

    # Distribution analysis
    lines.append("\nFOM distribution (per-run best):")
    for name, runs in all_results.items():
        foms = sorted([r["final_fom"] for r in runs])
        n = len(foms)
        p = lambda q: foms[min(int(n * q), n - 1)]
        lines.append(
            f"  {name:<10}: "
            f"min={foms[0]:.3f}, "
            f"p25={p(0.25):.3f}, "
            f"median={p(0.5):.3f}, "
            f"p75={p(0.75):.3f}, "
            f"max={foms[-1]:.3f}"
        )

    # Best designs found
    lines.append("\nBest design per strategy (single best across all repeats):")
    for name, runs in all_results.items():
        best_run = max(runs, key=lambda r: r["final_fom"])
        lines.append(
            f"  {name:<10}: L={best_run['best_L']:.0f}, "
            f"W={best_run['best_W']:.0f}, h={best_run['best_h']:.0f}, "
            f"FOM={best_run['final_fom']:.3f}, "
            f"phase_err={best_run['phase_err_final']:.1f}°"
        )

    lines.append("=" * 72)
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Random baseline comparison for agent evaluation"
    )
    parser.add_argument("--budget", type=int, default=45,
                        help="CST calls per trial (match agent budget)")
    parser.add_argument("--repeats", type=int, default=200,
                        help="Trials per strategy (200 for robust stats)")
    parser.add_argument("--strategies", nargs="+",
                        default=["uniform", "lhs", "grid"],
                        help="Baseline strategies to test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s")

    gt_model = _load_ground_truth()

    print(f"\nRandom baseline: {len(args.strategies)} strategies × "
          f"{args.repeats} repeats × {args.budget} CST budget\n")

    all_results: dict[str, list[dict]] = {}
    for strat in args.strategies:
        results = []
        t0 = time.time()
        for rep in range(args.repeats):
            seed = 42 + rep
            metrics = run_baseline(strat, args.budget, gt_model, seed)
            results.append(metrics)
        elapsed = time.time() - t0
        all_results[strat] = results
        best_fom = max(r["final_fom"] for r in results)
        mean_fom = np.mean([r["final_fom"] for r in results])
        print(f"  {strat}: {args.repeats} runs in {elapsed:.1f}s, "
              f"mean FOM={mean_fom:.3f}, best FOM={best_fom:.3f}")

    # ── Report ──
    report = format_baseline_report(all_results)
    print(f"\n{report}")

    # Save
    out_dir = AGENT_OUTPUT_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Strip fom_curve for JSON (too large)
    save_results = {}
    for k, runs in all_results.items():
        save_results[k] = [
            {kk: vv for kk, vv in r.items() if kk != "fom_curve"}
            for r in runs
        ]
    with open(out_dir / f"baseline_{ts}.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    with open(out_dir / f"baseline_{ts}.txt", "w") as f:
        f.write(report)

    print(f"\nSaved to {out_dir}/baseline_{ts}.*")


if __name__ == "__main__":
    main()
