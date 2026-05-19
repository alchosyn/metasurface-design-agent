"""
Ablation study: evaluate context compression strategies.

Runs the agent multiple times with different compression strategies,
collects performance metrics, and produces a comparison report.

Metrics collected per run:
  - final_fom:          best FOM achieved
  - cst_calls:          total CST simulations used
  - cst_to_threshold:   CST calls needed to reach FOM > -5 (near-HWP)
  - phase_err_final:    phase error of best design (deg)
  - T_avg_final:        average transmittance of best design (%)
  - redundant_fraction: fraction of post-compression CST calls that land
                        within 15 nm of a pre-compression point (measures
                        whether the agent re-explores known territory)

Usage:
    python -m agent.evaluation.compression_eval --repeats 3 --budget 50
    python -m agent.evaluation.compression_eval --strategies dual structured truncate
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

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import (
    AGENT_OUTPUT_DIR, MAX_CST_CALLS, GRAPH_RECURSION_LIMIT,
)
from agent.nodes import CompressionStrategy
from agent.tools import SharedState


logger = logging.getLogger(__name__)


# ── Metrics extraction ────────────────────────────────────────────────

FOM_THRESHOLD = -5.0       # "near-HWP" milestone
REDUNDANCY_RADIUS_NM = 15  # consider a CST call redundant if within this


def compute_run_metrics(shared: SharedState, compression_idx: int | None) -> dict:
    """Extract evaluation metrics from a completed agent run.

    Parameters
    ----------
    shared : SharedState
        The shared state after the agent run completes.
    compression_idx : int or None
        The CST call index at which compression first triggered.
        None if compression never triggered.

    Returns
    -------
    dict of metric name -> value.
    """
    db = shared.cst_database
    n = len(db)

    # Basic metrics
    best = shared.best_result or {}
    metrics = {
        "cst_calls": shared.cst_call_count,
        "final_fom": best.get("fom", -999),
        "phase_err_final": best.get("phase_err_deg", 999),
        "T_avg_final": best.get("T_avg_pct", 0),
        "surrogate_trained": shared.surrogate_trained,
    }

    # CST calls to reach threshold FOM
    threshold_idx = None
    for i, r in enumerate(db):
        if r["fom"] >= FOM_THRESHOLD:
            threshold_idx = i + 1
            break
    metrics["cst_to_threshold"] = threshold_idx  # None if never reached

    # Redundant exploration fraction (post-compression only)
    if compression_idx is not None and compression_idx < n:
        pre = db[:compression_idx]
        post = db[compression_idx:]
        if pre and post:
            pre_coords = np.array([[r["L"], r["W"], r["h"]] for r in pre])
            redundant = 0
            for r in post:
                pt = np.array([r["L"], r["W"], r["h"]])
                dists = np.linalg.norm(pre_coords - pt, axis=1)
                if dists.min() < REDUNDANCY_RADIUS_NM:
                    redundant += 1
            metrics["redundant_fraction"] = redundant / len(post)
            metrics["redundant_count"] = redundant
            metrics["post_compression_calls"] = len(post)
        else:
            metrics["redundant_fraction"] = None
    else:
        metrics["redundant_fraction"] = None

    return metrics


# ── Single run ────────────────────────────────────────────────────────

EVAL_TRIGGER_TOKENS = 4_000   # low threshold so compression fires in short runs
EVAL_TARGET_TOKENS = 2_000


def run_single(
    strategy: CompressionStrategy,
    max_cst_calls: int,
    api_key: str,
    seed: int,
) -> dict:
    """Run one agent with a given compression strategy and return metrics."""
    # Deterministic mock CST noise
    np.random.seed(seed)

    from agent.run import load_ground_truth_model
    from agent.graph import build_agent

    shared = SharedState(max_cst_calls=max_cst_calls, mock_cst=True)
    shared.ground_truth_model = load_ground_truth_model()

    agent, shared = build_agent(
        shared=shared,
        max_cst_calls=max_cst_calls,
        api_key=api_key,
        compression=strategy,
        compress_trigger_tokens=EVAL_TRIGGER_TOKENS,
        compress_target_tokens=EVAL_TARGET_TOKENS,
    )

    config = {
        "configurable": {"thread_id": f"eval_{strategy.value}_{seed}"},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    initial_state = {
        "messages": [(
            "user",
            f"Begin optimization. Budget: {max_cst_calls} CST calls. "
            "Start with Phase 1 initial exploration.",
        )],
        "cst_call_count": 0,
        "surrogate_trained": False,
        "surrogate_r2_dphi": -1.0,
        "best_fom": -999.0,
    }

    # Track when compression first fires
    compression_idx = None
    last_cst_count = 0

    t0 = time.time()
    try:
        for event in agent.stream(
            initial_state, config=config, stream_mode="values",
        ):
            # Detect compression event (message count drops)
            if compression_idx is None:
                cur_cst = shared.cst_call_count
                msgs = event.get("messages", [])
                # Check for compression summary in messages
                for m in msgs:
                    content = getattr(m, "content", "") or ""
                    if "STRUCTURED EXPERIMENT STATE" in content or "Context compressed" in content:
                        compression_idx = cur_cst
                        logger.info(
                            f"[{strategy.value}] Compression triggered at "
                            f"CST call #{cur_cst}"
                        )
                        break
    except Exception as e:
        logger.error(f"[{strategy.value}] Run failed: {e}")

    elapsed = time.time() - t0
    metrics = compute_run_metrics(shared, compression_idx)
    metrics["elapsed_s"] = elapsed
    metrics["strategy"] = strategy.value
    metrics["seed"] = seed
    metrics["compression_triggered_at"] = compression_idx

    return metrics


# ── Comparison report ─────────────────────────────────────────────────

def format_report(all_results: dict[str, list[dict]]) -> str:
    """Format a comparison table from results grouped by strategy."""
    lines = []
    lines.append("=" * 72)
    lines.append("COMPRESSION STRATEGY ABLATION STUDY")
    lines.append("=" * 72)

    header = (
        f"{'Strategy':<12} {'FOM':>8} {'Phase':>8} {'T_avg':>8} "
        f"{'CST':>5} {'Redund':>8} {'Time':>7}"
    )
    lines.append(header)
    lines.append("-" * 72)

    for strategy_name, runs in all_results.items():
        foms = [r["final_fom"] for r in runs]
        phase_errs = [r["phase_err_final"] for r in runs]
        t_avgs = [r["T_avg_final"] for r in runs]
        csts = [r["cst_calls"] for r in runs]
        redunds = [r["redundant_fraction"] for r in runs
                    if r["redundant_fraction"] is not None]
        times = [r["elapsed_s"] for r in runs]

        fom_str = f"{np.mean(foms):>7.3f}" + (f"±{np.std(foms):.3f}" if len(foms) > 1 else "")
        phase_str = f"{np.mean(phase_errs):>6.1f}°"
        t_str = f"{np.mean(t_avgs):>6.1f}%"
        cst_str = f"{np.mean(csts):>5.0f}"
        red_str = f"{np.mean(redunds)*100:>5.1f}%" if redunds else "  n/a"
        time_str = f"{np.mean(times):>5.0f}s"

        lines.append(
            f"{strategy_name:<12} {fom_str:>15} {phase_str:>8} {t_str:>8} "
            f"{cst_str:>5} {red_str:>8} {time_str:>7}"
        )

    lines.append("-" * 72)

    # Per-run details
    lines.append("\nPer-run details:")
    for strategy_name, runs in all_results.items():
        for r in runs:
            comp_at = r.get("compression_triggered_at")
            comp_str = f"comp@CST#{comp_at}" if comp_at else "no compression"
            lines.append(
                f"  [{strategy_name}] seed={r['seed']}: "
                f"FOM={r['final_fom']:.3f}, "
                f"phase_err={r['phase_err_final']:.1f}°, "
                f"T_avg={r['T_avg_final']:.1f}%, "
                f"CST={r['cst_calls']}, "
                f"{comp_str}, "
                f"{r['elapsed_s']:.0f}s"
            )

    lines.append("=" * 72)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: compression strategy comparison"
    )
    parser.add_argument(
        "--strategies", nargs="+",
        default=["dual", "structured", "truncate"],
        help="Strategies to compare (default: dual structured truncate)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Runs per strategy (default: 1; use 3+ for statistics)",
    )
    parser.add_argument(
        "--budget", type=int, default=50,
        help="CST budget per run (default: 50)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="DeepSeek API key",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: set DEEPSEEK_API_KEY or use --api-key")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    strategies = [CompressionStrategy(s) for s in args.strategies]
    all_results: dict[str, list[dict]] = {}

    print(f"\nAblation study: {len(strategies)} strategies × {args.repeats} repeats")
    print(f"CST budget: {args.budget} per run")
    print(f"Total runs: {len(strategies) * args.repeats}\n")

    for strat in strategies:
        results = []
        for rep in range(args.repeats):
            seed = 42 + rep
            print(f"─── Running: {strat.value} (seed={seed}) ───")
            metrics = run_single(strat, args.budget, api_key, seed)
            results.append(metrics)
            print(
                f"    Result: FOM={metrics['final_fom']:.3f}, "
                f"CST={metrics['cst_calls']}, "
                f"{metrics['elapsed_s']:.0f}s\n"
            )
        all_results[strat.value] = results

    # ── Report ──
    report = format_report(all_results)
    print(report)

    # Save results
    out_dir = AGENT_OUTPUT_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(out_dir / f"ablation_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(out_dir / f"ablation_{ts}.txt", "w") as f:
        f.write(report)

    print(f"\nSaved to {out_dir}/ablation_{ts}.*")


if __name__ == "__main__":
    main()
