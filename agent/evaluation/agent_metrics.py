"""
Comprehensive agent evaluation — collects all three key metrics:

  1. Surrogate retraining progression (R2 vs data points)
  2. Budget utilisation (convergence point vs 200 budget)
  3. Random baseline comparison (agent FOM vs random FOM distribution)

Runs a single agent mock and the random baseline, then produces a
unified report suitable for the dissertation.

Usage:
    python -m agent.evaluation.agent_metrics --budget 200 --seed 42
    python -m agent.evaluation.agent_metrics --budget 200 --seed 42 --baseline-repeats 500
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
    AGENT_OUTPUT_DIR, MAX_CST_CALLS, GRAPH_RECURSION_LIMIT,
    FOM_PHASE_WEIGHT, FOM_TRANS_WEIGHT,
)
from agent.tools import SharedState

logger = logging.getLogger(__name__)


# ── Run one full agent ───────────────────────────────────────────────

def run_agent(
    max_cst_calls: int,
    api_key: str,
    seed: int,
) -> SharedState:
    """Run one agent and return its SharedState for post-hoc analysis."""
    np.random.seed(seed)

    from agent.run import load_ground_truth_model
    from agent.graph import build_agent

    shared = SharedState(max_cst_calls=max_cst_calls, mock_cst=True)
    shared.ground_truth_model = load_ground_truth_model()

    agent, shared = build_agent(
        shared=shared,
        max_cst_calls=max_cst_calls,
        api_key=api_key,
    )

    config = {
        "configurable": {"thread_id": f"metrics_{seed}"},
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

    t0 = time.time()
    try:
        for event in agent.stream(
            initial_state, config=config, stream_mode="values",
        ):
            msgs = event.get("messages", [])
            if msgs:
                last = msgs[-1]
                role = getattr(last, "type", "?")
                content = (getattr(last, "content", "") or "")[:120]
                if role == "ai" and content:
                    content_safe = content.encode("ascii", errors="replace").decode("ascii")
                    print(f"  [{shared.cst_call_count}/{max_cst_calls}] {content_safe}")
    except Exception as e:
        logger.error(f"Agent error: {e}")

    elapsed = time.time() - t0
    print(f"  Agent finished: {shared.cst_call_count} CST calls, "
          f"{elapsed:.0f}s, best FOM={shared.best_result['fom']:.3f}"
          if shared.best_result else "  Agent finished (no results)")
    return shared


# ── Metric 1: Surrogate retraining progression ──────────────────────

def report_surrogate_progression(shared: SharedState) -> str:
    """Format surrogate R2 progression across retraining events."""
    history = shared.surrogate_train_history
    if not history:
        return "No surrogate training events recorded."

    lines = [
        "── Metric 1: Surrogate Retraining Progression ──",
        f"Total retraining events: {len(history)}",
        "",
        f"{'#':>3} {'Points':>7} {'CST#':>5}  {'R2_dphi':>8} {'R2_T_TE':>8} {'R2_T_TM':>8}  {'Δ R2_dphi':>10}",
        "-" * 68,
    ]

    prev_r2 = None
    for i, h in enumerate(history):
        delta = ""
        if prev_r2 is not None:
            d = h["r2_dphi"] - prev_r2
            delta = f"{d:>+10.3f}"
        else:
            delta = f"{'(first)':>10}"

        lines.append(
            f"{i+1:>3} {h['n_points']:>7} {h['cst_call_count']:>5}  "
            f"{h['r2_dphi']:>8.3f} {h['r2_T_TE']:>8.3f} {h['r2_T_TM']:>8.3f}  "
            f"{delta}"
        )
        prev_r2 = h["r2_dphi"]

    # Summary
    first_r2 = history[0]["r2_dphi"]
    last_r2 = history[-1]["r2_dphi"]
    first_n = history[0]["n_points"]
    last_n = history[-1]["n_points"]
    lines.append("-" * 68)
    lines.append(
        f"R2_dphi improvement: {first_r2:.3f} ({first_n} pts) → "
        f"{last_r2:.3f} ({last_n} pts), Δ={last_r2 - first_r2:+.3f}"
    )

    return "\n".join(lines)


# ── Metric 2: Budget utilisation ─────────────────────────────────────

def report_budget_utilisation(shared: SharedState) -> str:
    """Analyse CST budget efficiency and convergence dynamics."""
    db = shared.cst_database
    n = len(db)
    budget = shared.max_cst_calls

    lines = [
        "── Metric 2: Budget Utilisation ──",
        f"Budget: {budget}, Used: {n} ({n/budget*100:.0f}%)",
        f"Savings: {budget - n} calls ({(budget-n)/budget*100:.0f}% of budget unused)",
        "",
    ]

    if not db:
        lines.append("No CST data collected.")
        return "\n".join(lines)

    # FOM convergence curve
    fom_curve = []
    best_so_far = -999.0
    for r in db:
        best_so_far = max(best_so_far, r["fom"])
        fom_curve.append(best_so_far)

    # Phase milestones
    milestones = [
        ("FOM > -50 (any birefringence)", -50.0),
        ("FOM > -10 (near HWP)", -10.0),
        ("FOM > -5  (close to HWP)", -5.0),
        ("FOM > -1  (excellent HWP)", -1.0),
        ("FOM > 0   (T_avg bonus > phase penalty)", 0.0),
        ("FOM > 0.5 (high quality)", 0.5),
        ("FOM > 0.9 (near perfect)", 0.9),
    ]

    lines.append("Convergence milestones:")
    for label, threshold in milestones:
        hit = None
        for i, f in enumerate(fom_curve):
            if f >= threshold:
                hit = i + 1
                break
        if hit:
            lines.append(f"  CST #{hit:>3}: {label}")
        else:
            lines.append(f"  never  : {label}")

    # Exploration phases
    lines.append("")
    if shared.surrogate_train_history:
        first_train = shared.surrogate_train_history[0]["cst_call_count"]
        lines.append(f"Phase 1 (blind exploration): CST #1 - #{first_train}")
        lines.append(f"Phase 2 (surrogate-guided):  CST #{first_train+1} - #{n}")
        lines.append(f"  Phase 1 calls: {first_train}")
        lines.append(f"  Phase 2 calls: {n - first_train}")

        # FOM at phase transition
        if first_train <= len(fom_curve):
            fom_at_train = fom_curve[first_train - 1]
            lines.append(f"  FOM at first training: {fom_at_train:.3f}")
            lines.append(f"  FOM at end:           {fom_curve[-1]:.3f}")
            lines.append(
                f"  Phase 2 improvement:  {fom_curve[-1] - fom_at_train:+.3f}"
            )

    # Best result
    if shared.best_result:
        b = shared.best_result
        lines.append("")
        lines.append(
            f"Best design: L={b['L']:.0f}, W={b['W']:.0f}, h={b['h']:.0f}"
        )
        lines.append(
            f"  dphi={b['dphi_deg']:.1f}°, T_avg={b['T_avg_pct']:.1f}%, "
            f"FOM={b['fom']:.3f}"
        )

    return "\n".join(lines)


# ── Metric 3: Random baseline comparison ─────────────────────────────

def run_random_baselines(
    budget: int,
    n_repeats: int,
    gt_model,
) -> dict[str, list[dict]]:
    """Run random baselines and return results per strategy."""
    from agent.evaluation.random_baseline import run_baseline
    results = {}
    for strat in ["uniform", "lhs"]:
        runs = []
        for rep in range(n_repeats):
            r = run_baseline(strat, budget, gt_model, seed=42 + rep)
            runs.append(r)
        results[strat] = runs
    return results


def report_baseline_comparison(
    shared: SharedState,
    baseline_results: dict[str, list[dict]],
) -> str:
    """Compare agent performance against random baselines."""
    agent_fom = shared.best_result["fom"] if shared.best_result else -999
    agent_phase = shared.best_result["phase_err_deg"] if shared.best_result else 999
    agent_calls = shared.cst_call_count

    lines = [
        "── Metric 3: Agent vs Random Baseline ──",
        f"Agent: {agent_calls} CST calls, FOM={agent_fom:.3f}, "
        f"phase_err={agent_phase:.1f}°",
        "",
        f"{'Strategy':<10} {'Calls':>5} {'FOM mean':>10} {'FOM med':>10} "
        f"{'FOM max':>10} {'P(FOM>0)':>10} {'P(>agent)':>10}",
        "-" * 72,
    ]

    for name, runs in baseline_results.items():
        foms = [r["final_fom"] for r in runs]
        n = len(foms)
        p_above_0 = sum(1 for f in foms if f > 0) / n
        p_above_agent = sum(1 for f in foms if f > agent_fom) / n

        lines.append(
            f"{name:<10} {runs[0]['budget']:>5} "
            f"{np.mean(foms):>10.3f} "
            f"{np.median(foms):>10.3f} "
            f"{np.max(foms):>10.3f} "
            f"{p_above_0:>9.1%} "
            f"{p_above_agent:>9.1%}"
        )

    # Agent row
    lines.append(
        f"{'AGENT':<10} {agent_calls:>5} "
        f"{agent_fom:>10.3f} "
        f"{'—':>10} "
        f"{agent_fom:>10.3f} "
        f"{'—':>10} "
        f"{'—':>10}"
    )

    lines.append("-" * 72)

    # Statistical significance
    for name, runs in baseline_results.items():
        foms = sorted([r["final_fom"] for r in runs])
        n = len(foms)
        # What percentile is the agent's FOM in the baseline distribution?
        rank = sum(1 for f in foms if f < agent_fom)
        pctile = rank / n * 100
        lines.append(
            f"Agent FOM ({agent_fom:.3f}) is at the {pctile:.1f}th "
            f"percentile of {name} (n={n})"
        )

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive agent evaluation (3 key metrics)"
    )
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-repeats", type=int, default=500)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--skip-agent", action="store_true",
                        help="Skip agent run, only run baselines")
    parser.add_argument("--agent-results", type=str, default=None,
                        help="Load agent results from JSON instead of running")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key and not args.skip_agent:
        print("ERROR: set DEEPSEEK_API_KEY or use --api-key")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

    # ── Load ground truth ──
    from agent.run import load_ground_truth_model
    gt_model = load_ground_truth_model()

    # ── Run agent ──
    shared = None
    if not args.skip_agent:
        print(f"\n{'='*60}")
        print(f"Running agent (budget={args.budget}, seed={args.seed})")
        print(f"{'='*60}\n")
        shared = run_agent(args.budget, api_key, args.seed)

    # ── Run random baselines ──
    # Use same budget as agent actually used (not full 200)
    actual_budget = shared.cst_call_count if shared else args.budget
    print(f"\n{'='*60}")
    print(f"Running random baselines ({args.baseline_repeats} repeats × "
          f"{actual_budget} budget)")
    print(f"{'='*60}\n")
    baseline_results = run_random_baselines(
        actual_budget, args.baseline_repeats, gt_model,
    )

    # ── Reports ──
    print(f"\n{'='*72}")
    print("COMPREHENSIVE AGENT EVALUATION")
    print(f"{'='*72}\n")

    if shared:
        print(report_surrogate_progression(shared))
        print()
        print(report_budget_utilisation(shared))
        print()
        print(report_baseline_comparison(shared, baseline_results))
    else:
        # Baseline-only mode
        from agent.evaluation.random_baseline import format_baseline_report
        print(format_baseline_report(baseline_results))

    # ── Save ──
    out_dir = AGENT_OUTPUT_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if shared:
        save_data = {
            "agent": {
                "cst_calls": shared.cst_call_count,
                "max_budget": shared.max_cst_calls,
                "best_result": shared.best_result,
                "surrogate_train_history": shared.surrogate_train_history,
                "cst_database": shared.cst_database,
            },
            "baselines": {
                k: [{kk: vv for kk, vv in r.items() if kk != "fom_curve"}
                    for r in runs]
                for k, runs in baseline_results.items()
            },
        }
        with open(out_dir / f"metrics_{ts}.json", "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nSaved to {out_dir}/metrics_{ts}.json")


if __name__ == "__main__":
    main()
