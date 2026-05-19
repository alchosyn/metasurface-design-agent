"""
CLI entry point for the Optical Design Optimization Agent.

Usage:
    # Mock mode (uses pre-trained surrogate as ground truth):
    python -m agent.run --mock

    # Real CST mode:
    python -m agent.run --cst-project path/to/project.cst

    # Custom budget:
    python -m agent.run --mock --max-cst-calls 100
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import AGENT_OUTPUT_DIR, MAX_CST_CALLS, GRAPH_RECURSION_LIMIT
from agent.tools import SharedState


def setup_logging(log_file: Path | None = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )


def load_ground_truth_model():
    """Load the pre-trained surrogate from ml/ for mock CST mode."""
    ckpt_path = PROJECT_ROOT / "ml" / "checkpoints" / "surrogate_optimization.pth"
    if not ckpt_path.exists():
        logging.warning(
            f"No checkpoint at {ckpt_path}. Mock CST will use physics model."
        )
        return None

    import torch
    import numpy as np

    # Load with weights_only=False (our own checkpoint, trusted)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Reconstruct model using the agent's own ForwardSurrogate
    # (compatible architecture, just different default hidden size)
    cfg = ckpt["model_config"]
    from agent.surrogate.model import ForwardSurrogate, StandardScaler
    model = ForwardSurrogate(
        hidden=cfg["hidden"], n_layers=cfg["n_layers"],
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_scaler = StandardScaler.from_state(ckpt["x_scaler"])
    y_scaler = StandardScaler.from_state(ckpt["y_scaler"])

    logging.info(f"Loaded ground truth model from {ckpt_path}")
    return (model, x_scaler, y_scaler)


def save_run_results(shared: SharedState, run_dir: Path, elapsed: float):
    """Save experiment database and summary to disk."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save full database
    db_path = run_dir / "cst_database.json"
    with open(db_path, "w") as f:
        json.dump(shared.cst_database, f, indent=2, default=str)

    # Save summary
    summary = {
        "cst_calls": shared.cst_call_count,
        "max_budget": shared.max_cst_calls,
        "best_result": shared.best_result,
        "surrogate_trained": shared.surrogate_trained,
        "surrogate_r2": shared.surrogate_r2,
        "elapsed_seconds": elapsed,
        "mock_mode": shared.mock_cst,
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Run complete. Results saved to: {run_dir}")
    print(f"CST calls: {shared.cst_call_count}/{shared.max_cst_calls}")
    if shared.best_result:
        b = shared.best_result
        print(f"Best design: L={b['L']:.0f}, W={b['W']:.0f}, h={b['h']:.0f}")
        print(f"  dphi={b['dphi_deg']:.1f} deg, T_avg={b['T_avg_pct']:.1f}%, "
              f"FOM={b['fom']:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Optical Design Optimization Agent"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock CST (pre-trained surrogate as ground truth)",
    )
    parser.add_argument(
        "--cst-project", type=str, default=None,
        help="Path to CST .cst project file (required if not --mock)",
    )
    parser.add_argument(
        "--max-cst-calls", type=int, default=MAX_CST_CALLS,
        help=f"CST simulation budget (default: {MAX_CST_CALLS})",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="DeepSeek API key (default: from DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for this run",
    )
    args = parser.parse_args()

    # ── Setup ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else (
        AGENT_OUTPUT_DIR / f"run_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "agent.log")

    # ── Build shared state ──
    shared = SharedState(
        max_cst_calls=args.max_cst_calls,
        mock_cst=args.mock or (args.cst_project is None),
    )

    if shared.mock_cst:
        logging.info("Running in MOCK CST mode")
        shared.ground_truth_model = load_ground_truth_model()
    else:
        logging.info(f"Running with real CST: {args.cst_project}")
        from agent.cst.connection import CSTConnection
        shared.cst_connection = CSTConnection(args.cst_project)
        shared.cst_project_path = args.cst_project

    # ── Build agent ──
    from agent.graph import build_agent
    agent, shared = build_agent(
        shared=shared,
        max_cst_calls=args.max_cst_calls,
        api_key=args.api_key,
    )

    # ── Run ──
    print(f"\n{'='*60}")
    print(f"Optical Design Optimization Agent")
    print(f"Mode: {'Mock CST' if shared.mock_cst else 'Real CST'}")
    print(f"CST budget: {args.max_cst_calls}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    config = {
        "configurable": {"thread_id": f"run_{timestamp}"},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    initial_message = (
        "Begin optimization. You have 0 data points and a budget of "
        f"{args.max_cst_calls} CST calls. Start by planning your initial "
        "exploration strategy, then execute it."
    )

    # Initial state with all OptimizationState fields
    initial_state = {
        "messages": [("user", initial_message)],
        "cst_call_count": 0,
        "surrogate_trained": False,
        "surrogate_r2_dphi": -1.0,
        "best_fom": -999.0,
    }

    t0 = time.time()
    last_printed_id = None  # dedup across multi-node stream events
    try:
        for event in agent.stream(
            initial_state,
            config=config,
            stream_mode="values",
        ):
            if event.get("messages"):
                last_msg = event["messages"][-1]
                msg_id = getattr(last_msg, "id", None)

                # Skip if we already printed this message (sync_state
                # emits state without adding messages, so the last_msg
                # is unchanged from the previous event)
                if msg_id and msg_id == last_printed_id:
                    continue
                last_printed_id = msg_id

                if hasattr(last_msg, "content") and last_msg.content:
                    role = getattr(last_msg, "type", "unknown")
                    # Sanitise for Windows console (cp932 etc.)
                    text = last_msg.content
                    text = text.encode("ascii", errors="replace").decode("ascii")
                    if role == "ai":
                        print(f"\n[Agent] {text[:500]}")
                    elif role == "tool":
                        name = getattr(last_msg, "name", "tool")
                        print(f"\n[Tool: {name}] {text[:300]}")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        logging.exception(f"Agent error: {e}")
    finally:
        elapsed = time.time() - t0
        save_run_results(shared, run_dir, elapsed)

        # Cleanup CST connection
        if shared.cst_connection is not None:
            shared.cst_connection.close()


if __name__ == "__main__":
    main()
