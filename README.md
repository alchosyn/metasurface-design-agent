# Metasurface Design Agent

AI agent that optimizes TiO2 nanopillar geometry for DH-PSF metasurface depth sensing. Built on LangGraph. Drives CST simulations, trains its own surrogate model, validates via Fresnel propagation.

## How It Works

The agent has 7 tools across 3 cost tiers. It manages a simulation budget autonomously:

1. **Explore** -- run 20 CST full-wave simulations to sample the design space
2. **Build surrogate** -- train an MLP on collected data, scan dense grids, estimate uncertainty via MC Dropout
3. **Refine** -- validate surrogate predictions with targeted CST calls, retrain, repeat
4. **Validate** -- propagate the best design through 1000x1000 Fresnel diffraction, check PSF quality

## Key Technical Choices

**Two-layer state.** LangGraph checkpoints a serializable `OptimizationState` (call counts, metrics, best FOM). Non-serializable objects (torch models, COM handles, experiment DB) live in a `SharedState` closure. A `sync_state` node bridges them after every tool call.

**Dual-track context compression.** Track 1 deterministically serializes experiment facts from SharedState. Track 2 uses an LLM call to extract strategic reasoning (what regions were abandoned, what the current plan is). Fires when token count exceeds 32K.

**Surrogate with uncertainty.** 3-layer MLP (64 hidden, dropout 0.1). Phase output uses sin/cos encoding to avoid the +/-180 discontinuity. MC Dropout (50 forward passes) flags high-uncertainty regions for targeted CST validation.

**Conditional hint injection.** At 18/20 CST calls, the graph injects a nudge to train the surrogate before hitting the hard budget wall at 20.

## Structure

```
agent/
├── run.py              # Entry point
├── graph.py            # StateGraph wiring
├── nodes.py            # compress, sync_state, inject_hint
├── config.py           # Physical constants
├── prompts.py          # System prompt
├── cst/                # CST COM automation (geometry, solver, S-param extraction)
├── surrogate/          # MLP model, online trainer, MC Dropout uncertainty
├── tools/              # 7 LangChain tools (cst, surrogate, fresnel, fom, database)
└── evaluation/         # Fresnel propagation + PSF rotation linearity
```

## Usage

```bash
# Mock mode (no CST needed)
export DEEPSEEK_API_KEY=sk-xxxxx
python -m agent.run --mock --max-cst-calls 200

# Real CST (Windows only)
python -m agent.run --cst-project path/to/project.cst
```

Outputs a timestamped run directory with simulation database, summary metrics, and agent log.

## Stack

LangGraph, LangChain, PyTorch, DeepSeek-V3, CST Studio Suite (COM), NumPy/SciPy (Fresnel FFT).

---

MSc dissertation project, NTU EEE.
