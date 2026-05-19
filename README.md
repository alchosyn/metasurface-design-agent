# Metasurface Design Agent

LangGraph-based AI agent that autonomously optimizes TiO2 metasurface pillar geometry for double-helix point spread function (DH-PSF) depth sensing. The agent drives CST electromagnetic simulations, builds its own neural network surrogate model via active learning, and validates designs through Fresnel propagation.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │            LangGraph StateGraph          │
                        │                                         │
  START ──► agent ──► should_continue ──┬──► tools ──► sync_state ┤
                                        │         ▲               │
                                        │         ├── compress    │
                                        │         └── inject_hint │
                                        └──► END                  │
                        └─────────────────────────────────────────┘

  Two-layer state:
    OptimizationState (serializable, checkpointed)
    SharedState (torch models, COM connections, experiment DB)
```

The agent uses **DeepSeek-V3** for reasoning and tool selection, with a dual-track context compression strategy to maintain coherent long-horizon optimization within token budget limits.

## Optimization Workflow

The agent follows a mandatory 4-phase workflow:

| Phase | Description | CST Budget |
|-------|-------------|------------|
| **1. Exploration** | Coarse grid sampling across geometry space | 20 calls |
| **2. Surrogate** | Train MLP on CST data, dense grid scan + uncertainty estimation | 0 calls |
| **3. Refinement** | Targeted CST validation of surrogate predictions + high-uncertainty regions, retrain | Remaining budget |
| **4. Validation** | Fresnel PSF propagation of best candidate, report final metrics | 0 calls |

**Convergence criteria:** phase error < 2 deg, average transmittance > 90%, PSF rotation linearity R^2 > 0.9.

## Tools

Seven tools across three cost tiers:

| Tier | Tool | Cost | Description |
|------|------|------|-------------|
| 3 | `run_cst` | ~1 min | Full-wave EM simulation via CST COM interface |
| 2 | `train_surrogate` | ~30s | Train/retrain MLP on accumulated CST data |
| 2 | `run_fresnel_psf` | ~5s | 1000x1000 Fresnel propagation across 18 z-planes |
| 1 | `predict_surrogate` | <1s | Batch inference on up to 500 geometries |
| 1 | `estimate_uncertainty` | <1s | MC Dropout uncertainty quantification |
| 1 | `compute_fom` | <1s | Rank results by figure of merit |
| 1 | `query_database` | <1s | Browse and filter CST experiment history |

## Project Structure

```
agent/
├── run.py                  # CLI entry point
├── config.py               # Physical parameters and constants
├── graph.py                # LangGraph StateGraph assembly
├── state.py                # OptimizationState / SimResult
├── nodes.py                # Graph nodes (compress, sync_state, inject_hint)
├── prompts.py              # System prompt with 4-phase workflow
├── cst/                    # CST Studio Suite COM interface
│   ├── connection.py       #   COM connection manager (Windows)
│   ├── geometry.py         #   Set pillar dimensions
│   ├── solver.py           #   Run frequency domain solver
│   └── results.py          #   Extract S-parameters
├── surrogate/              # Online neural network surrogate
│   ├── model.py            #   3-layer MLP (64 hidden, dropout 0.1)
│   ├── trainer.py          #   Training loop with early stopping
│   └── uncertainty.py      #   MC Dropout (50 samples)
├── tools/                  # LangChain tool definitions
│   ├── cst_tool.py
│   ├── surrogate_tool.py
│   ├── fresnel_tool.py
│   ├── fom_tool.py
│   └── database_tool.py
└── evaluation/             # PSF quality assessment
    ├── fresnel.py          #   FFT-based Fresnel diffraction
    └── psf_quality.py      #   Peak finding + rotation linearity
```

## Getting Started

### Prerequisites

- Python 3.10+
- [DeepSeek API key](https://platform.deepseek.com/)
- For real simulations: Windows + CST Studio Suite with a configured `.cst` project

### Installation

```bash
git clone https://github.com/alchosyn/metasurface-design-agent.git
cd metasurface-design-agent
pip install langchain langchain-openai langgraph torch numpy scipy matplotlib
```

### Usage

Set your API key:

```bash
# Linux / macOS
export DEEPSEEK_API_KEY=sk-xxxxx

# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-xxxxx"
```

**Mock mode** (no CST required, uses pre-trained surrogate + noise):

```bash
python -m agent.run --mock --max-cst-calls 200 --output-dir ./results
```

**Real CST mode** (Windows only):

```bash
python -m agent.run --cst-project C:\path\to\project.cst --max-cst-calls 150
```

### Output

Each run creates a timestamped directory containing:

- `cst_database.json` -- all simulation results and best candidate
- `summary.json` -- run metrics, elapsed time, convergence status
- `agent.log` -- full execution transcript

## Design Problem

Optimize a TiO2 rectangular nanopillar on SiO2 substrate for Pancharatnam-Berry (PB) phase metasurface operating at 632 nm:

- **Goal:** half-wave plate condition |delta_phi| = 180 deg with maximum transmittance
- **Geometry:** length L in [60, 340] nm, width W in [60, 340] nm, height h in [600, 860] nm (L >= W)
- **Array:** 1000 x 1000 pillars, 400 nm period, aperture D = 400 um, NA = 0.1
- **FOM:** -10 * |abs(delta_phi) - 180| + (T_TE + T_TM) / 200

## License

This project is part of an MSc dissertation at Nanyang Technological University (EEE).
