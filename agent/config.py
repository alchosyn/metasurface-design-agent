"""
Configuration constants for the Optical Design Optimization Agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
ML_DIR = PROJECT_ROOT / "ml"
DATA_DIR = PROJECT_ROOT / "data"
MATLAB_DIR = PROJECT_ROOT / "matlab" / "Double Helix"
PHASEORI_MAT = MATLAB_DIR / "3 DHPSF Propagation" / "phaseori.mat"
ROTATION_ANGLES_CSV = (
    MATLAB_DIR / "3 DHPSF Propagation" / "Pillar_Rotation_Angles_Deg.csv"
)

# Agent output directory (created per run)
AGENT_OUTPUT_DIR = DATA_DIR / "agent_runs"

# ── Physical constants ─────────────────────────────────────────────
WAVELENGTH_UM = 0.632          # He-Ne laser, μm
WAVELENGTH_NM = 632.0          # nm
NA = 0.1                       # numerical aperture
APERTURE_UM = 400.0            # aperture diameter D, μm
PERIOD_NM = 400.0              # unit cell period, nm
ARRAY_SIZE = 1000              # 1000 × 1000 pillar array

# Pillar geometry bounds (nm)
L_MIN, L_MAX = 60.0, 340.0
W_MIN, W_MAX = 60.0, 340.0
H_MIN, H_MAX = 600.0, 860.0

# Design target
PHASE_TARGET_DEG = 180.0       # half-wave plate |Δφ| = 180°

# ── Surrogate model defaults ──────────────────────────────────────
SURROGATE_HIDDEN = 64          # smaller than original 128 (less data)
SURROGATE_N_LAYERS = 3
SURROGATE_DROPOUT = 0.1        # enable dropout for MC uncertainty
SURROGATE_TRAIN_EPOCHS = 1000
SURROGATE_TRAIN_LR = 1e-3
SURROGATE_TRAIN_BATCH = 32
SURROGATE_MIN_POINTS = 20
MC_DROPOUT_SAMPLES = 50        # forward passes for uncertainty

# ── FOM weights ───────────────────────────────────────────────────
FOM_PHASE_WEIGHT = 10.0        # penalty per degree of phase error
FOM_TRANS_WEIGHT = 1.0         # reward for transmittance

# ── Agent defaults ────────────────────────────────────────────────
MAX_CST_CALLS = 200
MAX_ITERATIONS = 50
GRAPH_RECURSION_LIMIT = 500    # nodes per run (each tool cycle ≈ 3-4 nodes)

# ── Fresnel propagation ──────────────────────────────────────────
FRESNEL_Z_POSITIONS_UM = [
    1750, 1780, 1810, 1840, 1870, 1900, 1930, 1960,
    1990, 2020, 2050, 2080, 2110, 2140, 2170, 2200,
    2230, 2260,
]
PSF_CROP_SIZE = 128

# ── LLM configuration ────────────────────────────────────────────
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4000


def get_deepseek_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY not set. "
            "Export it: set DEEPSEEK_API_KEY=sk-..."
        )
    return key
