"""
Extract S-parameter results from a solved CST project.

Result tree paths match the CSV column headers from result_navigator.csv:
  - SZmax(1) → TE mode
  - SZmax(2) → TM mode
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def extract_s_parameters(project) -> dict:
    """Extract TE/TM S-parameter magnitudes and phases.

    Returns
    -------
    dict with keys: TE_mag_dB, TE_phase, TM_mag_dB, TM_phase
    """
    tree = project.ResultTree()

    te_mag = float(tree.GetResultData(
        "Tables\\0D Results\\SZmax(1),Zmin(1)_0D", "yAtX"
    ))
    te_phase = float(tree.GetResultData(
        "Tables\\0D Results\\SZmax(1),Zmin(1)_0D", "yAtX_1"
    ))
    tm_mag = float(tree.GetResultData(
        "Tables\\0D Results\\SZmax(2),Zmin(2)_0D", "yAtX"
    ))
    tm_phase = float(tree.GetResultData(
        "Tables\\0D Results\\SZmax(2),Zmin(2)_0D", "yAtX_1"
    ))

    result = {
        "TE_mag_dB": te_mag,
        "TE_phase": te_phase,
        "TM_mag_dB": tm_mag,
        "TM_phase": tm_phase,
    }
    logger.debug(f"S-params: {result}")
    return result


def convert_s_params(sparam: dict) -> dict:
    """Convert raw S-parameters to physical quantities.

    Applies the same formulas as surrogate_optimization.py lines 126-131.
    """
    T_TE = 10 ** (sparam["TE_mag_dB"] / 10.0) * 100.0
    T_TM = 10 ** (sparam["TM_mag_dB"] / 10.0) * 100.0

    raw_diff = sparam["TE_phase"] - sparam["TM_phase"]
    dphi = ((raw_diff + 180.0) % 360.0) - 180.0

    return {
        "dphi_deg": dphi,
        "T_TE_pct": T_TE,
        "T_TM_pct": T_TM,
        "T_avg_pct": (T_TE + T_TM) / 2.0,
        "phase_err_deg": abs(abs(dphi) - 180.0),
    }
