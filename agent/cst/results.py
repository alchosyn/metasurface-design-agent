"""
Read S-parameter results using CST 2025's official Python API (cst.results).

CST 2025 dropped the old file-based result API (.sig files), so COM access
via ``Result0D`` / ``Result1DComplex`` no longer works. Instead we use the
bundled ``cst.results.ProjectFile``, which reads results straight from the
.cst project file and does NOT need CST running.

Tree paths used (from the result tree exploration):
  - 1D Results\\S-Parameters\\SZmax(1),Zmin(1)  → TE transmission (complex)
  - 1D Results\\S-Parameters\\SZmax(2),Zmin(2)  → TM transmission (complex)
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def extract_s_parameters(project_path: str, target_freq_thz: float) -> dict:
    """Read TE/TM S-parameters at the target frequency via cst.results.

    Parameters
    ----------
    project_path : str
        Absolute path to the .cst project file.
    target_freq_thz : float
        Target frequency in THz (e.g. c/632nm ≈ 474.4 THz).

    Returns
    -------
    dict with keys: TE_mag_dB, TE_phase, TM_mag_dB, TM_phase
    """
    # Import inside the function — sys.path is set by connection.py
    from cst.results import ProjectFile

    # cst.results prints "You are working in interactive mode." to stdout
    # on construction — silence it.
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        pf = ProjectFile(project_path, allow_interactive=True)
    threed = pf.get_3d()

    def _read(path: str) -> tuple[float, float]:
        item = threed.get_result_item(path)
        xs = item.get_xdata()
        ys = item.get_ydata()
        best_i = min(range(len(xs)), key=lambda i: abs(xs[i] - target_freq_thz))
        val = ys[best_i]
        mag_db = 10 * math.log10(max(abs(val) ** 2, 1e-20))
        phase_deg = math.degrees(math.atan2(val.imag, val.real))
        return mag_db, phase_deg

    te_mag, te_phase = _read("1D Results\\S-Parameters\\SZmax(1),Zmin(1)")
    tm_mag, tm_phase = _read("1D Results\\S-Parameters\\SZmax(2),Zmin(2)")

    result = {
        "TE_mag_dB": te_mag,
        "TE_phase": te_phase,
        "TM_mag_dB": tm_mag,
        "TM_phase": tm_phase,
    }
    logger.debug(f"S-params: {result}")
    return result


def convert_s_params(sparam: dict) -> dict:
    """Convert raw S-parameters to physical quantities (transmittance, phase diff)."""
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
