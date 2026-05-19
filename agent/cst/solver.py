"""
Run the CST frequency domain solver.

The project is pre-configured for Floquet port excitation (periodic BCs)
at 632 nm. Solver settings are saved in the .cst file.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def run_frequency_solver(project) -> float:
    """Run the frequency domain solver. Blocks until completion.

    Returns
    -------
    elapsed : float — wall-clock time in seconds
    """
    t0 = time.time()
    solver = project.FDSolver()
    solver.Start()
    elapsed = time.time() - t0
    logger.info(f"FD solver completed in {elapsed:.1f} s")
    return elapsed
