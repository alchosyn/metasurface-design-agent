"""
Run the CST frequency domain solver via cst.interface.

The project is pre-configured for Floquet port excitation (periodic BCs)
at 632 nm. Solver settings are stored in the .cst file.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def run_frequency_solver(model3d) -> float:
    """Run the frequency domain solver. Blocks until completion.

    Returns
    -------
    elapsed : float — wall-clock time in seconds
    """
    t0 = time.time()
    # run_solver is synchronous (blocks until done) and uses the
    # currently active solver, which is configured in the .cst project
    # as the frequency-domain Floquet solver.
    model3d.run_solver()
    elapsed = time.time() - t0
    logger.info(f"FD solver completed in {elapsed:.1f} s")
    return elapsed
