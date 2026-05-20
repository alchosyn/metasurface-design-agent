"""
CST Studio Suite connection manager using the official cst.interface API.

This replaces the old win32com COM approach because:
  - win32com late-binding fights with CST 2025's IDispatch
  - methods get mis-identified as properties (Rebuild, FDSolver, Save, ...)
  - cross-thread COM access loses method resolution after a few calls
  - StoreParameter silently no-ops in some COM states

cst.interface is the supported Python automation API: native bindings,
no COM, works from any thread.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Add CST's Python library to sys.path so `import cst.interface` and
# `import cst.results` work. Override via env var if installed elsewhere.
CST_PYTHON_LIB = os.environ.get(
    "CST_PYTHON_LIB", r"D:\CST\AMD64\python_cst_libraries"
)
if os.path.isdir(CST_PYTHON_LIB) and CST_PYTHON_LIB not in sys.path:
    sys.path.insert(0, CST_PYTHON_LIB)


class CSTConnection:
    """Manage a CST Studio Suite session via cst.interface.

    Connects to an already-running CST instance if one exists, otherwise
    starts a new one. Opens the project and keeps a handle to the
    model3d object for the active simulation domain.

    Usage
    -----
    >>> cst = CSTConnection("path/to/project.cst")
    >>> cst.model3d.StoreParameter("length", "310")
    >>> cst.close()
    """

    def __init__(self, project_path: str):
        from cst.interface import DesignEnvironment

        logger.info("Connecting to CST Studio Suite...")
        try:
            self.de = DesignEnvironment.connect_to_any()
            logger.info(f"Connected to existing CST (pid={self.de.pid})")
        except Exception:
            self.de = DesignEnvironment.new()
            logger.info(f"Started new CST (pid={self.de.pid})")

        # NOTE: Do NOT enable quiet mode here. The "Quiet/Scripting mode
        # is active" notification keeps re-appearing and blocks the user.
        # Instead, callers should use ``with cst.quiet_mode_enabled():``
        # around the actual simulation steps.

        # Open or attach to the project
        try:
            self.project = self.de.open_project(project_path)
        except Exception:
            # Already open — fetch the handle
            self.project = self.de.get_open_project(project_path)
        self.model3d = self.project.model3d
        logger.info(f"Opened CST project: {project_path}")

    def close(self):
        try:
            self.project.save()
        except Exception as e:
            logger.debug(f"Final save skipped: {e}")
        # Don't close the project — user may want to inspect it.
        # Don't close the DE — user may have other projects open.

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
