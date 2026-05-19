"""
CST Studio Suite COM connection manager.

CST exposes its automation API through the CSTStudio.Application ProgID.
COM must be initialised per-thread (STA model).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CSTConnection:
    """Manage lifecycle of a CST Studio Suite COM connection.

    Usage
    -----
    >>> cst = CSTConnection("path/to/project.cst")
    >>> cst.project.StoreParameter("length", "310")
    >>> cst.close()
    """

    def __init__(self, project_path: str):
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        self._com_initialised = True

        logger.info("Connecting to CST Studio Suite...")
        self.app = win32com.client.Dispatch("CSTStudio.Application")
        self.project = self.app.OpenFile(project_path)
        logger.info(f"Opened CST project: {project_path}")

    def close(self):
        import pythoncom

        try:
            self.project.Save()
            # Don't quit the app — other projects may be open
        except Exception as e:
            logger.warning(f"Error closing CST project: {e}")
        finally:
            if self._com_initialised:
                pythoncom.CoUninitialize()
                self._com_initialised = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
