"""
Set pillar geometry parameters in a CST project.

Parameter names match the CST project's parameter sweep configuration,
confirmed from the CSV column headers: "length", "width", "pillar_h".
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def set_pillar_geometry(project, L: float, W: float, h: float):
    """Set pillar dimensions and rebuild the geometry.

    Parameters
    ----------
    project : CST COM project object
    L : pillar length in nm
    W : pillar width in nm
    h : pillar height in nm
    """
    project.StoreParameter("length", str(L))
    project.StoreParameter("width", str(W))
    project.StoreParameter("pillar_h", str(h))
    project.Rebuild()
    logger.debug(f"Geometry set: L={L}, W={W}, h={h}")
