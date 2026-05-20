"""
Set pillar geometry parameters via cst.interface.

Parameter names match the CST project's parameter sweep configuration:
"length", "width", "pillar_h".
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def set_pillar_geometry(model3d, L: float, W: float, h: float):
    """Set pillar dimensions and force a geometry rebuild.

    Uses ``StoreDoubleParameter`` (numeric assignment) for clarity, then
    ``RebuildOnParametricChange`` so CST regenerates the model from the
    parameter values. Verifies via RestoreDoubleParameter.
    """
    model3d.StoreDoubleParameter("length", float(L))
    model3d.StoreDoubleParameter("width", float(W))
    model3d.StoreDoubleParameter("pillar_h", float(h))

    # Force CST to actually rebuild the geometry with the new params.
    # RebuildOnParametricChange handles the "results invalidated" case
    # silently when quiet mode is on.
    try:
        model3d.RebuildOnParametricChange(False, False)
    except Exception as e:
        logger.debug(f"RebuildOnParametricChange failed: {e}")
        try:
            model3d.Rebuild()
        except Exception as e2:
            logger.debug(f"Rebuild fallback also failed: {e2}")

    # Verify parameters were written
    try:
        a_L = model3d.RestoreDoubleParameter("length")
        a_W = model3d.RestoreDoubleParameter("width")
        a_h = model3d.RestoreDoubleParameter("pillar_h")
        logger.info(
            f"Geometry: requested L={L}, W={W}, h={h} → "
            f"CST has L={a_L}, W={a_W}, h={a_h}"
        )
        if abs(a_L - L) > 0.5 or abs(a_W - W) > 0.5 or abs(a_h - h) > 0.5:
            raise RuntimeError(
                f"Parameter mismatch — CST did not update geometry: "
                f"requested ({L},{W},{h}), got ({a_L},{a_W},{a_h})"
            )
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Parameter verification failed: {e}")
