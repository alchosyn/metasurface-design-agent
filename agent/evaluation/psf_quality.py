"""
PSF quality evaluation: rotation angle extraction and linearity analysis.

For a DH-PSF, the two lobes rotate as a function of defocus z.
A good design shows a linear relationship between rotation angle and z.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import center_of_mass


def _double_gaussian_model(params, x, y):
    """Two symmetric Gaussian lobes model.

    params: (x0, y0, sigma, amplitude)
    Lobes at (+x0, +y0) and (-x0, -y0).
    """
    x0, y0, sigma, amp = params
    g1 = amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g2 = amp * np.exp(-((x + x0) ** 2 + (y + y0) ** 2) / (2 * sigma ** 2))
    return g1 + g2


def fit_double_gaussian(psf: np.ndarray) -> dict:
    """Fit a double-Gaussian model to extract lobe positions.

    Parameters
    ----------
    psf : (H, W) float array — single PSF image, normalised

    Returns
    -------
    dict: x0, y0, sigma, amplitude, theta_deg (rotation angle), residual
    """
    H, W = psf.shape
    cy, cx = H / 2, W / 2

    # Create coordinate grids centred at image centre
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    x_grid = x_grid.astype(float) - cx
    y_grid = y_grid.astype(float) - cy

    # Initial guess from centre of mass of top-half and bottom-half
    # Use intensity-weighted approach
    threshold = psf.max() * 0.3
    mask = psf > threshold
    if mask.sum() < 5:
        # Very weak signal
        return {
            "x0": 0, "y0": 0, "sigma": 5, "amplitude": 0,
            "theta_deg": 0, "residual": float("inf"),
        }

    # Find two peaks by splitting the masked region
    coords = np.column_stack(np.where(mask))
    com = center_of_mass(psf * mask)
    cy_com, cx_com = com

    # Initial guess: offset from centre
    weighted_x = np.sum(x_grid * psf * mask) / np.sum(psf * mask)
    x0_init = max(abs(weighted_x), 3.0)
    y0_init = 0.0
    sigma_init = 3.0
    amp_init = psf.max()

    def cost(params):
        model = _double_gaussian_model(params, x_grid, y_grid)
        return np.sum((psf - model) ** 2)

    try:
        result = minimize(
            cost, [x0_init, y0_init, sigma_init, amp_init],
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 0.1, "fatol": 1e-6},
        )
        x0, y0, sigma, amp = result.x
        theta = np.rad2deg(np.arctan2(y0, x0))
        return {
            "x0": float(x0), "y0": float(y0),
            "sigma": float(abs(sigma)),
            "amplitude": float(abs(amp)),
            "theta_deg": float(theta),
            "residual": float(result.fun),
        }
    except Exception:
        return {
            "x0": 0, "y0": 0, "sigma": 5, "amplitude": 0,
            "theta_deg": 0, "residual": float("inf"),
        }


def evaluate_psf_stack(
    psfs: np.ndarray,
    z_positions: list[float],
    focal_length: float,
) -> dict:
    """Evaluate a full PSF z-stack for DH-PSF quality.

    Parameters
    ----------
    psfs : (N_z, H, W) float array
    z_positions : list of propagation distances (μm)
    focal_length : focal length (μm)

    Returns
    -------
    dict with keys:
        thetas : list of rotation angles (degrees)
        z_offsets : list of z offsets from focus (μm)
        r_squared : float — linearity of theta vs z
        theta_range : float — total rotation range (degrees)
        main_lobe_ratio : float — average ratio of peak to background
    """
    z_offsets = [z - focal_length for z in z_positions]
    thetas = []
    lobe_ratios = []

    for i, psf in enumerate(psfs):
        fit = fit_double_gaussian(psf)
        thetas.append(fit["theta_deg"])

        # Main lobe energy ratio
        peak = psf.max()
        mean_bg = np.percentile(psf, 50)
        ratio = peak / max(mean_bg, 1e-8)
        lobe_ratios.append(min(ratio, 100.0))

    thetas = np.array(thetas)
    z_arr = np.array(z_offsets)

    # Unwrap potential ±180° jumps
    for i in range(1, len(thetas)):
        diff = thetas[i] - thetas[i - 1]
        if diff > 90:
            thetas[i:] -= 180
        elif diff < -90:
            thetas[i:] += 180

    # Linear fit: theta = a * z + b
    if len(z_arr) > 2 and np.std(z_arr) > 0:
        coeffs = np.polyfit(z_arr, thetas, 1)
        theta_fit = np.polyval(coeffs, z_arr)
        ss_res = np.sum((thetas - theta_fit) ** 2)
        ss_tot = np.sum((thetas - thetas.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    else:
        r2 = 0.0

    return {
        "thetas": thetas.tolist(),
        "z_offsets": z_offsets,
        "r_squared": r2,
        "theta_range": float(thetas.max() - thetas.min()),
        "main_lobe_ratio": float(np.mean(lobe_ratios)),
    }
