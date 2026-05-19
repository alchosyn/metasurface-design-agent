"""
PSF quality evaluation: rotation angle extraction and linearity analysis.

For a DH-PSF, the two lobes rotate as a function of defocus z.
A good design shows a linear relationship between rotation angle and z.

Primary method: direct peak-finding (robust, fast).
Fallback: double-Gaussian fitting (slower, less reliable).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


def extract_lobe_angle(psf: np.ndarray) -> dict:
    """Extract DH-PSF rotation angle by finding the two brightest off-centre peaks.

    More robust than parametric fitting — directly locates the two lobes
    via local maximum detection and computes the angle of the line
    connecting them.

    Parameters
    ----------
    psf : (H, W) float array - single PSF image, normalised to [0, 1]

    Returns
    -------
    dict with keys:
        theta_deg : float - rotation angle in [0, 180) degrees
        success : bool - whether two valid lobes were found
        separation : float - distance between the two lobes (pixels)
        p1, p2 : tuple - (x, y) positions of the two lobes relative to centre
    """
    H, W = psf.shape
    cy, cx = H / 2.0, W / 2.0

    threshold = psf.max() * 0.25
    if threshold < 1e-10:
        return {"theta_deg": 0.0, "success": False, "separation": 0.0,
                "p1": (0.0, 0.0), "p2": (0.0, 0.0)}

    # Find local maxima above threshold
    local_max = (maximum_filter(psf, size=11) == psf) & (psf > threshold)
    peak_ys, peak_xs = np.where(local_max)

    if len(peak_ys) < 2:
        return {"theta_deg": 0.0, "success": False, "separation": 0.0,
                "p1": (0.0, 0.0), "p2": (0.0, 0.0)}

    # Distance from centre for each peak
    dists = np.sqrt((peak_xs - cx) ** 2 + (peak_ys - cy) ** 2)
    intensities = psf[peak_ys, peak_xs]

    # Prefer off-centre peaks (the DH lobes are away from centre)
    off_center = dists > 5
    if off_center.sum() >= 2:
        oc_idx = np.where(off_center)[0]
        oc_int = intensities[oc_idx]
        top2 = oc_idx[np.argsort(-oc_int)[:2]]
    elif off_center.sum() == 1 and len(peak_ys) >= 2:
        # One off-centre peak + pick the second-brightest overall
        oc_idx = np.where(off_center)[0][0]
        all_sorted = np.argsort(-intensities)
        second = all_sorted[0] if all_sorted[0] != oc_idx else all_sorted[1]
        top2 = np.array([oc_idx, second])
    else:
        # All peaks near centre — take brightest two anyway
        top2 = np.argsort(-intensities)[:2]

    p1x = float(peak_xs[top2[0]] - cx)
    p1y = float(peak_ys[top2[0]] - cy)
    p2x = float(peak_xs[top2[1]] - cx)
    p2y = float(peak_ys[top2[1]] - cy)

    # Subpixel refinement: intensity-weighted centroid in 5x5 region
    for idx, (px, py) in enumerate([(peak_xs[top2[0]], peak_ys[top2[0]]),
                                     (peak_xs[top2[1]], peak_ys[top2[1]])]):
        y_lo = max(0, py - 2)
        y_hi = min(H, py + 3)
        x_lo = max(0, px - 2)
        x_hi = min(W, px + 3)
        roi = psf[y_lo:y_hi, x_lo:x_hi]
        if roi.sum() > 0:
            ry, rx = np.indices(roi.shape)
            refined_y = float(np.average(ry, weights=roi)) + y_lo - cy
            refined_x = float(np.average(rx, weights=roi)) + x_lo - cx
            if idx == 0:
                p1x, p1y = refined_x, refined_y
            else:
                p2x, p2y = refined_x, refined_y

    # Angle of line from lobe2 to lobe1
    dx = p1x - p2x
    dy = p1y - p2y
    theta = np.rad2deg(np.arctan2(dy, dx))

    # Normalise to [0, 180) to remove lobe-labeling ambiguity
    theta = theta % 180.0

    separation = np.sqrt(dx ** 2 + dy ** 2)

    return {
        "theta_deg": float(theta),
        "success": True,
        "separation": float(separation),
        "p1": (p1x, p1y),
        "p2": (p2x, p2y),
    }


def evaluate_psf_stack(
    psfs: np.ndarray,
    z_positions: list[float],
    focal_length: float,
) -> dict:
    """Evaluate a full PSF z-stack for DH-PSF quality.

    Extracts rotation angle at each z position using peak-finding,
    then measures the linearity of the theta-vs-z relationship.

    Parameters
    ----------
    psfs : (N_z, H, W) float array
    z_positions : list of propagation distances (um)
    focal_length : focal length (um)

    Returns
    -------
    dict with keys:
        thetas : list of rotation angles (degrees, unwrapped)
        z_offsets : list of z offsets from focus (um)
        r_squared : float - linearity of theta vs z (higher = better DH-PSF)
        theta_range : float - total rotation range (degrees)
        main_lobe_ratio : float - average ratio of peak to background
        mean_separation : float - average distance between lobes (pixels)
    """
    z_offsets = [z - focal_length for z in z_positions]
    thetas = []
    lobe_ratios = []
    separations = []
    n_success = 0

    for psf in psfs:
        result = extract_lobe_angle(psf)
        thetas.append(result["theta_deg"])
        separations.append(result["separation"])
        if result["success"]:
            n_success += 1

        # Main lobe energy ratio (peak to median background)
        peak = psf.max()
        mean_bg = np.percentile(psf, 50)
        ratio = peak / max(mean_bg, 1e-8)
        lobe_ratios.append(min(ratio, 100.0))

    thetas = np.array(thetas)
    z_arr = np.array(z_offsets)

    # Unwrap: angles are in [0, 180), detect jumps near the boundary
    # A jump > 90 degrees between consecutive frames indicates wrapping
    for i in range(1, len(thetas)):
        diff = thetas[i] - thetas[i - 1]
        if diff > 90:
            thetas[i:] -= 180
        elif diff < -90:
            thetas[i:] += 180

    # Linear fit: theta = a * z + b
    if len(z_arr) > 2 and np.std(z_arr) > 0 and n_success >= 3:
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
        "mean_separation": float(np.mean(separations)),
    }
