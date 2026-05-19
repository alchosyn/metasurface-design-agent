"""
Fresnel propagation and PSF generation for system-level validation.

Reuses the core functions from ml/generate_psf_data.py:
  - fresnel2c: FFT-based Fresnel diffraction
  - build_phase_mask: DH phase + lens phase combination

Adapted to accept arbitrary (dphi, T) values for a given pillar design,
rather than the ideal phase mask.

IMPORTANT: The Pillar_Rotation_Angles_Deg.csv encodes the TOTAL phase
(DH + lens) divided by 2, not just the DH phase. So the PB phase
2*theta already includes both the DH pattern and the lens focusing.
Do NOT add the lens phase separately — that would double-apply it.
"""

from __future__ import annotations

import numpy as np
import scipy.io
from scipy.ndimage import zoom

from agent.config import (
    WAVELENGTH_UM, APERTURE_UM, NA, ARRAY_SIZE,
    PHASEORI_MAT, ROTATION_ANGLES_CSV,
    FRESNEL_Z_POSITIONS_UM, PSF_CROP_SIZE,
)


# ── Core Fresnel propagation (from ml/generate_psf_data.py) ───────
def fresnel2c(field, wavelength, Lx, Ly, d):
    """FFT-based Fresnel diffraction propagation."""
    M, N = field.shape
    u = np.arange(-N // 2 + 1, N // 2 + 1) / Lx
    v = np.arange(-M // 2 + 1, M // 2 + 1) / Ly
    uu, vv = np.meshgrid(u, v)
    c = np.exp(2j * np.pi * d / wavelength)
    H = c * np.exp(-1j * np.pi * d * wavelength * (uu ** 2 + vv ** 2))
    return np.fft.ifft2(np.fft.fft2(field) * np.fft.fftshift(H))


# ── Cached resources ──────────────────────────────────────────────
_CACHE: dict = {}


def _get_rotation_angles() -> np.ndarray:
    """Load and cache the 1000x1000 pillar rotation angle map (degrees)."""
    if "rotation_angles" not in _CACHE:
        angles = np.loadtxt(str(ROTATION_ANGLES_CSV), delimiter=",")
        assert angles.shape == (ARRAY_SIZE, ARRAY_SIZE), (
            f"Expected ({ARRAY_SIZE},{ARRAY_SIZE}), got {angles.shape}"
        )
        _CACHE["rotation_angles"] = angles
    return _CACHE["rotation_angles"]


def _get_aperture_and_focal() -> tuple[np.ndarray, float]:
    """Compute and cache the circular aperture and focal length."""
    if "aperture" not in _CACHE:
        R = ARRAY_SIZE
        D = APERTURE_UM
        f = D / (2 * np.tan(np.arcsin(NA)))

        x = np.linspace(-D / 2, D / 2, R)
        xx, yy = np.meshgrid(x, x)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        ap = (rho <= D / 2).astype(float)

        _CACHE["aperture"] = ap
        _CACHE["focal_length"] = f
    return _CACHE["aperture"], _CACHE["focal_length"]


def build_actual_phase_mask(
    dphi_deg: float,
    T_avg_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the metasurface field components for a given pillar design.

    The rotation angles CSV encodes the TOTAL optical phase (DH + lens)
    divided by 2, so the PB geometric phase 2*theta already includes both
    the double-helix phase pattern AND the focusing lens phase.

    Two orthogonal polarisation components:

    - Cross-polarised: E_cross = sin(dphi/2) * sqrt(T) * exp(i*2*theta)
      Carries the full PB phase (DH + lens). Produces the focused DH-PSF.

    - Co-polarised: E_co = cos(dphi/2) * sqrt(T)
      No geometric phase at all (no DH, no lens). Unfocused plane wave
      through the aperture - negligible intensity at the detector.

    For a perfect half-wave plate (dphi=180 deg), all light goes into
    the cross-polarised component, giving a clean, bright DH-PSF.

    Parameters
    ----------
    dphi_deg : actual phase difference of the pillar (degrees)
    T_avg_pct : average transmittance (%)

    Returns
    -------
    E_cross : (1000, 1000) complex array - cross-polarised field (DH+lens)
    E_co    : (1000, 1000) complex array - co-polarised field (unfocused)
    """
    rotation_angles = _get_rotation_angles()  # degrees
    ap, f = _get_aperture_and_focal()

    dphi_rad = np.deg2rad(dphi_deg)
    T_linear = T_avg_pct / 100.0

    # Cross-polarised: PB phase = 2*theta already includes DH + lens
    cross_amp = abs(np.sin(dphi_rad / 2)) * np.sqrt(T_linear)
    total_phase = 2.0 * np.deg2rad(rotation_angles)  # DH + lens combined
    E_cross = ap * cross_amp * np.exp(1j * total_phase)

    # Co-polarised: no PB phase (no DH, no lens) - flat wavefront
    co_amp = abs(np.cos(dphi_rad / 2)) * np.sqrt(T_linear)
    E_co = ap * co_amp  # uniform amplitude, zero phase

    return E_cross, E_co


def propagate_and_evaluate(
    dphi_deg: float,
    T_avg_pct: float,
    z_positions: list[float] | None = None,
    crop_size: int = PSF_CROP_SIZE,
) -> dict:
    """Full pipeline: build mask -> Fresnel propagate -> extract PSF metrics.

    Returns
    -------
    dict with keys:
        psfs : (N_z, crop, crop) float32 - normalised PSF images
        z_positions : list of z values (um)
        focal_length : float (um)
        pb_efficiency : float - sin^2(dphi/2), fraction of light in DH pattern
    """
    if z_positions is None:
        z_positions = FRESNEL_Z_POSITIONS_UM

    E_cross, E_co = build_actual_phase_mask(dphi_deg, T_avg_pct)
    R = E_cross.shape[0]
    D = APERTURE_UM
    half = crop_size // 2
    cx, cy = R // 2, R // 2

    # Co-polarised is an unfocused plane wave through the aperture.
    # At the detector (cropped to centre), its intensity is negligible
    # compared to the focused cross-polarised component.
    # We skip propagating it to save computation (~2x speedup).
    # For dphi near 180 deg, co_amp is essentially zero anyway.

    psfs = np.zeros((len(z_positions), crop_size, crop_size), dtype=np.float32)

    for i, d in enumerate(z_positions):
        prop_cross = fresnel2c(E_cross, WAVELENGTH_UM, D, D, d)
        intensity = np.abs(prop_cross) ** 2
        patch = intensity[cy - half:cy + half, cx - half:cx + half]
        pmax = patch.max()
        if pmax > 0:
            patch = patch / pmax
        psfs[i] = patch.astype(np.float32)

    # PB efficiency: fraction of transmitted light in the cross-pol DH pattern
    dphi_rad = np.deg2rad(dphi_deg)
    pb_efficiency = np.sin(dphi_rad / 2) ** 2

    return {
        "psfs": psfs,
        "z_positions": z_positions,
        "focal_length": _get_aperture_and_focal()[1],
        "pb_efficiency": float(pb_efficiency),
    }
