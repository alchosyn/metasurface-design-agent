"""
Fresnel propagation and PSF generation for system-level validation.

Reuses the core functions from ml/generate_psf_data.py:
  - fresnel2c: FFT-based Fresnel diffraction
  - build_phase_mask: DH phase + lens phase combination

Adapted to accept arbitrary (dphi, T) values for a given pillar design,
rather than the ideal phase mask.
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


def _get_lens_phase_and_aperture() -> tuple[np.ndarray, np.ndarray, float]:
    """Compute and cache the lens phase and circular aperture."""
    if "lens_phase" not in _CACHE:
        R = ARRAY_SIZE
        D = APERTURE_UM
        f = D / (2 * np.tan(np.arcsin(NA)))

        x = np.linspace(-D / 2, D / 2, R)
        xx, yy = np.meshgrid(x, x)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        ap = (rho <= D / 2).astype(float)
        PN = -2 * np.pi / WAVELENGTH_UM * (
            np.sqrt(xx ** 2 + yy ** 2 + f ** 2) - f
        )
        PN *= ap

        _CACHE["lens_phase"] = PN
        _CACHE["aperture"] = ap
        _CACHE["focal_length"] = f
    return _CACHE["lens_phase"], _CACHE["aperture"], _CACHE["focal_length"]


def build_actual_phase_mask(
    dphi_deg: float,
    T_avg_pct: float,
) -> np.ndarray:
    """Construct the metasurface phase mask for a given pillar design.

    All 1000x1000 pillars share the same (dphi, T), but have different
    rotation angles (PB phase encoding).

    Phase at each position:
        phase[i,j] = rotation_angle[i,j] * (dphi_actual / 180) * (pi / 180)
    Wait — PB phase is: phi_PB = 2 * theta, where theta is rotation angle.
    For a half-wave plate (dphi=180), phi_PB = 2*theta exactly.
    For non-ideal dphi, the PB efficiency is reduced.

    More precisely for PB metasurface:
        E_cross ∝ sin(dphi/2) * exp(i * 2*theta)
    So the actual phase of the cross-polarised component is 2*theta,
    but its amplitude is sin(dphi/2) * sqrt(T).

    Parameters
    ----------
    dphi_deg : actual phase difference of the pillar (degrees)
    T_avg_pct : average transmittance (%)

    Returns
    -------
    TR_phase : (1000, 1000) complex array — combined metasurface field
    """
    rotation_angles = _get_rotation_angles()  # degrees
    PN, ap, f = _get_lens_phase_and_aperture()

    # PB cross-polarisation amplitude and phase
    dphi_rad = np.deg2rad(dphi_deg)
    pb_amplitude = abs(np.sin(dphi_rad / 2)) * np.sqrt(T_avg_pct / 100.0)
    pb_phase = 2.0 * np.deg2rad(rotation_angles)  # geometric phase

    # Combined field
    TR_phase = ap * pb_amplitude * np.exp(1j * pb_phase) * np.exp(1j * PN)

    return TR_phase


def propagate_and_evaluate(
    dphi_deg: float,
    T_avg_pct: float,
    z_positions: list[float] | None = None,
    crop_size: int = PSF_CROP_SIZE,
) -> dict:
    """Full pipeline: build mask → Fresnel propagate → extract PSF metrics.

    Returns
    -------
    dict with keys:
        psfs : (N_z, crop, crop) float32 — normalised PSF images
        z_positions : list of z values (μm)
        focal_length : float (μm)
    """
    if z_positions is None:
        z_positions = FRESNEL_Z_POSITIONS_UM

    TR_phase = build_actual_phase_mask(dphi_deg, T_avg_pct)
    R = TR_phase.shape[0]
    D = APERTURE_UM
    half = crop_size // 2
    cx, cy = R // 2, R // 2

    psfs = np.zeros((len(z_positions), crop_size, crop_size), dtype=np.float32)

    for i, d in enumerate(z_positions):
        prop = fresnel2c(TR_phase, WAVELENGTH_UM, D, D, d)
        intensity = np.abs(prop) ** 2
        patch = intensity[cy - half:cy + half, cx - half:cx + half]
        pmax = patch.max()
        if pmax > 0:
            patch = patch / pmax
        psfs[i] = patch.astype(np.float32)

    return {
        "psfs": psfs,
        "z_positions": z_positions,
        "focal_length": _get_lens_phase_and_aperture()[2],
    }
