"""
Fresnel propagation + PSF quality tools — Tier 2 (medium cost, ~5s).
"""

from __future__ import annotations

from langchain_core.tools import tool

from agent.config import FOM_PHASE_WEIGHT, FOM_TRANS_WEIGHT


def make_fresnel_tool():

    @tool
    def run_fresnel_psf(dphi_deg: float, T_avg_pct: float) -> str:
        """Evaluate the PSF quality of a pillar design via Fresnel propagation.

        Given the pillar's phase difference and average transmittance,
        constructs the full 1000x1000 metasurface phase mask and propagates
        it through 18 defocus positions using Fresnel diffraction.

        Evaluates whether the resulting PSF is a good double-helix pattern
        by measuring rotation angle linearity (R-squared) and lobe contrast.

        Parameters:
            dphi_deg: Phase difference of the pillar in degrees (from CST or surrogate)
            T_avg_pct: Average transmittance in percent

        Returns:
            PSF quality metrics including rotation linearity R-squared,
            rotation angle range, and main lobe contrast ratio.

        Note: Takes ~5 seconds due to FFT-based propagation on a 1000x1000 grid.
        Use this to verify top candidates before declaring convergence.
        """
        from agent.evaluation.fresnel import propagate_and_evaluate
        from agent.evaluation.psf_quality import evaluate_psf_stack

        # Run Fresnel propagation
        prop_result = propagate_and_evaluate(dphi_deg, T_avg_pct)

        # Evaluate PSF quality
        quality = evaluate_psf_stack(
            prop_result["psfs"],
            prop_result["z_positions"],
            prop_result["focal_length"],
        )

        # Format response
        r2 = quality["r_squared"]
        theta_range = quality["theta_range"]
        lobe_ratio = quality["main_lobe_ratio"]
        pb_eff = prop_result.get("pb_efficiency", 1.0)

        if r2 > 0.95:
            verdict = "Excellent DH-PSF quality"
        elif r2 > 0.85:
            verdict = "Good DH-PSF quality"
        elif r2 > 0.70:
            verdict = "Moderate DH-PSF quality - consider optimising further"
        else:
            verdict = "Poor DH-PSF quality - design needs improvement"

        # Phase error and overall efficiency
        phase_err = abs(abs(dphi_deg) - 180.0)
        signal_eff = pb_eff * T_avg_pct / 100.0

        return (
            f"PSF evaluation (dphi={dphi_deg:.1f} deg, T_avg={T_avg_pct:.1f}%):\n"
            f"  Rotation linearity R-squared: {r2:.4f}\n"
            f"  Rotation angle range: {theta_range:.1f} deg\n"
            f"  Main lobe contrast ratio: {lobe_ratio:.1f}\n"
            f"  PB efficiency: {pb_eff:.4f} (sin^2(dphi/2))\n"
            f"  Signal efficiency: {signal_eff:.4f} (PB_eff x T_avg)\n"
            f"  Phase error: {phase_err:.1f} deg\n"
            f"  Verdict: {verdict}\n"
            f"  (Propagated through {len(prop_result['z_positions'])} "
            f"z-positions, f={prop_result['focal_length']:.0f} um)"
        )

    return run_fresnel_psf
