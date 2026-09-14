"""
Randles-CPE equivalent-circuit forward model for electrode-tissue impedance.

Circuit: series resistance Rs, in series with a parallel combination of a
charge-transfer resistance Rct and a constant-phase element (CPE) with
coefficient Q and exponent alpha.

    Z_CPE(f) = 1 / (Q * (j*2*pi*f)^alpha)
    Z(f)     = Rs + (Rct * Z_CPE(f)) / (Rct + Z_CPE(f))

This module is the single source of truth for:
  1. the forward model (impedance from parameters), and
  2. the parameter <-> training-scale transforms.

Phase 2 (model export) and Phase 3 (frontend inference) must reuse the
transform functions defined here (or a byte-identical port of them) so the
inverse-transform used at inference time can never drift from what the
model was trained on.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Frequency grid
# ---------------------------------------------------------------------------
# 20 log-spaced points spanning the standard EIS measurement range for
# electrode-tissue interfaces, 1 Hz to 100 kHz. This exact grid is what the
# "main" and "degenerate" slices are simulated on, and is stored explicitly
# in every dataset file (see generate_dataset.py) so downstream phases never
# need to hardcode it separately.
FREQ_MIN_HZ = 1.0
FREQ_MAX_HZ = 1.0e5
N_FREQ_POINTS = 20
FREQ_GRID_HZ = np.logspace(
    np.log10(FREQ_MIN_HZ), np.log10(FREQ_MAX_HZ), N_FREQ_POINTS
)

# Restricted sub-band used by the "low-frequency-coverage" slice (Task 4.3).
LOW_FREQ_BAND_LO_HZ = 100.0
LOW_FREQ_BAND_HI_HZ = 1.0e4


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
def z_cpe(f_hz: np.ndarray, Q: np.ndarray | float, alpha: np.ndarray | float) -> np.ndarray:
    """Impedance of a constant-phase element at frequency/frequencies f_hz.

    Z_CPE(f) = 1 / (Q * (j*2*pi*f)^alpha)

    Broadcasts over f_hz, Q, alpha (e.g. f_hz shape (F,), Q/alpha shape (N,1)
    gives a (N, F) array of complex impedances).
    """
    omega = 2.0 * np.pi * np.asarray(f_hz, dtype=np.float64)
    j_omega = 1j * omega
    return 1.0 / (Q * j_omega**alpha)


def randles_cpe_impedance(
    f_hz: np.ndarray,
    Rs: np.ndarray | float,
    Rct: np.ndarray | float,
    Q: np.ndarray | float,
    alpha: np.ndarray | float,
) -> np.ndarray:
    """Total Randles-CPE impedance Z(f) = Rs + (Rct * Z_CPE) / (Rct + Z_CPE).

    All of Rs, Rct, Q, alpha may be scalars or arrays that broadcast against
    f_hz. Typical usage for a batch of N parameter sets evaluated on a
    shared F-point frequency grid: pass f_hz with shape (F,) and Rs/Rct/Q/
    alpha with shape (N, 1); the result has shape (N, F).
    """
    z_c = z_cpe(f_hz, Q, alpha)
    z_parallel = (Rct * z_c) / (Rct + z_c)
    return Rs + z_parallel


# ---------------------------------------------------------------------------
# Parameter <-> training-scale transforms (Task 3)
# ---------------------------------------------------------------------------
# Rs, Rct, Q span multiple orders of magnitude -> log10. alpha is bounded in
# (0, 1) -> logit. These transforms MUST be reused identically (not
# re-derived) in Phase 2 and Phase 3 to avoid train/inference drift.


def params_to_transformed(
    Rs: np.ndarray, Rct: np.ndarray, Q: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw-scale (Rs, Rct, Q, alpha) -> training-scale
    (log10 Rs, log10 Rct, log10 Q, logit alpha)."""
    log_Rs = np.log10(Rs)
    log_Rct = np.log10(Rct)
    log_Q = np.log10(Q)
    logit_alpha = np.log(alpha / (1.0 - alpha))
    return log_Rs, log_Rct, log_Q, logit_alpha


def transformed_to_params(
    log_Rs: np.ndarray, log_Rct: np.ndarray, log_Q: np.ndarray, logit_alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of params_to_transformed: training-scale -> raw-scale."""
    Rs = 10.0**log_Rs
    Rct = 10.0**log_Rct
    Q = 10.0**log_Q
    alpha = 1.0 / (1.0 + np.exp(-logit_alpha))
    return Rs, Rct, Q, alpha


# ---------------------------------------------------------------------------
# Magnitude/phase <-> Re/Im helpers (both representations are stored in the
# dataset since both get used downstream)
# ---------------------------------------------------------------------------
def complex_to_mag_phase(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (magnitude [ohm], phase [degrees])."""
    magnitude = np.abs(z)
    phase_deg = np.angle(z, deg=True)
    return magnitude, phase_deg


def mag_phase_to_complex(magnitude: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    phase_rad = np.deg2rad(phase_deg)
    return magnitude * (np.cos(phase_rad) + 1j * np.sin(phase_rad))
