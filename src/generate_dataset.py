"""
Phase 1 synthetic training-data generator for the Randles-CPE electrode-
tissue impedance parameter extractor.

Usage:
    python generate_dataset.py

Produces data/train.npz, data/val.npz, data/test.npz. See PHASE1_NOTES.md
for parameter-range citations, noise-model rationale, and slice
composition. See verification/verify_forward_model.py for the required
hand-checks of the forward model itself.

Dependencies: numpy only.
"""

from __future__ import annotations

import os

import numpy as np

from forward_model import (
    FREQ_GRID_HZ,
    LOW_FREQ_BAND_HI_HZ,
    LOW_FREQ_BAND_LO_HZ,
    complex_to_mag_phase,
    params_to_transformed,
    randles_cpe_impedance,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG_SEED = 20260910
RNG = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# Parameter ranges (Task 1). See PHASE1_NOTES.md for citations / NOT
# VERIFIED flags. Numbers here must match what's written in that file.
# ---------------------------------------------------------------------------
RS_MIN_OHM, RS_MAX_OHM = 1.0e3, 5.0e4          # 1 k ohm - 50 k ohm, uniform
RCT_MIN_OHM, RCT_MAX_OHM = 1.0e4, 1.0e7        # 10 k ohm - 10 M ohm, log-uniform
Q_MIN, Q_MAX = 1.0e-9, 1.0e-6                  # S*s^alpha, log-uniform
ALPHA_MIN, ALPHA_MAX = 0.5, 1.0                # uniform

# Degenerate-slice extreme-end bands: sample Rct log-uniformly within the
# lowest/highest decade of the full Rct range (Task 4.2).
RCT_LOW_BAND = (RCT_MIN_OHM, RCT_MIN_OHM * 10.0)
RCT_HIGH_BAND = (RCT_MAX_OHM / 10.0, RCT_MAX_OHM)

# Noise ranges (Task 5). Both sampled per-spectrum, not fixed.
NOISE_MAG_PCT_MIN, NOISE_MAG_PCT_MAX = 0.5, 8.0    # % of magnitude, std dev
NOISE_PHASE_DEG_MIN, NOISE_PHASE_DEG_MAX = 0.1, 3.0  # degrees, std dev
# NOT VERIFIED: these two ranges are engineering judgment calls, not derived
# from a published instrument noise-floor spec. This noise model captures
# generic measurement noise only -- it does NOT model real instrument
# artifacts such as 50/60 Hz line-frequency pickup or slow electrode drift
# over the course of a scan. That gap is intentionally deferred to the
# Phase 4 limitations writeup (Task 5 instruction).

# ---------------------------------------------------------------------------
# Slice composition (Task 4)
# ---------------------------------------------------------------------------
FRAC_MAIN = 0.70
FRAC_DEGENERATE = 0.20
FRAC_LOW_FREQ = 0.10

# Total number of spectra to generate. Not specified by the task; chosen as
# a size that trains a small regression model comfortably while generating
# in well under a minute on a single core.
N_TOTAL = 100_000

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

SLICE_MAIN, SLICE_DEGENERATE, SLICE_LOW_FREQ = 0, 1, 2
SLICE_NAMES = {SLICE_MAIN: "main", SLICE_DEGENERATE: "degenerate", SLICE_LOW_FREQ: "low_freq_coverage"}


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, size: int) -> np.ndarray:
    return 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), size=size)


def sample_main_params(rng: np.random.Generator, n: int) -> dict:
    Rs = rng.uniform(RS_MIN_OHM, RS_MAX_OHM, size=n)
    Rct = _log_uniform(rng, RCT_MIN_OHM, RCT_MAX_OHM, n)
    Q = _log_uniform(rng, Q_MIN, Q_MAX, n)
    alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX, size=n)
    return dict(Rs=Rs, Rct=Rct, Q=Q, alpha=alpha)


def sample_degenerate_params(rng: np.random.Generator, n: int) -> dict:
    """Oversample Rct at the extreme low or high end of its range so the
    parallel Rct/CPE branch is dominated by just one element, producing
    near-indistinguishable spectra (Rct becomes unidentifiable when
    Rct >> |Z_CPE| across the whole band; Q/alpha become unidentifiable
    when Rct << |Z_CPE|). Rs, Q, alpha are otherwise drawn from their full
    main-slice distributions.
    """
    Rs = rng.uniform(RS_MIN_OHM, RS_MAX_OHM, size=n)
    Q = _log_uniform(rng, Q_MIN, Q_MAX, n)
    alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX, size=n)

    use_high_band = rng.uniform(size=n) < 0.5
    Rct = np.empty(n, dtype=np.float64)
    n_low = int(np.sum(~use_high_band))
    n_high = n - n_low
    Rct[~use_high_band] = _log_uniform(rng, *RCT_LOW_BAND, n_low)
    Rct[use_high_band] = _log_uniform(rng, *RCT_HIGH_BAND, n_high)
    return dict(Rs=Rs, Rct=Rct, Q=Q, alpha=alpha)


def sample_low_freq_params(rng: np.random.Generator, n: int) -> dict:
    """Same parameter distribution as the main slice; what differs is the
    simulated frequency band (applied later), which restricts the data to
    100 Hz-10 kHz so Rs (whose signature sits at the highest frequencies)
    is poorly constrained."""
    return sample_main_params(rng, n)


def simulate_spectra(params: dict, freq_mask: np.ndarray, rng: np.random.Generator) -> dict:
    """Compute noisy magnitude/phase/Re/Im spectra for a batch of parameter
    sets, given a boolean (N, F) mask of which frequency points are
    "measured" for each spectrum (unmeasured points are stored as NaN).
    """
    n = len(params["Rs"])
    f = FREQ_GRID_HZ  # shape (F,)

    Z_clean = randles_cpe_impedance(
        f,
        params["Rs"][:, None],
        params["Rct"][:, None],
        params["Q"][:, None],
        params["alpha"][:, None],
    )  # shape (N, F), complex

    mag_clean, phase_clean = complex_to_mag_phase(Z_clean)

    noise_mag_pct = rng.uniform(NOISE_MAG_PCT_MIN, NOISE_MAG_PCT_MAX, size=n)
    noise_phase_deg = rng.uniform(NOISE_PHASE_DEG_MIN, NOISE_PHASE_DEG_MAX, size=n)

    mag_noise = rng.normal(0.0, 1.0, size=mag_clean.shape) * (noise_mag_pct[:, None] / 100.0)
    phase_noise = rng.normal(0.0, 1.0, size=phase_clean.shape) * noise_phase_deg[:, None]

    mag_noisy = mag_clean * (1.0 + mag_noise)
    phase_noisy = phase_clean + phase_noise

    # Re/Im are derived from the *same* noisy magnitude/phase, so the two
    # representations stay mutually consistent.
    phase_noisy_rad = np.deg2rad(phase_noisy)
    re_noisy = mag_noisy * np.cos(phase_noisy_rad)
    im_noisy = mag_noisy * np.sin(phase_noisy_rad)

    mag_noisy = np.where(freq_mask, mag_noisy, np.nan)
    phase_noisy = np.where(freq_mask, phase_noisy, np.nan)
    re_noisy = np.where(freq_mask, re_noisy, np.nan)
    im_noisy = np.where(freq_mask, im_noisy, np.nan)

    return dict(
        magnitude_ohm=mag_noisy.astype(np.float32),
        phase_deg=phase_noisy.astype(np.float32),
        re_ohm=re_noisy.astype(np.float32),
        im_ohm=im_noisy.astype(np.float32),
        noise_mag_pct=noise_mag_pct.astype(np.float32),
        noise_phase_deg=noise_phase_deg.astype(np.float32),
    )


def build_slice(slice_id: int, n: int, rng: np.random.Generator) -> dict:
    if slice_id == SLICE_MAIN:
        params = sample_main_params(rng, n)
        band_lo = np.full(n, FREQ_GRID_HZ[0])
        band_hi = np.full(n, FREQ_GRID_HZ[-1])
    elif slice_id == SLICE_DEGENERATE:
        params = sample_degenerate_params(rng, n)
        band_lo = np.full(n, FREQ_GRID_HZ[0])
        band_hi = np.full(n, FREQ_GRID_HZ[-1])
    elif slice_id == SLICE_LOW_FREQ:
        params = sample_low_freq_params(rng, n)
        band_lo = np.full(n, LOW_FREQ_BAND_LO_HZ)
        band_hi = np.full(n, LOW_FREQ_BAND_HI_HZ)
    else:
        raise ValueError(slice_id)

    freq_mask = (FREQ_GRID_HZ[None, :] >= band_lo[:, None]) & (
        FREQ_GRID_HZ[None, :] <= band_hi[:, None]
    )
    spectra = simulate_spectra(params, freq_mask, rng)

    log_Rs, log_Rct, log_Q, logit_alpha = params_to_transformed(
        params["Rs"], params["Rct"], params["Q"], params["alpha"]
    )

    return dict(
        Rs=params["Rs"].astype(np.float32),
        Rct=params["Rct"].astype(np.float32),
        Q=params["Q"].astype(np.float32),
        alpha=params["alpha"].astype(np.float32),
        log_Rs=log_Rs.astype(np.float32),
        log_Rct=log_Rct.astype(np.float32),
        log_Q=log_Q.astype(np.float32),
        logit_alpha=logit_alpha.astype(np.float32),
        slice_label=np.full(n, slice_id, dtype=np.int8),
        degenerate_flag=np.full(n, slice_id == SLICE_DEGENERATE, dtype=bool),
        low_freq_flag=np.full(n, slice_id == SLICE_LOW_FREQ, dtype=bool),
        freq_band_lo_hz=band_lo.astype(np.float32),
        freq_band_hi_hz=band_hi.astype(np.float32),
        valid_mask=freq_mask,
        **spectra,
    )


def _split_indices(n: int, rng: np.random.Generator) -> dict:
    """Shuffle n indices and cut into train/val/test at the target
    fractions, returning index arrays."""
    idx = rng.permutation(n)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    return dict(
        train=idx[:n_train],
        val=idx[n_train : n_train + n_val],
        test=idx[n_train + n_val :],
    )


def _concat(dicts: list[dict]) -> dict:
    keys = dicts[0].keys()
    return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in keys}


def main() -> None:
    n_main = int(round(N_TOTAL * FRAC_MAIN))
    n_degenerate = int(round(N_TOTAL * FRAC_DEGENERATE))
    n_low_freq = N_TOTAL - n_main - n_degenerate  # remainder absorbs rounding

    slices = {
        SLICE_MAIN: build_slice(SLICE_MAIN, n_main, RNG),
        SLICE_DEGENERATE: build_slice(SLICE_DEGENERATE, n_degenerate, RNG),
        SLICE_LOW_FREQ: build_slice(SLICE_LOW_FREQ, n_low_freq, RNG),
    }

    # Split each slice independently so main/degenerate/low-freq are
    # proportionally represented in train/val/test (Task 6 requirement),
    # then concatenate and shuffle each split.
    split_data = {"train": [], "val": [], "test": []}
    for slice_id, data in slices.items():
        n = len(data["Rs"])
        idx_by_split = _split_indices(n, RNG)
        for split_name, idx in idx_by_split.items():
            split_data[split_name].append({k: v[idx] for k, v in data.items()})

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)

    for split_name, parts in split_data.items():
        merged = _concat(parts)
        shuffle_idx = RNG.permutation(len(merged["Rs"]))
        merged = {k: v[shuffle_idx] for k, v in merged.items()}
        merged["freq_grid_hz"] = FREQ_GRID_HZ.astype(np.float64)

        path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez_compressed(path, **merged)

        counts = {
            SLICE_NAMES[sid]: int(np.sum(merged["slice_label"] == sid))
            for sid in SLICE_NAMES
        }
        print(f"{split_name}: {len(merged['Rs'])} spectra -> {path}")
        print(f"  slice counts: {counts}")


if __name__ == "__main__":
    main()
