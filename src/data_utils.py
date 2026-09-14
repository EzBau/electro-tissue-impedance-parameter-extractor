"""
Shared data-loading utilities for Phase 2 (training, export, verification).

Input feature convention (must match the exported ONNX spec in
PHASE2_NOTES.md exactly): a 40-dim vector, [Re(Z) at freq_grid[0..19],
Im(Z) at freq_grid[0..19]] -- Re block first, then Im block, both in
ascending-frequency order matching forward_model.FREQ_GRID_HZ.

Frequency points outside a spectrum's measured band (only relevant for the
low-freq-coverage slice, stored as NaN in the Phase 1 dataset) are
zero-imputed: a "not measured" point contributes Re=Im=0 raw ohms to the
input vector. This is a deliberate, documented design choice (see
PHASE2_NOTES.md) -- it is not a claim that 0 ohms was actually measured.
"""

from __future__ import annotations

import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SLICE_MAIN, SLICE_DEGENERATE, SLICE_LOW_FREQ = 0, 1, 2
SLICE_NAMES = {SLICE_MAIN: "main", SLICE_DEGENERATE: "degenerate", SLICE_LOW_FREQ: "low_freq_coverage"}

TARGET_KEYS = ("log_Rs", "log_Rct", "log_Q", "logit_alpha")
RAW_PARAM_KEYS = ("Rs", "Rct", "Q", "alpha")


def build_features(re_ohm: np.ndarray, im_ohm: np.ndarray) -> np.ndarray:
    """(N, 20) Re, (N, 20) Im, possibly containing NaN -> (N, 40) float32,
    NaN zero-imputed, Re block first then Im block."""
    re_filled = np.nan_to_num(re_ohm, nan=0.0)
    im_filled = np.nan_to_num(im_ohm, nan=0.0)
    return np.concatenate([re_filled, im_filled], axis=1).astype(np.float32)


def load_split(split_name: str) -> dict:
    """Load one of train/val/test.npz and return a dict with:
    X (N,40) float32 features, Y (N,4) float32 transformed targets,
    params_raw (N,4) float32 [Rs,Rct,Q,alpha], slice_label (N,) int8.
    """
    path = os.path.join(DATA_DIR, f"{split_name}.npz")
    d = np.load(path)
    X = build_features(d["re_ohm"], d["im_ohm"])
    Y = np.stack([d[k] for k in TARGET_KEYS], axis=1).astype(np.float32)
    params_raw = np.stack([d[k] for k in RAW_PARAM_KEYS], axis=1).astype(np.float32)
    return dict(
        X=X,
        Y=Y,
        params_raw=params_raw,
        slice_label=d["slice_label"],
        freq_grid_hz=d["freq_grid_hz"],
    )


def compute_feature_stats(X_train: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over the training set only. std is floored at
    `eps` to avoid division by zero (safety only; no training feature is
    expected to be exactly constant)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)
