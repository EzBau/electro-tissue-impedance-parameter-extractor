"""
Phase 2 Task 6: run the trained model on the same 3 hand-verified cases used
in Phase 1's forward-model verification (verify_forward_model.py), and
report predicted vs. ground-truth parameters. This is a sanity check that
the trained network behaves reasonably on cases we independently understand
-- it is NOT a substitute for Task 3's full per-parameter/per-slice metrics.

Usage: python verify_phase2_handchecks.py
Output: eval/phase2_handcheck_report.txt
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data_utils import build_features  # noqa: E402
from forward_model import FREQ_GRID_HZ, randles_cpe_impedance  # noqa: E402
import generate_dataset as gen  # noqa: E402
from model_def import MLP, ExportWrapper  # noqa: E402
from train_model import HIDDEN_SIZES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
EVAL_DIR = os.path.join(ROOT, "eval")

PARAM_NAMES = ["Rs", "Rct", "Q", "alpha"]
UNITS = ["ohm", "ohm", "S*s^alpha", "(dimensionless)"]


def load_export_model() -> ExportWrapper:
    state_dict = torch.load(os.path.join(MODELS_DIR, "mlp_state_dict.pt"), weights_only=True)
    mlp = MLP(hidden_sizes=HIDDEN_SIZES)
    mlp.load_state_dict(state_dict)
    mlp.eval()
    stats = np.load(os.path.join(MODELS_DIR, "norm_stats.npz"))
    model = ExportWrapper(mlp, torch.from_numpy(stats["feature_mean"]), torch.from_numpy(stats["feature_std"]))
    model.eval()
    return model


def predict(model: ExportWrapper, Rs: float, Rct: float, Q: float, alpha: float) -> np.ndarray:
    Z = randles_cpe_impedance(FREQ_GRID_HZ, Rs, Rct, Q, alpha)
    re = Z.real[None, :]
    im = Z.imag[None, :]
    X = build_features(re, im)  # (1,40), no NaNs for these full-band cases
    with torch.no_grad():
        out = model(torch.from_numpy(X)).numpy()[0]
    return out


def report_case(lines: list, title: str, ground_truth: dict, predicted: np.ndarray) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Parameter | Unit | Ground truth | Predicted | Abs error | Rel error |")
    lines.append("|---|---|---|---|---|---|")
    for i, (name, unit) in enumerate(zip(PARAM_NAMES, UNITS)):
        gt = ground_truth[name]
        pred = float(predicted[i])
        abs_err = abs(gt - pred)
        rel_err = abs_err / max(abs(gt), 1e-30)
        lines.append(f"| {name} | {unit} | {gt:.6g} | {pred:.6g} | {abs_err:.4g} | {rel_err:.4%} |")
    lines.append("")


def main() -> None:
    model = load_export_model()
    lines = ["# Phase 2 Task 6: Trained Model on Phase 1 Hand-Checked Cases", ""]

    # Case 1: alpha = 1.0, simple parallel-RC (Phase 1 verification Case 1)
    gt1 = dict(Rs=2000.0, Rct=50000.0, Q=2.0e-7, alpha=1.0)
    # alpha=1.0 is out of this model's training range [0.5, 1.0) -- Phase 1
    # samples alpha strictly < 1.0 (numpy uniform high-exclusive), so we use
    # alpha=0.999999 here (indistinguishable physically, avoids evaluating
    # exactly on the training-range boundary/outside it).
    gt1_eval = dict(gt1, alpha=0.999999)
    pred1 = predict(model, gt1_eval["Rs"], gt1_eval["Rct"], gt1_eval["Q"], gt1_eval["alpha"])
    report_case(
        lines,
        "Case 1 -- alpha ~= 1.0, parallel RC (Rs=2000, Rct=50000, Q=2e-7, alpha=0.999999)",
        gt1_eval,
        pred1,
    )

    # Case 2: alpha = 0.5, Warburg-like (Phase 1 verification Case 2).
    # Original hand-check used Rct->infinity (1e18), far outside the
    # training range [1e4,1e7] -- use Rct at the top of the trained range
    # instead so the prediction is a meaningful in-distribution check.
    gt2 = dict(Rs=5000.0, Rct=1.0e7, Q=3.0e-8, alpha=0.5)
    pred2 = predict(model, gt2["Rs"], gt2["Rct"], gt2["Q"], gt2["alpha"])
    report_case(
        lines,
        "Case 2 -- alpha = 0.5, Warburg-like (Rs=5000, Rct=1e7, Q=3e-8, alpha=0.5)",
        gt2,
        pred2,
    )

    # Case 3: the same degenerate-slice sample used in Phase 1 verification
    # (rng seed 4242, first draw from sample_degenerate_params).
    rng = np.random.default_rng(4242)
    params = gen.sample_degenerate_params(rng, 1)
    gt3 = {k: float(params[k][0]) for k in ("Rs", "Rct", "Q", "alpha")}
    pred3 = predict(model, gt3["Rs"], gt3["Rct"], gt3["Q"], gt3["alpha"])
    report_case(lines, "Case 3 -- degenerate-slice sample (same as Phase 1 verification Case 3)", gt3, pred3)

    lines.append(
        "Note: this is a sanity check on 3 specific points, not a substitute for "
        "the full per-parameter/per-slice metrics in eval/phase2_eval_report.md "
        "(Task 3). Case 3 in particular is expected to show larger error on Rct "
        "and/or alpha, consistent with the degenerate-slice non-identifiability "
        "documented there."
    )

    report = "\n".join(lines)
    os.makedirs(EVAL_DIR, exist_ok=True)
    out_path = os.path.join(EVAL_DIR, "phase2_handcheck_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
