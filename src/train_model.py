"""
Phase 2 training script.

Trains the MLP defined in model_def.py on the Phase 1 synthetic dataset,
with early stopping on validation loss, then reports per-parameter,
per-slice R^2 and MAE (raw scale) on train/val/test.

Usage: python train_model.py
Outputs:
  models/mlp_state_dict.pt   - trained weights (native PyTorch format)
  models/norm_stats.npz      - train-set feature mean/std (40,) each
  eval/phase2_eval_report.md - per-parameter, per-slice metrics
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import SLICE_NAMES, compute_feature_stats, load_split  # noqa: E402
from forward_model import transformed_to_params  # noqa: E402
from model_def import MLP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
EVAL_DIR = os.path.join(ROOT, "eval")

SEED = 20260910
BATCH_SIZE = 512
MAX_EPOCHS = 800
PATIENCE = 40
LR = 2e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZES = (128, 128, 64)  # tuned up from the 64/64/32 starting point (Task 2 invites tuning)

TARGET_NAMES = ["Rs", "Rct", "Q", "alpha"]
TARGET_UNITS = ["ohm", "ohm", "S*s^alpha", "(dimensionless)"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def train() -> tuple[MLP, np.ndarray, np.ndarray]:
    set_seed(SEED)

    train_d = load_split("train")
    val_d = load_split("val")

    feature_mean, feature_std = compute_feature_stats(train_d["X"])

    X_train = (train_d["X"] - feature_mean) / feature_std
    X_val = (val_d["X"] - feature_mean) / feature_std
    Y_train, Y_val = train_d["Y"], val_d["Y"]

    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val)

    model = MLP(hidden_sizes=HIDDEN_SIZES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=12)
    loss_fn = nn.MSELoss()

    n_train = X_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print(f"Training on {n_train} samples, validating on {X_val_t.shape[0]} samples")
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            xb, yb = X_train_t[idx], Y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, Y_val_t).item()
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            marker = " *" if improved else ""
            print(f"  epoch {epoch:4d}  train_mse={epoch_loss:.5f}  val_mse={val_loss:.5f}{marker}")

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    print(f"Training took {time.time() - t0:.1f}s. Best val MSE (transformed scale): {best_val_loss:.5f}")
    model.load_state_dict(best_state)
    model.eval()
    return model, feature_mean, feature_std


def evaluate_split(model: MLP, feature_mean: np.ndarray, feature_std: np.ndarray, split_name: str) -> dict:
    d = load_split(split_name)
    X = (d["X"] - feature_mean) / feature_std
    with torch.no_grad():
        Y_pred_transformed = model(torch.from_numpy(X)).numpy()

    Rs_pred, Rct_pred, Q_pred, alpha_pred = transformed_to_params(
        Y_pred_transformed[:, 0], Y_pred_transformed[:, 1], Y_pred_transformed[:, 2], Y_pred_transformed[:, 3]
    )
    params_pred = np.stack([Rs_pred, Rct_pred, Q_pred, alpha_pred], axis=1)
    params_true = d["params_raw"]
    slice_label = d["slice_label"]

    from data_utils import SLICE_DEGENERATE

    results = {}
    subsets = {"overall": np.ones(len(slice_label), dtype=bool)}
    for sid, name in SLICE_NAMES.items():
        subsets[name] = slice_label == sid

    # Diagnostic-only breakdown: the degenerate slice is built from two
    # distinct non-identifiability regimes (Rct forced to the lowest or
    # highest decade of its range). Averaging them together can mask the
    # per-regime signal, so report them separately too (Rct is expected to
    # be poorly identified in the high-Rct half; Q/alpha in the low-Rct
    # half -- see PHASE2_NOTES.md).
    degenerate_mask = slice_label == SLICE_DEGENERATE
    rct_true = params_true[:, 1]
    subsets["degenerate_low_Rct_extreme"] = degenerate_mask & (rct_true < 3.162e5)
    subsets["degenerate_high_Rct_extreme"] = degenerate_mask & (rct_true >= 3.162e5)

    for subset_name, mask in subsets.items():
        n = int(mask.sum())
        per_param = {}
        for i, pname in enumerate(TARGET_NAMES):
            y_true = params_true[mask, i]
            y_pred = params_pred[mask, i]
            mae = float(np.mean(np.abs(y_true - y_pred)))
            r2 = r2_score(y_true, y_pred)
            per_param[pname] = dict(r2=r2, mae=mae, n=n)
        results[subset_name] = per_param
    return results


def format_report(all_results: dict) -> str:
    lines = ["# Phase 2 Evaluation Report", ""]
    for split_name, results in all_results.items():
        lines.append(f"## Split: {split_name}")
        lines.append("")
        for subset_name in [
            "overall",
            "main",
            "degenerate",
            "low_freq_coverage",
            "degenerate_low_Rct_extreme",
            "degenerate_high_Rct_extreme",
        ]:
            per_param = results[subset_name]
            n = per_param[TARGET_NAMES[0]]["n"]
            lines.append(f"### {subset_name} (n={n})")
            lines.append("")
            lines.append("| Parameter | Unit | R² | MAE |")
            lines.append("|---|---|---|---|")
            for pname, unit in zip(TARGET_NAMES, TARGET_UNITS):
                r2 = per_param[pname]["r2"]
                mae = per_param[pname]["mae"]
                lines.append(f"| {pname} | {unit} | {r2:.4f} | {mae:.6g} |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    model, feature_mean, feature_std = train()

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "mlp_state_dict.pt"))
    np.savez(os.path.join(MODELS_DIR, "norm_stats.npz"), feature_mean=feature_mean, feature_std=feature_std)
    print(f"Saved model state dict and normalization stats to {MODELS_DIR}")

    all_results = {}
    for split_name in ["train", "val", "test"]:
        all_results[split_name] = evaluate_split(model, feature_mean, feature_std, split_name)

    report = format_report(all_results)
    report_path = os.path.join(EVAL_DIR, "phase2_eval_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Wrote evaluation report to {report_path}")
    print()
    print(report)

    # Flag check (Task 3): main-slice overall R^2 on the test set should be
    # high for a converged fit on this synthetic, noise-bounded problem.
    main_test = all_results["test"]["main"]
    poor = {p: v["r2"] for p, v in main_test.items() if v["r2"] < 0.8}
    if poor:
        print("\n*** WARNING: main-slice test R^2 below 0.8 for:", poor)
        print("*** This may indicate a bug (normalization/parameterization), not just a modeling limit.")
    else:
        print("\nMain-slice test R^2 all >= 0.8 -- no red flag raised.")


if __name__ == "__main__":
    main()
