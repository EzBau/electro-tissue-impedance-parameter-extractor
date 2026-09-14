"""
Phase 2 ONNX export (Task 4).

Wraps the trained MLP with baked-in input normalization and baked-in
inverse transforms (10^x, sigmoid) using model_def.ExportWrapper, exports
to ONNX, then sanity-checks the exported graph against the native PyTorch
model on real test-set spectra using the Python onnxruntime (a *separate*
check from Task 5's browser-runtime verification, which uses
onnxruntime-web instead).

Usage: python export_onnx.py
Outputs:
  models/model.onnx
  eval/onnx_python_sanity_check.txt
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import load_split  # noqa: E402
from model_def import MLP, ExportWrapper  # noqa: E402
from train_model import HIDDEN_SIZES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
EVAL_DIR = os.path.join(ROOT, "eval")

N_SANITY_SAMPLES = 20
OPSET = 17


def main() -> None:
    state_dict = torch.load(os.path.join(MODELS_DIR, "mlp_state_dict.pt"), weights_only=True)
    mlp = MLP(hidden_sizes=HIDDEN_SIZES)
    mlp.load_state_dict(state_dict)
    mlp.eval()

    stats = np.load(os.path.join(MODELS_DIR, "norm_stats.npz"))
    feature_mean = torch.from_numpy(stats["feature_mean"])
    feature_std = torch.from_numpy(stats["feature_std"])

    export_model = ExportWrapper(mlp, feature_mean, feature_std)
    export_model.eval()

    dummy_input = torch.zeros(1, 40, dtype=torch.float32)
    onnx_path = os.path.join(MODELS_DIR, "model.onnx")
    torch.onnx.export(
        export_model,
        (dummy_input,),
        onnx_path,
        input_names=["re_im_spectrum"],
        output_names=["params_raw"],
        dynamic_axes={"re_im_spectrum": {0: "batch"}, "params_raw": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    print(f"Exported ONNX model to {onnx_path}")

    # --- Python-side sanity check: PyTorch ExportWrapper vs ONNX Runtime ---
    test_d = load_split("test")
    rng = np.random.default_rng(7)
    idx = rng.choice(len(test_d["X"]), size=N_SANITY_SAMPLES, replace=False)
    X_sample = test_d["X"][idx]  # raw (unstandardized) Re/Im, (N,40)

    with torch.no_grad():
        torch_out = export_model(torch.from_numpy(X_sample)).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["params_raw"], {"re_im_spectrum": X_sample.astype(np.float32)})[0]

    abs_diff = np.abs(torch_out - onnx_out)
    rel_diff = abs_diff / np.maximum(np.abs(torch_out), 1e-12)

    lines = ["Phase 2 Task 4: PyTorch (ExportWrapper) vs ONNX Runtime (Python) sanity check", ""]
    lines.append(f"{N_SANITY_SAMPLES} random test-set spectra, params = [Rs, Rct, Q, alpha]")
    lines.append("")
    param_names = ["Rs", "Rct", "Q", "alpha"]
    for i in range(N_SANITY_SAMPLES):
        lines.append(f"sample {i}:")
        for j, pname in enumerate(param_names):
            lines.append(
                f"  {pname:5s} torch={torch_out[i, j]:.8g}  onnx={onnx_out[i, j]:.8g}  "
                f"abs_diff={abs_diff[i, j]:.3e}  rel_diff={rel_diff[i, j]:.3e}"
            )
    lines.append("")
    lines.append(f"max abs diff overall: {abs_diff.max():.6e}")
    lines.append(f"max rel diff overall: {rel_diff.max():.6e}")
    report = "\n".join(lines)

    os.makedirs(EVAL_DIR, exist_ok=True)
    report_path = os.path.join(EVAL_DIR, "onnx_python_sanity_check.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nWrote {report_path}")

    if rel_diff.max() > 1e-3:
        print("*** WARNING: PyTorch vs ONNX Runtime outputs diverge more than expected. ***")
    else:
        print("PyTorch and ONNX Runtime outputs match closely.")


if __name__ == "__main__":
    main()
