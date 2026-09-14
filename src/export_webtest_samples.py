"""
Phase 2 Task 5 helper: pick 5 fixed test-set spectra, run them through the
exported ONNX model with the Python onnxruntime, and dump both the raw
inputs and the Python-side predictions to JSON so a Node/onnxruntime-web
script can run the *same* inputs through the *same* model file and the
comparison is apples-to-apples.

Usage: python export_webtest_samples.py
Output: webtest/test_samples.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import load_split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
WEBTEST_DIR = os.path.join(ROOT, "webtest")

N_SAMPLES = 5
PARAM_NAMES = ["Rs", "Rct", "Q", "alpha"]


def main() -> None:
    test_d = load_split("test")
    rng = np.random.default_rng(99)
    idx = rng.choice(len(test_d["X"]), size=N_SAMPLES, replace=False)
    X_sample = test_d["X"][idx].astype(np.float32)
    params_true = test_d["params_raw"][idx]
    slice_label = test_d["slice_label"][idx]

    onnx_path = os.path.join(MODELS_DIR, "model.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    python_pred = sess.run(["params_raw"], {"re_im_spectrum": X_sample})[0]

    payload = {
        "onnx_model_relpath": "../models/model.onnx",
        "param_names": PARAM_NAMES,
        "samples": [
            {
                "index": int(idx[i]),
                "slice_label": int(slice_label[i]),
                "input_re_im_40": X_sample[i].tolist(),
                "ground_truth_raw": {name: float(params_true[i, j]) for j, name in enumerate(PARAM_NAMES)},
                "python_onnxruntime_prediction": {
                    name: float(python_pred[i, j]) for j, name in enumerate(PARAM_NAMES)
                },
            }
            for i in range(N_SAMPLES)
        ],
    }

    os.makedirs(WEBTEST_DIR, exist_ok=True)
    out_path = os.path.join(WEBTEST_DIR, "test_samples.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path} with {N_SAMPLES} samples")


if __name__ == "__main__":
    main()
