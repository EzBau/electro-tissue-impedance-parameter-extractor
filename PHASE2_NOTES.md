# Phase 2 Notes: Model Training and ONNX Export

Scope: this phase trains a point-estimate regressor for Randles-CPE
parameters and exports it for browser inference. No frontend work happens
here.

## 1. Architecture

Feedforward MLP, tuned up from the Task 2 starting point:

```
40 -> 128 -> 128 -> 64 -> 4   (ReLU between hidden layers, linear output head)
```

This started from the suggested 40→64→64→32→4. The wider version converged
to meaningfully better validation loss and main-slice R² (see section 4),
so it replaced the starting point rather than being layered on top of it.

The output head predicts the four transformed-scale targets directly:
`[log10(Rs), log10(Rct), log10(Q), logit(alpha)]`. No raw-scale prediction
happens inside the trainable network; raw-scale conversion only happens in
the `ExportWrapper` (section 3) via `forward_model.transformed_to_params`,
the same function used throughout Phase 1/2 for this conversion.

Defined in [`src/model_def.py`](src/model_def.py) (`MLP` class).

## 2. Input representation and normalization

The features are a 40-dim vector, `[Re(Z) at freq_grid[0..19], Im(Z) at
freq_grid[0..19]]`, the Re block first, then the Im block, both in
ascending-frequency order. This exact order is the frontend contract
(section 6).

Normalization is per-feature standardization, `(x - mean) / std`, with
`mean`/`std` computed once from the train split only (all 70,000 training
spectra, across all three slices), never touching val/test. The 40
mean/std values are stored in `models/norm_stats.npz` and are baked into
the exported ONNX graph (section 3), so the frontend does not need to
replicate this arithmetic.

The low-freq-coverage slice stores `NaN` for frequency points outside its
measured 100 Hz–10 kHz sub-band, a Phase 1 convention. For both training
and inference, a missing point is zero-imputed (`Re = Im = 0.0` raw ohms)
before standardization. This is a deliberate, simple design choice, not a
claim that 0 ohms was actually measured, and the network is given no
separate "this point is missing" signal (no mask channel), per Task 1's
fixed 40-value input spec. See section 5 for how this shows up in the
low-freq-coverage metrics, and section 7 for why this matters for the
frontend.

Implemented in [`src/data_utils.py`](src/data_utils.py) and
[`src/train_model.py`](src/train_model.py).

## 3. ONNX export (baked-in transforms)

[`src/model_def.py`](src/model_def.py)'s `ExportWrapper` composes, as a
single `torch.nn.Module` and therefore a single ONNX graph after export:

1. input normalization using the stored train-set `feature_mean`/`feature_std`
   (registered as ONNX-graph buffers, not external files),
2. the trained MLP forward pass,
3. inverse transforms: `10**log_Rs`, `10**log_Rct`, `10**log_Q` (ONNX `Pow`
   op) and `sigmoid(logit_alpha)` (ONNX `Sigmoid` op).

Exported via `torch.onnx.export(..., dynamo=False, opset_version=17)` to
[`models/model.onnx`](models/model.onnx) (about 125 KB). Export script:
[`src/export_onnx.py`](src/export_onnx.py). The exported graph takes raw
Re/Im ohms in and returns raw-scale `[Rs, Rct, Q, alpha]` out, so the
frontend should not need to reimplement any normalization or
inverse-transform math.

The Python-side sanity check (PyTorch `ExportWrapper` vs. ONNX Runtime, 20
test spectra) found a max relative difference of 2.26e-6, consistent with
float32 rounding noise; see
[`eval/onnx_python_sanity_check.txt`](eval/onnx_python_sanity_check.txt).

## 4. Training

The optimizer is Adam, initial LR `2e-3`, weight decay `1e-5`, with an
LR schedule of `ReduceLROnPlateau(factor=0.5, patience=12)` on val loss.
Batch size is 512 and the loss is plain MSE on the 4 transformed-scale
targets (their train-set standard deviations are within about 3x of each
other, 0.36 to 1.17, so no per-target loss reweighting was needed). Early
stopping uses a patience of 40 epochs on val MSE, with a max of 800
epochs.

The actual run stopped at epoch 418 (about 38.7 minutes on CPU), with a
best val MSE (transformed scale, mean over the 4 outputs) of 0.0700. The
seed was 20260910 (numpy and torch), the same seed as Phase 1's dataset
generation.

Script: [`src/train_model.py`](src/train_model.py). Saved artifacts:
[`models/mlp_state_dict.pt`](models/mlp_state_dict.pt) (native PyTorch
weights) and [`models/norm_stats.npz`](models/norm_stats.npz).

## 5. Per-parameter, per-slice evaluation (Task 3)

Full numbers are in [`eval/phase2_eval_report.md`](eval/phase2_eval_report.md).
No bug flag is raised: main-slice test R² is 0.92–0.95 for all 4
parameters, comfortably clearing Task 3's 0.8 sanity threshold (chosen as
the point below which the result would indicate a bug).

### Test-split main-slice reference numbers

| Parameter | R² | MAE |
|---|---|---|
| Rs | 0.9243 | 2116 ohm |
| Rct | 0.9544 | 156,089 ohm |
| Q | 0.9239 | 2.35e-8 S·sᵅ |
| alpha | 0.9338 | 0.0226 |

### Degenerate slice: the honest, non-obvious result

The task brief predicted that alpha and Rct would show visibly worse
metrics on the degenerate slice. Taken as one aggregate slice, that's not
what the raw R² numbers show. Test-split degenerate R² (Rs 0.918, Rct
0.955, Q 0.915, alpha 0.911) looks about the same as, or even slightly
better than, main. That doesn't mean the model is quietly doing fine
everywhere. It means R² is a bad summary statistic for this particular
slice, and reporting it without the breakdown below would have hidden the
real signal, exactly what Task 3 warned against.

The degenerate slice is built from two distinct non-identifiability
regimes, Rct forced to the lowest or highest decade of its range (see
[`PHASE1_NOTES.md`](PHASE1_NOTES.md) section 5). Splitting it into those
two halves, added as a diagnostic-only breakdown in `train_model.py` as
`degenerate_low_Rct_extreme` / `degenerate_high_Rct_extreme`, shows the
expected pattern cleanly on the test split:

| Parameter | Low-Rct extreme (Rct forced ~1e4–1e5) | High-Rct extreme (Rct forced ~1e6–1e7) | Main slice (reference) |
|---|---|---|---|
| Rct | R²=0.953, MAE=3,681 ohm | R²=0.901, MAE=398,944 ohm | R²=0.954, MAE=156,089 ohm |
| Q | R²=0.852, MAE=3.83e-8 | R²=0.978, MAE=1.12e-8 | R²=0.924, MAE=2.35e-8 |
| alpha | R²=0.835, MAE=0.0429 | R²=0.993, MAE=0.0093 | R²=0.934, MAE=0.0226 |

This is exactly the textbook identifiability behavior. When `Rct <<
|Z_CPE|`, the parallel branch is dominated by `Rct` alone, so Rct is read
off cleanly (MAE 42x smaller than main) while Q and alpha are
comparatively degraded relative to main. When `Rct >> |Z_CPE|`, the branch
is dominated by the CPE instead, so Q and alpha are pinned down extremely
well, better than main, while Rct itself becomes the poorly identified
one, with MAE 2.6x worse than main. Averaging these two opposite failure
modes together produces a slice-level R² that looks deceptively
unremarkable, because R²'s denominator (variance of the true values
within the subset) is inflated by the wide gap between the two Rct
clusters, which masks the within-cluster error growth. The per-parameter
MAE and the low/high breakdown are the numbers that actually carry the
signal here; the aggregate degenerate-slice R² should not be read as "no
degradation."

This result wasn't tuned to come out this way. It's what the trained
model actually does, reported plainly as required.

### Low-freq-coverage slice: real, substantial degradation, as expected

Test split, main versus low-freq-coverage:

| Parameter | Main R² | Low-freq-coverage R² | Main MAE | Low-freq-coverage MAE |
|---|---|---|---|---|
| Rs | 0.924 | 0.754 | 2,116 ohm | 4,416 ohm |
| Rct | 0.954 | 0.454 | 156,089 ohm | 722,238 ohm |
| Q | 0.924 | 0.602 | 2.35e-8 | 6.60e-8 |
| alpha | 0.934 | 0.776 | 0.0226 | 0.0478 |

Restricting the measured band to 100 Hz–10 kHz degrades all four
parameters, most severely Rct (the low-frequency real-axis plateau that
pins down `Rs+Rct` often falls outside this narrower band for larger
`Rct*Q` time constants) and Q (the full curve shape is under-sampled). Rs
degrades moderately, since its signature is at the highest frequencies,
partly still inside 10 kHz but missing the 10 kHz–100 kHz tail. This
matches the task's expectation and Phase 1's stated intent for this
slice. This model produces point estimates only, not calibrated
uncertainty. It degrades in the right direction but does not itself flag
low confidence to a caller; see section 8.

## 6. Task 5: browser-runtime verification (onnxruntime-web)

Full output is in
[`eval/browser_runtime_verification.txt`](eval/browser_runtime_verification.txt).
The setup, in [`webtest/`](webtest/), is a Node script
(`run_verification.js`) using the `onnxruntime-web` npm package, the WASM
backend and the actual browser runtime library, not `onnxruntime-node`'s
native binding, run under Node 24.19 for convenience and loading
`models/model.onnx` directly.

Five fixed test-set spectra, `onnxruntime-web` output versus the
Python-side `onnxruntime` prediction on the identical inputs:

| Sample | Param | onnxruntime-web | python onnxruntime | rel. diff |
|---|---|---|---|---|
| idx 14368 | Rs | 11712.299 | 11712.286 | 1.08e-6 |
| idx 14368 | Rct | 17800.785 | 17800.785 | 0.0 |
| idx 14368 | Q | 8.4506e-9 | 8.4505e-9 | 2.21e-6 |
| idx 14368 | alpha | 0.946703 | 0.946703 | 6.30e-8 |
| idx 7588 | Rs | 41479.738 | 41479.738 | 0.0 |
| idx 2667 | Rct | 328508.94 | 328509.31 | 1.14e-6 |
| idx 8475 | Q | 5.68498e-7 | 5.68498e-7 | 0.0 |
| idx 11363 | alpha | 0.769163 | 0.769163 | 3.10e-7 |

The full 5-sample by 4-param table is in the linked file. The max
relative difference across all 20 values is 4.49e-6, consistent with
float32 rounding noise and not a functional discrepancy.
`onnxruntime-web` loads the model and runs successfully with the WASM
execution provider, and no op-compatibility issues were found (`Sub`,
`Div`, `Gemm`/`MatMul`, `Relu`, `Pow`, and `Sigmoid` all ran without
incident).

## 7. Task 6: hand-checked-case sanity check

Full output is in
[`eval/phase2_handcheck_report.txt`](eval/phase2_handcheck_report.txt).
This is a spot check on 3 specific points, not a substitute for section
5's aggregate metrics. Errors on individual off-distribution points can
legitimately be much larger than the aggregate MAE/R² above.

| Case | Rs rel. err | Rct rel. err | Q rel. err | alpha rel. err |
|---|---|---|---|---|
| 1: α≈1.0 parallel-RC (Rs=2000, Rct=50000, Q=2e-7, α=0.999999) | 10.2% | 3.6% | 3.4% | 2.8% |
| 2: α=0.5 Warburg-like (Rs=5000, Rct=1e7, Q=3e-8, α=0.5) | 34.5% | 8.6% | 1.8% | 1.0% |
| 3: degenerate-slice sample (Rs≈39330, Rct≈2.22e6, Q≈1.03e-7, α≈0.619) | 6.6% | 1.5% | 9.2% | 0.6% |

Cases 1 and 2 both sit at or near the edge of the training distribution.
α=0.999999 is at the extreme edge of the [0.5, 1.0) range, and Rct=1e7 in
Case 2 is the exact upper boundary of the trained range, where training
data is sparser and error is expected to be larger than the
in-distribution aggregate metrics. This is consistent with section 5, not
contradictory to it. Case 3's errors are small and line up with the
high-Rct-extreme diagnostic row in section 5, since its Rct≈2.22e6 falls
in that regime: excellent alpha accuracy (0.6% error, matching that
regime's R²=0.993), good Rct accuracy (1.5% error), and a larger relative
Q error (9.2%). Q's absolute error is still small, but Q's true value
here, 1.03e-7, is close to the low end of its log-uniform range, which
inflates relative error for a fixed absolute error.

## 8. Honest summary: where this model is reliable and where it isn't

The model is reliable, meaning R² above 0.9, under main-slice-like
conditions: the full 1 Hz–100 kHz band, with Rct not at an extreme. That
covers all four parameters, when the input spectrum spans the full
frequency band and Rct isn't within roughly a decade of the sampled
range's edges.

It degrades in a specific, physically expected way that isn't a modeling
bug. Rct becomes less reliable specifically when Rct is very large
relative to the CPE branch impedance, a fundamental non-identifiability
that isn't fixable by more training (see section 5). Q and alpha become
less reliable specifically when Rct is very small relative to the CPE
branch impedance, for the same reason. All four parameters degrade, Rct
most severely, when the input spectrum only covers a restricted sub-band
(100 Hz–10 kHz here) rather than the full 1 Hz–100 kHz range.

A few gaps are not addressed in this phase. There is no uncertainty
output: this is a point-estimate regressor that degrades accuracy in the
expected direction on hard cases (section 5) but never itself says "I'm
not confident here." A caller can't distinguish a well-identified
prediction from a degenerate one without external knowledge of which
regime the input spectrum falls into. If Phase 3/4 wants calibrated
confidence, that requires new modeling work, such as an ensemble,
quantile regression, or an auxiliary classifier for slice membership,
which is out of scope here. Zero-imputation of missing frequency points
is undisclosed to the network: no mask channel was added (Task 1
specifies a fixed 40-value input), so a genuinely zero Re/Im reading at
some frequency and a "not measured" point are indistinguishable to the
model. This only matters if Phase 3 ever feeds it a partial-band real
measurement; a full-band measurement, the expected normal case, is
unaffected. The noise model gap carried over from Phase 1, no
line-frequency pickup or drift modeling, still applies, since this model
was never trained or evaluated against those artifact types. And the
hand-picked edge-of-range points in section 7 show larger error than the
in-distribution aggregate, so this model should not be trusted to
extrapolate meaningfully outside the trained parameter ranges documented
in [`PHASE1_NOTES.md`](PHASE1_NOTES.md) section 2.

## 9. Frontend input/output spec (Task 4)

Model file: [`models/model.onnx`](models/model.onnx) (opset 17, about 125
KB). It's self-contained; no separate normalization file is needed at
inference time.

Input: tensor named `re_im_spectrum`, shape `(batch, 40)`, `float32`, raw
ohms (not normalized; normalization is baked into the graph). Column
order is fixed:

```
index  0- 9: Re(Z) at f = [1, 1.833, 3.360, 6.158, 11.288, 20.691, 37.927, 69.519, 127.427, 233.572] Hz
index 10-19: Re(Z) at f = [428.133, 784.760, 1438.450, 2636.651, 4832.930, 8858.668, 16237.767, 29763.514, 54555.948, 100000.0] Hz
index 20-29: Im(Z) at the same 10 frequencies as index 0-9
index 30-39: Im(Z) at the same 10 frequencies as index 10-19
```

In other words, `[Re(Z) @ freq_grid[0..19]] ++ [Im(Z) @ freq_grid[0..19]]`,
both blocks in ascending-frequency order. `freq_grid` is the exact
20-point `np.logspace(0, 5, 20)` grid defined in
`forward_model.FREQ_GRID_HZ` in Phase 1; reuse that grid rather than
hardcoding a different one. If a real measured spectrum doesn't cover a
given point, the current convention (section 2) is to supply `0.0` for
both Re and Im at that point. Be aware this is unvalidated outside the
low-freq-coverage slice's synthetic band (see the gap note in section 8).

Output: tensor named `params_raw`, shape `(batch, 4)`, `float32`, raw
scale, fixed column order:

```
index 0: Rs    [ohm]
index 1: Rct   [ohm]
index 2: Q     [S * s^alpha]
index 3: alpha [dimensionless, in (0, 1)]
```

No further transformation is needed. Normalization and the inverse
`10^x`/`sigmoid` transforms are already applied inside the graph.

## 10. Deliverables index

1. Training script and trained model: [`src/train_model.py`](src/train_model.py),
   [`models/mlp_state_dict.pt`](models/mlp_state_dict.pt)
2. ONNX export: [`src/export_onnx.py`](src/export_onnx.py),
   [`models/model.onnx`](models/model.onnx)
3. Per-parameter/per-slice report: [`eval/phase2_eval_report.md`](eval/phase2_eval_report.md)
4. Browser-runtime verification: [`webtest/run_verification.js`](webtest/run_verification.js),
   [`eval/browser_runtime_verification.txt`](eval/browser_runtime_verification.txt)
5. This file.

Additional supporting files: [`src/model_def.py`](src/model_def.py),
[`src/data_utils.py`](src/data_utils.py),
[`src/export_webtest_samples.py`](src/export_webtest_samples.py),
[`verification/verify_phase2_handchecks.py`](verification/verify_phase2_handchecks.py)
(Task 6), [`eval/onnx_python_sanity_check.txt`](eval/onnx_python_sanity_check.txt),
[`eval/phase2_handcheck_report.txt`](eval/phase2_handcheck_report.txt).
