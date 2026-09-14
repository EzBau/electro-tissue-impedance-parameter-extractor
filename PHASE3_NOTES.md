# Phase 3 Notes: Frontend, Confidence Estimation, Static Space

Scope: a static, client-side page (served from the project root) that loads the Phase 2 ONNX
model, refines its output with a hand-written Levenberg-Marquardt fit, and
surfaces three independent confidence/warning signals. No backend calls.

## 1. How the confidence signal works

There is no single confidence score. Three independent signals are shown,
each attached to the specific parameter or parameters it's actually
about, and each can fire on its own.

### 1a. General residual (Task 1): `js/confidence.js: classifyResidual`

After the NN produces an initial estimate, [`js/lm_refine.js`](js/lm_refine.js)
runs a hand-written Levenberg-Marquardt local refinement (numeric
Jacobian, trust-region step clamping, box-constrained in transformed
parameter space) against the same forward model, minimizing
modulus-weighted residuals between predicted and measured Re/Im at every
provided frequency. The refinement's final weighted RMS residual is
bucketed: below 0.05 is "Low residual," a trustworthy fit; 0.05 to 0.15
is "Moderate residual," use with caution; above 0.15 is "High residual,"
meaning the fit did not converge to something consistent with the data,
shown explicitly rather than hidden behind clean-looking numbers.

Both the NN-only output and the post-refinement output are always shown
side by side, plus a per-parameter "NN vs refined" percentage. If they
diverge by more than 20% on any parameter, an explicit "NN estimate and
refined fit diverge" box appears. This was observed in testing: in the
Case 2 Warburg-like preset, the NN's Rs=6,725Ω versus the refined
Rs=5,000Ω, a 34.5% gap, and it's treated as informative rather than noise
to suppress.

### 1b. Rct-regime flag (Task 1a): `js/confidence.js: classifyRctRegime`

This is independent of the residual signal, because a fit can converge to
a low residual while still landing on a genuinely underdetermined
parameter combination. This is exactly the Phase 2 degenerate-slice
finding: the circuit is mathematically non-identifiable in these regimes,
not numerically unstable, so LM will happily converge, just to the wrong
Rct or the wrong Q/alpha.

It's checked against the NN's initial Rct estimate, per Task 1a's
instruction, using thresholds that intentionally match Phase 1's
degenerate-slice sampling bands exactly rather than the suggested generic
15-20% of range. If Rct is below 1e5 Ω, the "low extreme" of Phase 1's
`RCT_LOW_BAND`, the message reads "Rct estimate is well-constrained; Q
and α estimates are less reliable," since Phase 2 measured Q R²=0.85 and
α R²=0.84 there against 0.92-0.93 typical. If Rct is above 1e6 Ω, the
"high extreme" of Phase 1's `RCT_HIGH_BAND`, the message reads "Q and α
estimates are well-constrained; Rct estimate is less reliable," since
Phase 2 measured Rct MAE about 2.6x worse than typical there. Otherwise,
no flag appears, and the page relies on the residual signal alone.

The reasoning behind these exact thresholds, and not 15-20%: a 3-decade
Rct range (1e4-1e7) at 15-20% would cut around 1e4.5-1e4.6, roughly 28-40
kΩ. Phase 2 never evaluated degradation at that narrower boundary. It
only evaluated the full bottom and top decade ([1e4,1e5) and (1e6,1e7]),
because that's what Phase 1's degenerate slice sampled. Using a narrower,
untested cutoff would have been an unsupported extrapolation of an
empirical claim, so the wider, evidence-matched bands were used instead.
This was verified live in the app: the Case 1 preset (Rct=50,000, inside
the low band) correctly shows the low-regime flag, Case 2 (Rct=1e7,
inside the high band) correctly shows the high-regime flag, and values
in between, including the default page-load state, which lands in the
high band at Rct≈2.2e6, show the flag that matches their regime.

### 1c. Frequency-coverage check (Task 4): `js/confidence.js: checkFrequencyCoverage`

This is independent of both signals above, driven by bandwidth rather
than the Rct value or the residual. The thresholds again match a
specific Phase 2-evaluated band exactly: Phase 1's low-freq-coverage
slice restricted spectra to [100 Hz, 10 kHz], and Phase 2 measured severe
degradation there (Rct R² dropping from 0.95 to 0.45, Rs R² from 0.92 to
0.75). If the input's highest frequency doesn't exceed 10 kHz, the page
warns that Rs may be underdetermined, citing the exact Phase 2 R² drop.
If the input's lowest frequency doesn't go below 100 Hz, it warns that
Rct/Q/α may be underdetermined, citing the exact Phase 2 R² drop.

These use strict inequalities on purpose. A spectrum whose data spans
exactly [100 Hz, 10 kHz] and no further is the precise band Phase 2 found
problematic, so it must still trigger the warning rather than being
waved through as adequate for merely reaching the boundary. This was
caught and fixed during manual testing, where an early
inclusive-comparison version had silently passed that exact boundary
case as fine.

Both coverage warnings and the Rct-regime flag were confirmed in testing
to fire simultaneously and independently when both conditions hold, per
the task's explicit requirement.

## 2. Honest limits of this confidence signal

None of this is a statistically rigorous confidence interval. To be
specific about why:

The residual buckets, 0.05 and 0.15, are round numbers chosen because
Phase 1's noise model spans roughly 0.5% to 8% magnitude noise, so a
well-converged fit to noisy data should land in that rough order of
magnitude. They weren't fit to any calibration data, and there's no claim
that "moderate residual" corresponds to a specific error probability.

The Rct-regime flag is a rule of thumb read off Phase 2's aggregate
test-set metrics for two specific sampling bands. It says that in this
regime, on average, across thousands of synthetic examples, this
parameter was less reliable. It says nothing about the specific spectrum
in front of the user: a given noiseless spectrum in the low-Rct-extreme
band can still be fit exactly, and this was observed repeatedly in
testing, with the LM refinement recovering ground truth to under 1e-10
residual on several degenerate-regime presets with no noise added.

The frequency-coverage check only knows about the two exact bands Phase 2
evaluated. It has no model of how much worse things get between just
short of 10 kHz and way short of 10 kHz. It's a binary threshold check,
not a continuous confidence curve.

The LM refinement's box constraints, roughly a decade of slack beyond the
trained parameter ranges on each side, were added after observing the
unconstrained refinement diverge to physically nonsensical values (Rct
reaching roughly 1e15 Ω) from a bad initial guess in one test case. The
bounds prevent that specific failure mode, but they're a hand-picked
safeguard, not a principled uncertainty quantification.

None of the three signals combine into a single calibrated probability.
A user could see low residual, no regime flag, and adequate coverage all
at once, and still be looking at a locally optimal but wrong fit if the
true parameters sit in some other pathological region this project's
synthetic data never covered.

For Phase 4: this is diagnostic tooling built directly on what Phase 1/2
actually measured, not a substitute for real uncertainty quantification
such as an ensemble, quantile regression, or a proper
Bayesian/Fisher-information-based interval. That gap should carry
forward into the Phase 4 limitations writeup.

## 3. Two input modes (Task 2)

Synthesize mode uses sliders (log-scale for Rs/Rct/Q, linear for α) to
generate a clean spectrum via [`js/forward_model.js`](js/forward_model.js),
with an optional Gaussian noise toggle (magnitude percent and phase
degrees, independently sampled per point, mirroring Phase 1's noise model
at a simplified single-level-per-run granularity). Three preset buttons
load the exact Phase 1 hand-checked case parameters, and the page
defaults to Case 3's parameters on load rather than an arbitrary default.
That means simply opening the page runs the full NN+refinement pipeline
against a known case and visibly displays the regime flag firing
correctly: a live regression check every time the page loads, as
required.

Upload mode lets the user paste or upload a 3-column CSV (frequency,
col1, col2).

## 4. JS forward-model port (deliverable 2)

[`js/forward_model.js`](js/forward_model.js) is a
from-scratch JS port of `src/forward_model.py` (closed-form complex
arithmetic, no generic complex-number library). It's verified against the
same 3 Phase 1 hand-checked cases in
[`verification/verify_js_forward_model.mjs`](verification/verify_js_forward_model.mjs)
(`node verification/verify_js_forward_model.mjs`). All pass, matching the
Python reference to relative error around 1e-16, the float64 precision
limit, on Cases 1 and 2, and all Nyquist-shape checks pass on Case 3.

## 5. Upload unit validation (Task 3)

[`js/csv_parse.js`](js/csv_parse.js) never infers units. The user
must explicitly select frequency unit (Hz / rad/s), impedance
representation (Re/Im / Magnitude/Phase), and phase unit (degrees/radians,
shown only when Magnitude/Phase is selected). An early CSS bug, where a
`.field { display: flex }` rule overrode the `hidden` attribute, was
caught and fixed during testing, along with adding a page-wide
`[hidden] { display: none !important; }` rule as a safeguard. Parsed rows
are converted to canonical Hz / ohm and displayed back to the user in a
table before any inference runs; both the Re/Im and Magnitude/Phase paths
were verified to convert correctly in testing.

## 6. Visualization (Task 5)

This uses hand-rolled SVG, no chart library, in
[`js/plots.js`](js/plots.js): a Nyquist plot (equal-aspect, -Im(Z)
vs Re(Z)) and a two-panel Bode plot (log|Z| and phase vs log-frequency).
Both plots draw the measured spectrum as points on its actual frequency
grid, and the reconstructed spectrum as a smooth line evaluated on a
dense log grid via the same `computeSpectrum` function used for fitting.
There's one code path for what the spectrum looks like, shared between
the fit and the display, so a plotted mismatch can't hide a sign or unit
bug that isn't also present in the fit itself.

## 7. Architecture note: the NN's fixed input grid versus arbitrary upload data

The ONNX model requires exactly 40 values at the 20 canonical frequencies
(the 1 Hz to 100 kHz log grid). An uploaded spectrum can have any
frequencies. The NN's input is built by piecewise-linear interpolation,
in log-frequency space, of the uploaded Re/Im onto the canonical grid,
clamped rather than extrapolated outside the uploaded range. This only
feeds the NN's initial guess. The subsequent LM refinement always fits
against the real, uploaded frequency and Re/Im values, never the
interpolated ones, so the reported residual and final parameters are
never influenced by fabricated data. This design decision isn't specified
in the task brief, which implicitly assumes 20-point canonical input
everywhere, and it's called out here explicitly since it's a nontrivial
judgment call.

## 8. Static Space constraints (Task 6)

Everything runs client-side: `onnxruntime-web` (WASM backend,
`numThreads: 1` so no COOP/COEP cross-origin-isolation headers are
required, since a plain static host won't set those), and hand-written JS
for LM refinement and SVG plotting. There are no backend calls, confirmed
by inspection (no `fetch` to any non-local URL in the code) and by the
app functioning fully with the local static file server as the only
server involved.

Page weight: the total static-page payload is about 14 MB, almost entirely
the onnxruntime-web WASM binary (`ort-wasm-simd-threaded.wasm`, about
13.96 MB uncompressed). This build of onnxruntime-web (1.29.0) only ships
combined SIMD+threaded WASM variants, with no smaller single-purpose
build available. The actual model (`model.onnx`) is 125 KB, and all
hand-written app code (HTML/CSS/JS) totals about 76 KB. The WASM runtime
dominates page weight by roughly two orders of magnitude over everything
this project actually built, worth flagging plainly rather than glossing
over. A static HF Space will serve about 14 MB on first load, cached by
the browser after that. WASM also compresses well over the wire,
typically to a fraction of its raw size, though this wasn't measured with
real gzip/brotli headers since the local test server doesn't set them. If
page weight becomes a concern, a smaller or older onnxruntime-web build
could be evaluated in a later phase; that's out of scope here.

A couple of non-blocking bugs were found and fixed during live testing in
this phase. They aren't part of the required deliverables, but they're
worth recording. First, `ort.env.wasm.wasmPaths` resolves the `.wasm`
binary relative to `ort.wasm.min.js`'s own script URL, while dynamically
importing the `.mjs` wrapper relative to the page URL. These two
different resolution bases meant a single base-directory string couldn't
correctly locate both files, so the explicit per-filename mapping form is
used instead (see `js/onnx_inference.js`). Second, an invalid
`toPrecision(0)` call in one warning message (the argument must be
between 1 and 100) was only reachable after fixing the frequency-coverage
boundary logic, so it had been hidden by the first bug.

## Deliverables index

1. Working page: the project root (`index.html`, `js/*.js`, `css/style.css`,
   `assets/model.onnx`, `vendor/onnxruntime-web/`)
2. JS forward-model port and verification:
   [`js/forward_model.js`](js/forward_model.js),
   [`verification/verify_js_forward_model.mjs`](verification/verify_js_forward_model.mjs)
3. This file.
