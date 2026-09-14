# Electrode-Tissue Impedance Parameter Extractor

**[Live demo on Hugging Face Spaces](https://huggingface.co/spaces/EzBau/Electrode-Tissue-Impedance-Parameter-Extractor)**

A browser-based tool that recovers Randles-CPE equivalent-circuit
parameters (series resistance `Rs`, charge-transfer resistance `Rct`,
constant-phase-element coefficient `Q`, and exponent `alpha`) from an
electrode-tissue impedance spectrum. A small neural network gives an
initial estimate. A hand-written Levenberg-Marquardt refinement then fits
that estimate against the same forward model used to generate training
data, and the page surfaces three independent, explicitly scoped
confidence signals instead of four equally confident-looking numbers.
Everything runs client-side (ONNX Runtime Web plus plain JS), with no
backend, and no data leaves the page.

This project was built in four phases: synthetic data generation (Phase
1), model training and ONNX export (Phase 2), the interactive frontend
(Phase 3), and this documentation and citation pass (Phase 4). The app
is deployed as a static Hugging Face Space (link above); this repository
holds the full source, training data, and phase-by-phase notes behind it.

## Methods

### Randles-CPE forward model

```
Z_CPE(f) = 1 / (Q * (j*2*pi*f)^alpha)
Z(f)     = Rs + (Rct * Z_CPE(f)) / (Rct + Z_CPE(f))
```

This is implemented twice: once in Python (`src/forward_model.py`, the
source of truth for training-data generation) and once in JS
(`js/forward_model.js`, for the frontend's synthesize mode and the
LM refinement). Both were verified against three hand-computed cases: a
pure Rs+Rct parallel-RC limit at `alpha`≈1, a Warburg-like closed form at
`alpha`=0.5, and a randomly sampled degenerate-slice spectrum checked for
a correct depressed-semicircle shape on a Nyquist plot. See
[`verification/verify_forward_model.py`](verification/verify_forward_model.py)
(Python) and
[`verification/verify_js_forward_model.mjs`](verification/verify_js_forward_model.mjs)
(JS). Both pass, and the JS port matches the Python original to about
1e-16 relative error, the limit of float64 precision.

### Synthetic training data

Parameters are sampled log-uniformly (`Rct`, `Q`) or uniformly (`Rs`,
`alpha`) over the ranges in the Grounding table below, across a 20-point
log-spaced frequency grid from 1 Hz to 100 kHz, in three slices:

| Slice | Fraction | Purpose |
|---|---|---|
| Main | 70% | Independent sampling across the full ranges |
| Degenerate-region stress test | 20% | `Rct` forced into the bottom or top decade of its range, to deliberately create near-indistinguishable spectra (see Limitations) |
| Low-frequency-coverage | 10% | Spectrum restricted to a 100 Hz to 10 kHz sub-band, to test behavior when `Rs` is poorly constrained |

The noise model is per-spectrum Gaussian multiplicative magnitude noise
(0.5% to 8% of magnitude, sampled per spectrum) plus additive Gaussian
phase noise (0.1° to 3.0°, sampled per spectrum), applied independently at
each frequency point. See Limitations for what this noise model doesn't
capture.

Targets are stored transformed (`log10(Rs)`, `log10(Rct)`, `log10(Q)`,
`logit(alpha)`) to keep the multi-order-of-magnitude parameter spread
learnable. The exact transform and inverse-transform functions live in
`forward_model.py`/`forward_model.js` and are reused unchanged from
training through browser inference, so there's no risk of drift between
implementations. Full detail is in [`PHASE1_NOTES.md`](PHASE1_NOTES.md).

### Neural network

A feedforward MLP: `40 → 128 → 128 → 64 → 4` (ReLU, linear output head).
The input is a 40-dim vector, `[Re(Z) × 20, Im(Z) × 20]`, at the canonical
frequency grid, standardized using train-set-only mean and standard
deviation. The output is the four transformed targets directly. This is
exported to ONNX with input normalization and the inverse transforms
(`10^x`, `sigmoid`) baked into the graph, so the exported model takes raw
Re/Im in and returns raw-scale `Rs`/`Rct`/`Q`/`alpha` out with no math for
the frontend to replicate. Test-set main-slice R² sits between 0.92 and
0.95 across all four parameters. Full detail, including the
degenerate-slice per-parameter breakdown, is in
[`PHASE2_NOTES.md`](PHASE2_NOTES.md).

### Local refinement (LM)

The neural network's output seeds a hand-written Levenberg-Marquardt
optimizer (`js/lm_refine.js`) that fits the same forward model
against the measured spectrum by minimizing modulus-weighted residuals
(weighting by `1/|Z_measured|` at each point, standard EIS-fitting
practice that keeps low-frequency, high-magnitude points from dominating
over high-frequency, low-magnitude ones). It runs in the transformed
parameter space, so `Rs`, `Rct`, and `Q` stay positive and `alpha` stays
between 0 and 1 automatically, with a numeric Jacobian, trust-region step
clamping, and box constraints roughly a decade beyond the trained
parameter range on each side. Those bounds were added after testing showed
an unconstrained refinement could diverge to physically nonsensical
values (`Rct` reaching roughly 1e15 ohm) from a bad initial guess.

### Confidence-signal design

Three independent signals are shown, each attached to the specific
parameter or parameters it concerns:

1. General residual: the LM refinement's final weighted RMS residual,
   bucketed low, moderate, or high.
2. Rct-regime flag: checked against the neural network's initial `Rct`
   estimate, using the exact bands Phase 1's degenerate slice sampled
   (`Rct`<1e5 Ω or `Rct`>1e6 Ω). Phase 2 found that `Rct` and `Q`/`alpha`
   degrade in opposite regimes there, which is real circuit-theory
   non-identifiability rather than a training artifact (see Limitations).
3. Frequency-coverage warning: checked against Phase 1's exact
   low-freq-coverage band (100 Hz to 10 kHz), the only band Phase 2
   actually measured degradation on.

Both the neural-network-only and post-refinement outputs are always shown
side by side, and disagreement of more than 20% between them is flagged
explicitly. Full design rationale and honest limits are in
[`PHASE3_NOTES.md`](PHASE3_NOTES.md).

## Grounding

The table below shows which parameter ranges are literature-cited and
which are flagged as chosen for numerical coverage without literature
support. All citations were re-verified in Phase 4 against primary-source
PDF text directly, not secondary search summaries; see "Citation
verification process" below for what changed from Phase 1.

| Parameter | Range used | Grounding |
|---|---|---|
| `Rs` | 1 kΩ to 50 kΩ | Supported by two directly measured values: Abidian & Martin 2009's 4.2 to 8.9 kΩ [2] and Nimbalkar et al. 2018's glassy-carbon `Rsol` = 4.4 kΩ [4], both real neural-microelectrode fits falling inside this range. The exact endpoints (1 kΩ, 50 kΩ) are **not literature-verified; chosen for numerical coverage** beyond these specific electrode geometries. |
| `Rct` | 10 kΩ to 10 MΩ | **Narrower than the real fitted spread.** The directly cited core, Cogan 2008's 50 kΩ to 1 MΩ total impedance at 1 kHz [1] and Abidian & Martin 2009's 48.8 to 129.4 kΩ charge-transfer resistance [2], falls inside this range. But real fitted charge-transfer-resistance-like values reported elsewhere span far wider: Nimbalkar et al. 2018 fit `Rct` = 3.34 kΩ on a glassy-carbon microelectrode [4], below this range's floor, and Greenhorn/Delacour et al. 2024 fit resistances up to 97 MΩ for rigid TiN/graphene-on-sapphire microelectrodes (versus 0.7-0.98 MΩ for their own flexible gold/graphene devices) [5], above this range's ceiling. Read this range as targeting a specific moderate-impedance electrode class deliberately, not as a literature-pinned outer bound; widening the ceiling toward 50-100 MΩ would cover the fuller measured spread, at the cost of stretching the degenerate-slice sampling further. |
| `Q` | 1e-9 to 1e-6 S·sᵅ | Close to, but slightly narrower than, the real measured spread. The lower portion is grounded by Abidian & Martin 2009's directly measured 5.6e-10 to 7.0e-10 S·sⁿ [2]. Greenhorn/Delacour et al. 2024 report CPE coefficients spanning 6.7e-10 to 1.25e-6 S·sᵅ across their own and cited rigid/flexible microelectrodes [5], almost exactly bracketing this range; the minor undercoverage at both ends is treated as acceptable, not as a verified exact bound. |
| `alpha` | 0.5 to 1.0 | The upper portion is directly supported by real measurements: Abidian & Martin's `n` of 0.55 to 0.93 [2], Nimbalkar et al.'s `alpha` = 0.87 [4], and Greenhorn/Delacour et al.'s `n` of 0.88 to 1.0 for flexible gold/graphene microelectrodes [5]. The 0.5 floor sits below every directly measured value found here (lowest: 0.55); it is a modest extension past the lowest measured point, broadly consistent with the general CPE-modeling convention (`alpha` in [0,1]), rather than a value hit exactly by a citation. |
| The qualitative claim that alpha near 1.0 means a healthy interface and lower alpha means a degraded one (motivation only, not an implemented label) | n/a | **Not literature-verified**, after two search attempts (Phase 1 and Phase 4) across multiple candidate chronic-implant EIS studies. No implemented behavior depends on this claim being true. It only motivated sampling the full 0.5 to 1.0 range. |

## Limitations

These are stated specifically, not as generic hedging.

This model is trained entirely on synthetic spectra from the closed-form
Randles-CPE model. It has not been validated against real bench EIS
measurements (Gamry, Autolab, or equivalent instrumentation) on actual
electrode-tissue or electrode-electrolyte systems, because that validation
data doesn't exist for this project. Every number reported here, R², MAE,
the confidence signals, describes performance on data drawn from the same
synthetic distribution the model was trained on, not real-world
generalization.

The noise model is multiplicative Gaussian magnitude noise plus additive
Gaussian phase noise, and nothing else. It does not capture real
instrument artifacts: 50/60 Hz line-frequency pickup, electrode drift over
the course of a scan, contact-motion artifacts, or non-Gaussian outliers.
This was an explicit, documented scope decision in Phase 1, carried
through unchanged.

The three confidence signals (Rct-regime flag, frequency-coverage warning,
LM residual) are empirical rules of thumb read off Phase 2's aggregate
test-set metrics, not calibrated statistical intervals. A noiseless
spectrum whose true parameters fall inside a flagged regime, `Rct` in the
degenerate low or high band, for example, can still be fit essentially
exactly by the LM refinement. This was observed repeatedly in Phase 3
testing, with residuals down to about 1e-10 on several degenerate-regime
presets with no added noise. The flags describe average behavior across
thousands of synthetic examples in that regime, not a guarantee about the
specific spectrum in front of a given user.

The model's degenerate-region behavior, the tradeoff between `Rct` and
`Q`/`alpha`, is real circuit-theory non-identifiability, not a training
artifact. When `Rct` is very large relative to the CPE branch impedance,
the circuit behaves almost like a pure CPE across the measured band:
`Q`/`alpha` are well-constrained by the spectrum's shape, but `Rct`'s exact
value barely affects the total impedance and is genuinely underdetermined
by the data. The opposite holds when `Rct` is very small. More training
data would not fix this, because it's a property of the circuit and the
measurement, not of the model.

Page weight is dominated by the ONNX Runtime Web WASM binary (about 14
MB), not by this project's own model (125 KB). This version of
onnxruntime-web (1.29.0) only ships combined SIMD+threaded WASM builds; no
smaller single-purpose variant was available to substitute. The decision
made in Phase 3/4 was to accept this tradeoff as-is, since WASM compresses
well over the wire and is cached by the browser after the first load,
rather than pin an older, smaller onnxruntime-web build and take on its
version-compatibility risk for a size difference that only matters once.
This can be revisited if page weight becomes a concrete problem in a later
phase.

## References

1. Cogan, S.F. (2008). "Neural Stimulation and Recording Electrodes."
   *Annual Review of Biomedical Engineering* 10:275-309.
   DOI: [10.1146/annurev.bioeng.10.061807.160518](https://doi.org/10.1146/annurev.bioeng.10.061807.160518).
   Used for order-of-magnitude support for the `Rct` range (in vivo 1 kHz
   impedance, 50 kΩ to 1 MΩ, directly quoted from the primary text).
2. Abidian, M.R. & Martin, D.C. (2009). "Experimental and theoretical
   characterization of implantable neural microelectrodes modified with
   conducting polymer nanotubes." *Biomaterials* 29(9):1273-1283.
   PMCID: [PMC2692518](https://pmc.ncbi.nlm.nih.gov/articles/PMC2692518/).
   Used for direct measurements of `Rs`, `Rct` (charge-transfer resistance,
   corrected in Phase 4 to use the paper's `Rt` column rather than `Rp`),
   `Q`, and `alpha` for real neural microelectrodes (Tables 1-2).
3. "Electrode impedance analysis of chronic tungsten microwire neural
   implants: understanding abiotic vs. biotic contributions."
   PMCID: [PMC4021112](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4021112/).
   Consulted in Phase 4 as a candidate source for a chronic-implant
   `alpha`-vs-time relationship. It does not report CPE exponent values per
   timepoint, so it does not support that claim. It's listed here for
   transparency about what was checked, not as a citation for a specific
   number used in this project.
4. Nimbalkar, S., Castagnola, E., Balasubramani, A., Scarpellini, A.,
   Samejima, S., Khorasani, A., Boissenin, A., Thongpang, S., Moritz, C. &
   Kassegne, S. (2018). "Ultra-Capacitive Carbon Neural Probe Allows
   Simultaneous Long-Term Electrical Stimulations and High-Resolution
   Neurotransmitter Detection." *Scientific Reports* 8, 6958.
   DOI: [10.1038/s41598-018-25198-x](https://doi.org/10.1038/s41598-018-25198-x).
   PMCID: [PMC5934383](https://pmc.ncbi.nlm.nih.gov/articles/PMC5934383/).
   Used for a directly reported modified-Randles-CPE fit (Figure 5 caption)
   on a glassy-carbon microelectrode: `Rsol` = 4.4 kΩ, `Rct` = 3.34 kΩ,
   `Y0` (`Q`) = 4.07e-11 S·sᵅ, `alpha` = 0.87. Confirmed directly against
   the primary PMC full-text article, not a secondary summary.
5. Greenhorn, S., Coizet, V., Dupuit, V., Fernandez, B., Bres, G.,
   Claudel, A., Gasner, P., Warnking, J.M., Barbier, E.L. & Delacour, C.
   (2024). "Ultrathin, flexible and MRI-compatible microelectrode array
   for chronic single units recording within subcortical layers."
   arXiv:[2402.04389](https://arxiv.org/abs/2402.04389) [q-bio.NC].
   Manuscript template indicates submission to *Science Advances*.
   Used for real fitted charge-transfer-resistance-like and CPE values
   across flexible gold/graphene and rigid TiN/graphene-on-sapphire neural
   microelectrodes (main text and Figure 2, and supplementary Table S1),
   read directly from the primary PDF. Confirms real fitted resistances
   ranging from about 0.7 MΩ (flexible) to 97 MΩ (rigid), and CPE
   coefficients from about 6.7e-10 to 1.25e-6 S·sᵅ, both wider than this
   project's `Rct` and `Q` ranges.

### Citation verification process (Phase 4)

Phase 1's citations were drawn from secondary AI-generated search-result
summaries (WebSearch/WebFetch tool output), and were flagged at the time
as needing re-verification against primary sources. Phase 4 downloaded the
actual Cogan 2008 PDF and the actual Abidian & Martin 2009 PMC full text,
and read them directly.

Confirmed accurate: Cogan 2008's "50 kΩ to 1 MΩ" claim, whose exact quote
is on page 275, and Abidian & Martin 2009's `Rs` and `Q`/`alpha` values.

Found and corrected: the claim that Cogan 2008 discusses CPE exponent
behavior was false. That paper never mentions CPE, constant phase
elements, Warburg elements, or equivalent-circuit fitting at all, and the
claim has been retracted from `PHASE1_NOTES.md`. The `Rct` values
attributed to Abidian & Martin 2009 had used the wrong table column, pore
resistance `Rp` instead of charge-transfer resistance `Rt`, and have been
corrected to 48.8 to 129.4 kΩ (from the previously claimed 49 to 234 kΩ).

Remains unresolved after a genuine second attempt: a quantitative source
for the claim that alpha decreases with chronic fibrotic encapsulation.

**Second citation pass:** two further primary sources, Nimbalkar et al.
2018 [4] and Greenhorn/Delacour et al. 2024 [5], were located and their
exact numeric claims verified directly against primary full text (a PMC
article and an arXiv PDF respectively, not secondary summaries). These
don't replace or contradict the Cogan/Abidian & Martin verification above;
they add two more directly-cited real fits that close part of the
previously open gap ("a second EIS study with directly reported `Q`
values for larger electrodes... not found") and reveal that the `Rct`
range used here is narrower than the real measured spread across electrode
types (3.34 kΩ to 97 MΩ). See the Grounding table above and
[`PHASE1_NOTES.md`](PHASE1_NOTES.md) section 2 for full detail.

## License

[MIT](LICENSE)
