# Phase 1 Notes: Synthetic Training Data Generation

Scope: this phase generates and verifies the synthetic dataset only. No model
training or frontend work happens here.

> **Phase 4 correction pass (see section 2 for details):** the primary
> sources cited below were re-verified directly (downloaded PDFs, full-text
> searched) rather than relying on Phase 1's secondary search summaries.
> One citation was found to be wrong and has been corrected. The `alpha`
> claim attributed to Cogan 2008 does not actually appear in that paper and
> has been retracted. The `Rct` numbers cited from Abidian & Martin 2009
> used the wrong circuit-model column (pore resistance instead of
> charge-transfer resistance) and have been corrected. See `README.md`
> for the consolidated, corrected citation list.
>
> **Second citation pass:** two further primary sources, Nimbalkar et al.
> 2018 [4] and Greenhorn/Delacour et al. 2024 [5] (full PDF/full-text read
> directly, not secondary summaries), were added below. They don't
> contradict the correction above; they add real fitted values that close
> part of the previously open `Q`-range gap and show the `Rct` range used
> here is narrower than the real measured spread across electrode types.

## 1. Circuit model

Randles-CPE: series resistance `Rs`, in series with a parallel combination of
charge-transfer resistance `Rct` and a constant-phase element (`Q`, `alpha`).

```
Z_CPE(f) = 1 / (Q * (j*2*pi*f)^alpha)
Z(f)     = Rs + (Rct * Z_CPE(f)) / (Rct + Z_CPE(f))
```

Implemented in [`src/forward_model.py`](src/forward_model.py).

## 2. Parameter ranges: what's cited vs. NOT VERIFIED

| Parameter | Range used | Sampling | Status |
|---|---|---|---|
| `Rs` | 1 kΩ – 50 kΩ | uniform | Supported: real fitted values fall inside this range, Abidian & Martin's 4.2–8.9 kΩ [2] and Nimbalkar et al.'s `Rsol` = 4.4 kΩ [4]. Exact endpoints are still an engineering judgment call (see below). |
| `Rct` | 10 kΩ – 10 MΩ | log-uniform | **Narrower than the real fitted spread** [1][2]: real `Rct`-like fitted values range from 3.34 kΩ [4] up to 97 MΩ [5] across gold/graphene/TiN/glassy-carbon neural microelectrodes, exceeding both ends of this range (see below). |
| `Q` | 1e-9 – 1e-6 S·sᵅ | log-uniform | Close to, but slightly narrower than, the real measured spread of 6.7e-10 to 1.25e-6 S·sᵅ [5]; treated as minor, acceptable undercoverage rather than a verified exact bound. |
| `alpha` | 0.5 – 1.0 | uniform | Upper portion (0.87–1.0) directly supported [2][4][5]. The 0.5 floor reflects the general CPE-modeling convention (`alpha` in [0,1]) extended slightly past the lowest directly measured value (0.55 [2]), not a value hit exactly by a citation. |

### Sources consulted

1. **Cogan, S.F. (2008). "Neural Stimulation and Recording Electrodes."
   Annu. Rev. Biomed. Eng. 10:275–309.**
   ([Annual Reviews](https://www.annualreviews.org/content/journals/10.1146/annurev.bioeng.10.061807.160518),
   [PDF via microprobes.com](https://microprobes.com/files/pdf/publications/gen-knowledge/cogan_2008_neural_stimulation.pdf))

   Recording-electrode impedance magnitude at 1 kHz is reported to vary
   widely in vivo, roughly **50 kΩ to 1 MΩ**. This is a direct quote from
   page 275, verified against the primary PDF text and not a secondary
   summary: "Recording electrodes are typically characterized by their
   impedance at 1 kHz, which is quite variable, ranging in vivo from
   approximately 50 kΩ to 1 MΩ." This is total `|Z(1kHz)|`, not a
   decomposed `Rct` value, so it's used here as order-of-magnitude support
   for the `Rct` range only, not a direct `Rct` measurement.

   Correction from the Phase 4 re-verification: Phase 1 also attributed
   the claim "CPE exponent is usually higher than 0.5 for a flat rough
   electrode" to this paper. Phase 4 downloaded the primary PDF and
   full-text-searched it directly. Cogan 2008 does not discuss constant
   phase elements, CPE exponents, Warburg elements, Nyquist plots, or
   equivalent-circuit fitting at all: zero occurrences of "CPE",
   "constant phase", "Warburg", "Nyquist", or "equivalent circuit" turn up
   in the full 37-page text. That claim was a Phase 1 sourcing error,
   likely conflated from an unrelated secondary search result, and it is
   retracted here. The `alpha` floor of 0.5 is now grounded only by the
   Abidian & Martin measurements below (source 2), which is in any case a
   stronger source for this project since it's real neural microelectrode
   data rather than a generic claim.

2. **Abidian, M.R. & Martin, D.C. (2009). "Experimental and theoretical
   characterization of implantable neural microelectrodes modified with
   conducting polymer nanotubes."** *Biomaterials* 29(9):1273–1283.
   PMCID: PMC2692518. ([full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC2692518/))

   This paper reports a more detailed equivalent-circuit fit than a plain
   Randles circuit: `Rs` in series with [coating capacitance `Cc` in
   parallel with pore resistance `Rp`], in series with [double-layer CPE
   (`Y0`, `n`) in parallel with charge-transfer resistance `Rt`], in series
   with a finite-diffusion (Warburg-like) element, for gold neural-probe
   sites (1250 µm²). Table 1 (area-normalized) and Table 2 (absolute,
   `Rs`/`Rt` in kΩ) were read directly from the primary PDF text during the
   Phase 4 re-verification:

   | Coating | Rs (Ω) | Rt ≈ Rct (Ω) | Q ≈ Y0 (S·sⁿ) | n ≈ alpha |
   |---|---|---|---|---|
   | PPy film | 8,900 | 129,400 | 5.63e-10 | 0.55 |
   | PPy nanotubes | 6,600 | 48,800 | 7.00e-10 | 0.59 |
   | PEDOT film | 5,600 | 60,000 | 6.38e-10 | 0.91 |
   | PEDOT nanotubes | 4,200 | 79,200 | 6.00e-10 | 0.93 |

   Correction from the Phase 4 re-verification: Phase 1's table used the
   paper's pore resistance `Rp` (Table 1: 2922, 1181, 1305, 618 mΩ·cm²,
   giving roughly 49–234 kΩ once area-normalized) labeled as "Rct." That
   was the wrong column. This circuit's pore resistance sits in a
   different branch, parallel with the coating capacitance `Cc`, not
   parallel with the CPE. The paper's own charge-transfer resistance, `Rt`
   (Table 2, reported directly in absolute kΩ, parallel with the CPE and
   the correct structural analog of this project's `Rct`), is **48.8–129.4
   kΩ** across the four coatings, not 49–234 kΩ. `Rs` and `Q`/`n` were
   unaffected by this error; `Rs` from Table 2 matches Phase 1's
   area-normalized Table 1 calculation exactly, confirming that part of
   the original math was right.

   `Rs` (4.2–8.9 kΩ) and `alpha` (0.55–0.93) fall squarely inside the
   ranges used here, the strongest direct support in the dataset. `Rct`
   (48.8–129.4 kΩ, corrected) falls inside the 10 kΩ–10 MΩ range used,
   spanning less than half a decade of the roughly 3-decade range used
   here. The wider range is meant to additionally cover smaller and
   larger electrode geometries and a spectrum from healthy to degraded
   interfaces, which this single study doesn't itself characterize, so
   the full 10 kΩ–10 MΩ span should still be read as plausible but **NOT
   VERIFIED** beyond the directly-cited roughly 49–129 kΩ core (narrower
   than Phase 1 originally claimed).

   `Q` (5.6e-10–7.0e-10 S·sⁿ) sits below the 1e-9 floor used in this
   dataset. Since CPE coefficient scales roughly with electrode area, and
   these are small (1250 µm²) sites, larger contacts such as
   macro-electrodes or ECoG/DBS-style contacts would be expected to have
   proportionally larger `Q`. The 1e-9–1e-6 range remains an
   extrapolation and is flagged **NOT VERIFIED** at the exact bounds.
   Phase 4 searched again for a second, larger-electrode EIS study to
   narrow this gap and did not find one with directly reported `Y0`/area
   data suitable for this purpose; this gap is carried forward, unresolved,
   to Phase 5 and beyond.

3. **The qualitative claim that "alpha near 1.0 means a healthy,
   near-ideal interface, and lower alpha means a degraded, fibrotic
   interface,"** used in the task brief. Phase 4 made a second, targeted
   search attempt, specifically for longitudinal chronic-implant EIS
   studies reporting CPE exponent values over time, and checked two
   candidate primary sources directly. Abidian & Martin 2009 (above)
   reports `n` at a single timepoint only, freshly coated electrodes, with
   no chronic or longitudinal data. A chronic tungsten-microwire impedance
   study (PMC4021112, "Electrode impedance analysis of chronic tungsten
   microwire neural implants: understanding abiotic vs. biotic
   contributions") uses a CPE-based equivalent circuit across implant days
   1–205, but its reported tables give encapsulation resistance (`R_en`,
   9–77 kΩ) and electrolyte resistance, not CPE exponent values per
   timepoint.

   No source was found that reports `alpha` numerically as a function of
   implant time or fibrosis stage. This claim remains flagged **NOT
   VERIFIED** as a precise, citable fact after a genuine second attempt.
   It's used here only as qualitative motivation for why the dataset spans
   the full 0.5–1.0 range, not as a label mapping. This is an honest gap,
   not a placeholder left unchecked.

4. **Nimbalkar, S., Castagnola, E., Balasubramani, A., Scarpellini, A.,
   Samejima, S., Khorasani, A., Boissenin, A., Thongpang, S., Moritz, C. &
   Kassegne, S. (2018). "Ultra-Capacitive Carbon Neural Probe Allows
   Simultaneous Long-Term Electrical Stimulations and High-Resolution
   Neurotransmitter Detection."** *Scientific Reports* 8, 6958.
   DOI: [10.1038/s41598-018-25198-x](https://doi.org/10.1038/s41598-018-25198-x).
   PMCID: [PMC5934383](https://pmc.ncbi.nlm.nih.gov/articles/PMC5934383/).

   Reports a modified Randles-CPE fit for a homogeneous glassy-carbon (GC)
   neural microelectrode: electrolyte resistance in series with a parallel
   combination of `Rct`, a CPE, and a Warburg element. The fitted values,
   read directly from the Figure 5 caption in the primary PMC full-text
   article (not a secondary summary): `Rsol` = 4.4 kΩ, `Rct` = 3.34 kΩ,
   `Y0` (`Q`) = 4.07e-11 S·sᵅ, `alpha` = 0.87. This resolves the
   second-source citation this project previously carried only as an
   unconfirmed secondary (ResearchGate figure-caption) reference; the
   numbers above were independently confirmed against the primary PMC
   article text.

   `Rsol` (4.4 kΩ) falls inside this project's `Rs` range. `Rct` (3.34 kΩ)
   falls *below* the 10 kΩ floor used here, directly demonstrating that a
   real neural microelectrode can have a lower `Rct` than this dataset's
   sampled range covers. `Q` (4.07e-11 S·sᵅ) also falls below this
   project's 1e-9 floor. `alpha` (0.87) falls inside the upper portion of
   the range used here.

5. **Greenhorn, S., Coizet, V., Dupuit, V., Fernandez, B., Bres, G.,
   Claudel, A., Gasner, P., Warnking, J.M., Barbier, E.L. & Delacour, C.
   (2024). "Ultrathin, flexible and MRI-compatible microelectrode array
   for chronic single units recording within subcortical layers."**
   arXiv:[2402.04389](https://arxiv.org/abs/2402.04389) [q-bio.NC].
   Manuscript header indicates a *Science Advances* submission template.

   Reports electrochemical impedance spectroscopy on flexible gold and
   graphene microelectrodes, compared against rigid TiN-on-glass and
   graphene-on-sapphire microelectrodes, all fit to a resistive element in
   series with two Randles circuits (CPE parallel with `Rct`, in series
   with a linear diffusion element). Read directly from the primary PDF
   (main text and Figure 2, plus the supplementary Table S1 data quoted in
   text):

   | Device | Fitted resistance | CPE coefficient (`Q`) | CPE exponent (`n`/`alpha`) |
   |---|---|---|---|
   | Flexible gold | 0.98 MΩ | 84 nS (8.4e-8 S·sᵅ) | 0.886 |
   | Flexible graphene | 0.70 MΩ | 98 nS (9.8e-8 S·sᵅ) | 1.0 |
   | Rigid (TiN/graphene-sapphire) | up to 97 MΩ | as low as 670 pS (6.7e-10 S·sᵅ) | — |
   | Cited comparable rigid-substrate literature | 0.3 MΩ – 84 MΩ | 69 nS – 1.25 µS (6.9e-8 – 1.25e-6 S·sᵅ) | — |

   The exact quote: "The fitting resistances are lower (R1 = 0.7 and
   0.98 MΩ versus 97 MΩ respectively) and the capacitance are higher
   (CPE = 84nS and 98 nS versus 670 pS and 24nS respectively), for the
   flexible gold and graphene MEA, in comparison with all rigid
   microelectrodes... ranging from 0.3 to 84 MΩ and 1.25 µS to 69 nS."

   This is the source for the `Rct` range's 97 MΩ ceiling example and the
   `Q` range's 6.7e-10 to 1.25e-6 S·sᵅ span cited above. Note this paper's
   equivalent circuit is more complex than this project's single-Rct
   Randles-CPE model (it uses two Randles circuits plus a series
   resistance), so "fitted resistance" here is read as the structurally
   closest analog to `Rct`, not an identical decomposition; this is stated
   plainly rather than glossed over.

Use this dataset for its intended purpose, bootstrapping a
parameter-regression model, but do not present the exact numeric endpoints
of `Rs`, `Rct`, or the upper end of `Q` as literature-derived facts without
further verification. They are order-of-magnitude-consistent
extrapolations from the two sources above, explicitly marked as such.

## 3. Frequency grid

20 log-spaced points, 1 Hz to 100 kHz (`np.logspace(0, 5, 20)`), stored
explicitly as `freq_grid_hz` in every output file. Never hardcode this
grid separately in Phase 2/3; it's defined once in
`forward_model.FREQ_GRID_HZ`.

## 4. Training-scale parameterization

Targets are stored as `log10(Rs)`, `log10(Rct)`, `log10(Q)`, and
`logit(alpha) = ln(alpha/(1-alpha))`. Forward and inverse transforms live
in `forward_model.params_to_transformed` and `transformed_to_params`.
Reuse these functions, or a byte-identical port, in Phase 2 and Phase 3
rather than re-deriving them, to avoid train/inference drift.

## 5. Sampling strategy and slice composition

The total dataset is **100,000 spectra**. This size wasn't specified by
the task brief; it was chosen because it comfortably trains a small
regression model while generating in seconds on one core. The split is
70/15/15 train/val/test, with each of the three slices below split
independently at the same ratio before merging, so slice composition is
proportional across all three splits.

| Slice | Fraction | Train | Val | Test | Description |
|---|---|---|---|---|---|
| `main` (label 0) | 70% | 49,000 | 10,500 | 10,500 | Independent log-uniform/uniform sampling across full ranges |
| `degenerate` (label 1) | 20% | 14,000 | 3,000 | 3,000 | `Rct` forced into the lowest or highest decade of its range (50/50), rest sampled normally |
| `low_freq_coverage` (label 2) | 10% | 7,000 | 1,500 | 1,500 | Full parameter distribution, but spectrum restricted to 100 Hz–10 kHz sub-band |

`slice_label` (int8: 0/1/2), plus explicit boolean `degenerate_flag` and
`low_freq_flag` columns, let Phase 2 evaluate these slices separately
instead of averaging them into overall metrics, as required.

Degenerate slice mechanism: when `Rct >> |Z_CPE|` across the whole band,
the parallel branch is dominated by the CPE and `Rct` becomes practically
unidentifiable from the spectrum. When `Rct << |Z_CPE|`, the branch is
dominated by `Rct` and `Q`/`alpha` become unidentifiable instead. Both
regimes are sampled 50/50 by drawing `Rct` log-uniformly from the lowest
or highest decade of the full range.

Low-frequency-coverage slice mechanism: spectra are computed on the same
20-point grid, but points outside [100 Hz, 10 kHz] are stored as `NaN`
and flagged `False` in a per-sample `valid_mask` (N, 20) array, alongside
per-sample `freq_band_lo_hz`/`freq_band_hi_hz`. `Rs`'s signature is at the
highest frequencies, so restricting the band should starve the model of
the information needed to constrain `Rs`. That's intended for testing
low-confidence and high-uncertainty behavior in Phase 2, and isn't
resolved here.

## 6. Noise model

Applied per spectrum, not fixed across the dataset. Multiplicative
magnitude noise is Gaussian, with standard deviation sampled per spectrum
from 0.5% to 8% of magnitude, applied independently at each frequency
point. Phase noise is additive Gaussian, with standard deviation sampled
per spectrum from 0.1° to 3.0°, also applied independently at each
frequency point. Re/Im values are derived from that same noisy magnitude
and phase, not noised independently, so all four stored representations
stay mutually consistent.

Both noise ranges are engineering judgment calls, not pulled from a
published instrument noise-floor spec, and are flagged **NOT VERIFIED**.
This noise model captures only generic per-point measurement noise and
does not model real instrument artifacts such as line-frequency pickup
(50/60 Hz) or electrode drift over the course of a scan. That gap is a
known limitation, intended to be written up in the Phase 4 limitations
section rather than solved here.

## 7. Output format

`.npz` (numpy, no extra dependencies), one file per split:
`data/train.npz` (70,000), `data/val.npz` (15,000), `data/test.npz` (15,000).

Each file contains `freq_grid_hz` (20,), the shared frequency grid;
`magnitude_ohm`, `phase_deg`, `re_ohm`, and `im_ohm` (N, 20) float32,
the noisy spectra with `NaN` outside a sample's valid band; `valid_mask`
(N, 20) bool along with `freq_band_lo_hz`/`freq_band_hi_hz` (N,); `Rs`,
`Rct`, `Q`, `alpha` (N,) float32, the ground truth at raw scale;
`log_Rs`, `log_Rct`, `log_Q`, `logit_alpha` (N,) float32, the ground
truth at transformed scale; `slice_label` (N,) int8 (0=main,
1=degenerate, 2=low_freq_coverage) along with `degenerate_flag` and
`low_freq_flag` (N,) bool; and `noise_mag_pct`, `noise_phase_deg` (N,)
float32, the noise level applied to that spectrum.

## 8. Verification results (Task 7)

Run via `python verification/verify_forward_model.py`. All three required
hand-checks pass.

**Case 1: α = 1.0, reduces to standard parallel RC.**
`Rs=2000Ω, Rct=50000Ω, Q=2e-7 F` compared against the independent textbook
formula `Z = Rs + Rct/(1 + jωRctC)` at f = 10, 1000, 50000 Hz:

| f (Hz) | forward_model.py | independent formula | rel. error |
|---|---|---|---|
| 10 | 37847.8 − 22523.9j | 37847.8 − 22523.9j | 0.0 |
| 1000 | 2012.66 − 795.57j | 2012.66 − 795.57j | 5.3e-17 |
| 50000 | 2000.01 − 15.92j | 2000.01 − 15.92j | 0.0 |

f→0 limit: 52000 − 3.1j Ω, versus the expected `Rs+Rct=52000` (rel. err
6.0e-5). f→∞ limit: 2000 − 0.0008j Ω, versus the expected `Rs=2000` (rel.
err 4.0e-7). All pass.

**Case 2: α = 0.5, Warburg-like.**
Bare CPE (`Rs=5000Ω, Q=3e-8`) compared against the closed-form
`Z_CPE = 1/(Q·√(jω))`, which has an exact, frequency-independent phase of
−45°:

| f (Hz) | Z_CPE computed | Z_CPE closed-form | phase |
|---|---|---|---|
| 1 | 9.403e6 − 9.403e6j | 9.403e6 − 9.403e6j | −45.000000° |
| 1000 | 297354 − 297354j | 297354 − 297354j | −45.000000° |
| 100000 | 29735.4 − 29735.4j | 29735.4 − 29735.4j | −45.000000° |

The full circuit with `Rct→∞` (1e18 Ω) also matches `Rs + Z_CPE` to rel.
err below 2e-11. All pass.

**Case 3: random degenerate-slice sample, Nyquist shape check.**
Sampled: `Rs=3.93e4 Ω, Rct=2.22e6 Ω, Q=1.03e-7 S·sᵅ, alpha=0.619`. Four
numeric shape checks, not just a visual look, all pass: `Re(Z)` is
monotonically non-increasing with frequency; `−Im(Z) ≥ 0` at all points,
with no sign artifact; low-frequency `Re(Z)` is at least as large as
high-frequency `Re(Z)`, so the arc points the right way; and `−Im(Z)` is
unimodal across the sweep (a single arc, not multiple lobes; one sign
change was observed against a requirement of at most one).

The Nyquist plot saved to
[`verification/case3_degenerate_nyquist.png`](verification/case3_degenerate_nyquist.png)
visually confirms a single depressed semicircle, rising from the origin
region and beginning to close at the high-`Rct` end of the sampled band,
with no crossovers or self-intersections. A real computational artifact
would instead show a "return" loop, negative real values, or multiple
humps.

## 9. Known limitations (for Phase 4)

The noise model is generic Gaussian only, with no line-frequency pickup,
drift, or contact-motion artifacts; this gap is explicitly deferred, see
section 6. Several parameter-range endpoints are order-of-magnitude
extrapolations from a small number of cited sources rather than directly
measured values (see section 2); re-verify against a broader literature
survey before treating this as a clinically grounded prior. The
distinction between a healthy and a degraded interface, in terms of
`alpha`, is qualitative motivation only, not an implemented label or a
verified quantitative relationship.
