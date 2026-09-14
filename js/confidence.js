// Confidence diagnostics for a single fit result. Three independent
// signals are surfaced to the user here. None of them is a rigorous
// statistical confidence interval; PHASE3_NOTES.md spells out the honest
// limits of each.
//
//  1. classifyResidual: general fit-quality signal from the LM refinement's
//     final weighted residual (Task 1).
//  2. classifyRctRegime: regime-specific flag from where the NN's *initial*
//     Rct estimate falls in the trained range, grounded directly in the
//     Phase 2 degenerate-slice evaluation (Task 1a).
//  3. checkFrequencyCoverage: bandwidth-driven flag from the input
//     spectrum's frequency span, grounded in the Phase 2 low-freq-coverage
//     slice evaluation (Task 4).

// --- Task 1a: Rct-regime flag -------------------------------------------
// The thresholds match Phase 1's degenerate-slice sampling bands exactly
// (RCT_LOW_BAND=[1e4,1e5), RCT_HIGH_BAND=(1e6,1e7]) rather than a generic
// "15-20% of range" cut. Those exact bands are what Phase 2's per-parameter
// metrics actually validated as showing the asymmetric Rct-vs-Q/alpha
// degradation; a narrower band we never evaluated would be an unsupported
// extrapolation of the confidence claim.
export const RCT_LOW_EXTREME_THRESHOLD_OHM = 1.0e5;
export const RCT_HIGH_EXTREME_THRESHOLD_OHM = 1.0e6;

export function classifyRctRegime(nnRctEstimateOhm) {
  if (nnRctEstimateOhm < RCT_LOW_EXTREME_THRESHOLD_OHM) {
    return {
      regime: "low",
      affectedParams: ["Q", "alpha"],
      message:
        "At this low a contact resistance, the electrode's capacitor-like behavior barely shows up in the " +
        "signal, so the tool can't read it out precisely. Rct itself is well-constrained here; Q and alpha " +
        "are the two numbers to treat with caution (Phase 2 measured Q R²=0.85 and alpha R²=0.84 in this " +
        "regime, versus 0.92 to 0.93 under typical conditions).",
    };
  }
  if (nnRctEstimateOhm > RCT_HIGH_EXTREME_THRESHOLD_OHM) {
    return {
      regime: "high",
      affectedParams: ["Rct"],
      message:
        "At this high a contact resistance, changing Rct barely changes what the spectrum looks like, so " +
        "the tool has very little to go on for that one number. The data itself doesn't pin it down; Q and " +
        "alpha are well-constrained here, but treat the Rct estimate with caution (Phase 2 measured Rct's " +
        "average error about 2.6 times worse than typical conditions in this regime).",
    };
  }
  return { regime: "none", affectedParams: [], message: null };
}

// --- Task 1: general residual-based confidence ---------------------------
// The weighted-RMS-residual thresholds are a heuristic, not derived from a
// formal calibration. Phase 1's synthetic noise model spans roughly
// 0.5% to 8% magnitude noise per spectrum, so a well-converged fit to noisy
// data should land with weighted RMS residual on the same rough order as
// the noise fraction. These bucket edges are round numbers chosen to sit
// comfortably inside or outside that range; they are not a fitted
// statistical boundary. See PHASE3_NOTES.md.
export function classifyResidual(weightedRmsResidual) {
  if (weightedRmsResidual < 0.05) {
    return {
      level: "low",
      label: "Low residual",
      message:
        "Once this answer was found, plugging it back into the model reproduces your spectrum almost " +
        "exactly. The refined fit matches the input spectrum closely: treat this as a trustworthy result.",
    };
  }
  if (weightedRmsResidual < 0.15) {
    return {
      level: "moderate",
      label: "Moderate residual",
      message:
        "The best answer the tool could find still doesn't quite reproduce your spectrum. There's a " +
        "visible gap between what the model predicts and what you measured, so use these numbers with " +
        "caution.",
    };
  }
  return {
    level: "high",
    label: "High residual",
    message:
      "No combination of the four numbers this model can produce reproduces your spectrum well. Either " +
      "the data is unusually noisy or messy, or it doesn't look like a typical electrode-tissue response " +
      "at all. The refinement could not converge to a fit consistent with the input spectrum, so these " +
      "parameter values are unreliable.",
  };
}

// --- Task 4: frequency-coverage check ------------------------------------
// The thresholds match Phase 1's low-freq-coverage slice band exactly
// ([100 Hz, 10 kHz]), the only band Phase 2 actually evaluated degradation
// on. See PHASE1_NOTES.md and PHASE2_NOTES.md.
export const HIGH_FREQ_COVERAGE_THRESHOLD_HZ = 1.0e4;
export const LOW_FREQ_COVERAGE_THRESHOLD_HZ = 100.0;

export function checkFrequencyCoverage(freqArrayHz) {
  const minFreq = Math.min(...freqArrayHz);
  const maxFreq = Math.max(...freqArrayHz);
  const warnings = [];

  // Strict inequalities on purpose. Phase 2 measured severe degradation for
  // a band spanning exactly [100 Hz, 10 kHz] (the low-freq-coverage slice),
  // so a spectrum that only reaches those exact boundaries and no further
  // should still trigger the warning rather than being waved through as
  // "adequate" on a technicality.
  const highOk = maxFreq > HIGH_FREQ_COVERAGE_THRESHOLD_HZ;
  const lowOk = minFreq < LOW_FREQ_COVERAGE_THRESHOLD_HZ;

  if (!highOk) {
    warnings.push({
      affectedParams: ["Rs"],
      message:
        "This test didn't sweep fast enough. Rs only shows itself clearly at the very fastest test speeds, " +
        "so without that data the tool is partly guessing at it. " +
        `This spectrum's highest frequency (${maxFreq.toPrecision(4)} Hz) doesn't go beyond ` +
        `${HIGH_FREQ_COVERAGE_THRESHOLD_HZ.toExponential(0)} Hz, and Rs is isolated at the high-frequency ` +
        "end of the spectrum. Without it, Rs is likely underdetermined " +
        "(Phase 2 measured Rs R² drop from 0.92 to 0.75 on a band missing this range).",
    });
  }
  if (!lowOk) {
    warnings.push({
      affectedParams: ["Rct", "Q", "alpha"],
      message:
        "This test didn't sweep slowly enough. Rct, Q, and alpha only fully show themselves at the slowest " +
        "test speeds, so without that data the tool is partly guessing at all three. " +
        `This spectrum's lowest frequency (${minFreq.toPrecision(4)} Hz) doesn't go below ` +
        `${LOW_FREQ_COVERAGE_THRESHOLD_HZ.toFixed(0)} Hz, and Rct/Q/alpha all depend on seeing the ` +
        "low-frequency part of the arc. Without it, they are likely underdetermined " +
        "(Phase 2 measured Rct R² drop from 0.95 to 0.45 on a band missing this range).",
    });
  }

  return { minFreq, maxFreq, decadesSpanned: Math.log10(maxFreq / minFreq), highOk, lowOk, warnings };
}
