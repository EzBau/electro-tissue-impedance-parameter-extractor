// JS port of src/forward_model.py. Must stay numerically identical to the
// Python version. See verification/verify_js_forward_model.mjs for the
// Phase 1 hand-checked-case porting sanity check.
//
// Z_CPE(f) = 1 / (Q * (j*2*pi*f)^alpha)
// Z(f)     = Rs + (Rct * Z_CPE(f)) / (Rct + Z_CPE(f))
//
// (j*omega)^alpha for real omega>0, real alpha, is derived in closed form
// (no generic complex-power routine needed): j*omega has magnitude omega
// and angle pi/2, so (j*omega)^alpha = omega^alpha * exp(i*alpha*pi/2).
// This is the same principal-branch value numpy's `(1j*omega)**alpha`
// computes for these always-positive-omega, real-alpha inputs.

export const FREQ_MIN_HZ = 1.0;
export const FREQ_MAX_HZ = 1.0e5;
export const N_FREQ_POINTS = 20;

// np.logspace(0, 5, 20) equivalent
export const FREQ_GRID_HZ = Array.from({ length: N_FREQ_POINTS }, (_, i) => {
  const exponent = 0 + (5 - 0) * (i / (N_FREQ_POINTS - 1));
  return Math.pow(10, exponent);
});

export const LOW_FREQ_BAND_LO_HZ = 100.0;
export const LOW_FREQ_BAND_HI_HZ = 1.0e4;

function complexDiv(aRe, aIm, bRe, bIm) {
  const denom = bRe * bRe + bIm * bIm;
  return { re: (aRe * bRe + aIm * bIm) / denom, im: (aIm * bRe - aRe * bIm) / denom };
}

// Z_CPE at a single frequency (Hz). Returns {re, im}.
export function zCpe(fHz, Q, alpha) {
  const omega = 2.0 * Math.PI * fHz;
  const magPart = 1.0 / (Q * Math.pow(omega, alpha));
  const angle = -alpha * (Math.PI / 2); // negative: 1/exp(i*alpha*pi/2) = exp(-i*alpha*pi/2)
  return { re: magPart * Math.cos(angle), im: magPart * Math.sin(angle) };
}

// Total Randles-CPE impedance at a single frequency (Hz). Returns {re, im}.
export function randlesCpeImpedance(fHz, Rs, Rct, Q, alpha) {
  const zc = zCpe(fHz, Q, alpha);
  const numRe = Rct * zc.re;
  const numIm = Rct * zc.im;
  const denRe = Rct + zc.re;
  const denIm = zc.im;
  const parallel = complexDiv(numRe, numIm, denRe, denIm);
  return { re: Rs + parallel.re, im: parallel.im };
}

// Full spectrum over an arbitrary frequency array (Hz). Returns
// { re: number[], im: number[] } matching freqArray's length/order.
export function computeSpectrum(Rs, Rct, Q, alpha, freqArray = FREQ_GRID_HZ) {
  const re = new Array(freqArray.length);
  const im = new Array(freqArray.length);
  for (let i = 0; i < freqArray.length; i++) {
    const z = randlesCpeImpedance(freqArray[i], Rs, Rct, Q, alpha);
    re[i] = z.re;
    im[i] = z.im;
  }
  return { re, im };
}

export function complexToMagPhaseDeg(re, im) {
  const magnitude = Math.hypot(re, im);
  const phaseDeg = (Math.atan2(im, re) * 180) / Math.PI;
  return { magnitude, phaseDeg };
}

// --- Parameter <-> training-scale transforms (must match forward_model.py) ---

export function paramsToTransformed(Rs, Rct, Q, alpha) {
  return [Math.log10(Rs), Math.log10(Rct), Math.log10(Q), Math.log(alpha / (1.0 - alpha))];
}

export function transformedToParams(logRs, logRct, logQ, logitAlpha) {
  const Rs = Math.pow(10, logRs);
  const Rct = Math.pow(10, logRct);
  const Q = Math.pow(10, logQ);
  const alpha = 1.0 / (1.0 + Math.exp(-logitAlpha));
  return [Rs, Rct, Q, alpha];
}
