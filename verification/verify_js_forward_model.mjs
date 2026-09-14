// Phase 3 deliverable #2: verify the JS forward-model port
// (webapp/js/forward_model.js) against the same 3 hand-checked cases used
// in Phase 1's verify_forward_model.py, confirming the port matches the
// Python original.
//
// Usage: node verify_js_forward_model.mjs

import {
  FREQ_GRID_HZ,
  randlesCpeImpedance,
  zCpe,
} from "../webapp/js/forward_model.js";

const PASS = "PASS";
const FAIL = "FAIL";
let allOk = true;

function check(label, got, expectedRe, expectedIm, tol = 1e-6) {
  const absDiff = Math.hypot(got.re - expectedRe, got.im - expectedIm);
  const expectedMag = Math.hypot(expectedRe, expectedIm) || 1e-30;
  const relErr = absDiff / expectedMag;
  const status = relErr < tol ? PASS : FAIL;
  if (status === FAIL) allOk = false;
  console.log(
    `    ${label}: got=${got.re.toPrecision(8)}${got.im >= 0 ? "+" : ""}${got.im.toPrecision(8)}j  ` +
      `expected=${expectedRe.toPrecision(8)}${expectedIm >= 0 ? "+" : ""}${expectedIm.toPrecision(8)}j  ` +
      `rel_err=${relErr.toExponential(3)}  [${status}]`
  );
}

// --- Case 1: alpha = 1.0 -> parallel RC ---
console.log("\n[Case 1] alpha = 1.0: Randles-CPE must reduce to standard parallel RC");
{
  const Rs = 2000.0,
    Rct = 50000.0,
    Q = 2.0e-7,
    alpha = 1.0;
  const C = Q;
  for (const f of [10.0, 1000.0, 50000.0]) {
    const omega = 2 * Math.PI * f;
    // Independent closed form: Z = Rs + Rct/(1 + j*omega*Rct*C)
    const denRe = 1;
    const denIm = omega * Rct * C;
    const denomSq = denRe * denRe + denIm * denIm;
    const expectedRe = Rs + (Rct * denRe) / denomSq;
    const expectedIm = (Rct * -denIm) / denomSq;
    const got = randlesCpeImpedance(f, Rs, Rct, Q, alpha);
    check(`f=${f} Hz`, got, expectedRe, expectedIm);
  }
  const zLow = randlesCpeImpedance(1e-3, Rs, Rct, Q, alpha);
  const zHigh = randlesCpeImpedance(1e9, Rs, Rct, Q, alpha);
  check("f->0 limit (expect Rs+Rct)", zLow, Rs + Rct, 0, 1e-3);
  check("f->inf limit (expect Rs)", zHigh, Rs, 0, 1e-3);
}

// --- Case 2: alpha = 0.5 -> Warburg-like ---
console.log("\n[Case 2] alpha = 0.5: CPE must reduce to Warburg-like closed form");
{
  const Rs = 5000.0,
    Q = 3.0e-8;
  for (const f of [1.0, 1000.0, 100000.0]) {
    const omega = 2 * Math.PI * f;
    // Z_CPE = 1/(Q*sqrt(j*omega)) = 1/(Q*sqrt(omega)) * (1-j)/sqrt(2)
    const mag = 1.0 / (Q * Math.sqrt(omega) * Math.sqrt(2));
    const expectedRe = mag;
    const expectedIm = -mag;
    const got = zCpe(f, Q, 0.5);
    check(`Z_CPE f=${f} Hz`, got, expectedRe, expectedIm);
    const phaseDeg = (Math.atan2(got.im, got.re) * 180) / Math.PI;
    const ok = Math.abs(phaseDeg - -45.0) < 1e-6;
    if (!ok) allOk = false;
    console.log(`      phase(Z_CPE) = ${phaseDeg.toFixed(6)} deg (expected exactly -45.0)  [${ok ? PASS : FAIL}]`);
  }
  const RctInf = 1e18;
  for (const f of [1.0, 10000.0]) {
    const omega = 2 * Math.PI * f;
    const mag = 1.0 / (Q * Math.sqrt(omega) * Math.sqrt(2));
    const got = randlesCpeImpedance(f, Rs, RctInf, Q, 0.5);
    check(`Rs+Z_CPE f=${f} Hz (Rct->inf)`, got, Rs + mag, -mag, 1e-4);
  }
}

// --- Case 3: degenerate-slice sample, shape checks ---
console.log("\n[Case 3] Degenerate-slice sample (same params as Phase 1 verification Case 3): Nyquist-shape check");
{
  // Same sample as verify_forward_model.py's Case 3 (rng seed 4242, first
  // draw from sample_degenerate_params in Phase 1).
  const Rs = 39330.3,
    Rct = 2.22138e6,
    Q = 1.03386e-7,
    alpha = 0.61869;
  console.log(`    params: Rs=${Rs}, Rct=${Rct}, Q=${Q}, alpha=${alpha}`);

  const reArr = [];
  const imArr = [];
  for (const f of FREQ_GRID_HZ) {
    const z = randlesCpeImpedance(f, Rs, Rct, Q, alpha);
    reArr.push(z.re);
    imArr.push(-z.im); // Nyquist convention: -Im(Z)
  }

  let monotonic = true;
  for (let i = 1; i < reArr.length; i++) {
    if (reArr[i] - reArr[i - 1] > 1e-6) monotonic = false;
  }
  console.log(`    Re(Z) monotonically non-increasing with frequency: [${monotonic ? PASS : FAIL}]`);
  if (!monotonic) allOk = false;

  const nonNegativeIm = imArr.every((v) => v >= -1e-9);
  console.log(`    -Im(Z) >= 0 at all frequencies: [${nonNegativeIm ? PASS : FAIL}]`);
  if (!nonNegativeIm) allOk = false;

  const endpointsOk = reArr[0] >= reArr[reArr.length - 1];
  console.log(`    Low-f Re(Z) >= high-f Re(Z): [${endpointsOk ? PASS : FAIL}]`);
  if (!endpointsOk) allOk = false;
}

console.log("\n" + "=".repeat(60));
console.log("JS FORWARD MODEL PORT VERIFICATION: " + (allOk ? "ALL PASS" : "SOME FAILED"));
console.log("=".repeat(60));

if (!allOk) process.exit(1);
