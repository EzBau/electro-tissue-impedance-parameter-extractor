// Levenberg-Marquardt local refinement of Randles-CPE parameters, seeded by
// the NN's estimate. Operates in the same transformed parameter space the
// NN was trained on (log10 Rs, log10 Rct, log10 Q, logit alpha) so that
// positivity/range constraints (Rs,Rct,Q > 0, 0 < alpha < 1) are enforced
// automatically by the parameterization rather than by explicit bounds.
//
// Residuals are modulus-weighted: residual_i = (predicted - measured) /
// |measured_i|, for both Re and Im at each frequency. This is the standard
// "modulus weighting" used in EIS fitting software (ZView, etc). It keeps
// a spectrum's high-Rct low-frequency points (huge magnitude) from
// swamping the fit relative to its high-frequency points (small
// magnitude), which would otherwise happen with unweighted least squares
// given the multi-decade magnitude spread typical of these spectra.

import { computeSpectrum, paramsToTransformed, transformedToParams } from "./forward_model.js";

const EPS_WEIGHT = 1e-9; // avoids divide-by-zero if a measured point is exactly 0

// Box constraints on the transformed parameters, applied by clipping after
// each accepted step. Without this, an insensitive or near-singular
// direction (a bad initial guess combined with a genuinely
// under-determined parameter) can let the raw Gauss-Newton/LM step wander
// far outside any physically meaningful region: testing without bounds
// produced an Rct that diverged to about 1e15 ohm. Bounds are set with
// about a decade of slack beyond the Phase 1 training range on each side
// (Rs 1e3-5e4, Rct 1e4-1e7, Q 1e-9-1e-6, alpha 0.5-1.0), so the refinement
// can still move slightly past the trained domain for real data without
// being able to escape to nonsensical values. This is a simple
// clip-projection, not a rigorous constrained optimizer, but it's enough
// to keep a 4-parameter local refinement well-behaved.
const DEFAULT_BOUNDS = {
  logRs: [Math.log10(2e2), Math.log10(2e5)],
  logRct: [Math.log10(1e3), Math.log10(1e8)],
  logQ: [Math.log10(1e-10), Math.log10(1e-5)],
  logitAlpha: [Math.log(0.3 / 0.7), Math.log(0.9995 / 0.0005)],
};

function clipToBounds(theta) {
  const b = DEFAULT_BOUNDS;
  return [
    Math.min(Math.max(theta[0], b.logRs[0]), b.logRs[1]),
    Math.min(Math.max(theta[1], b.logRct[0]), b.logRct[1]),
    Math.min(Math.max(theta[2], b.logQ[0]), b.logQ[1]),
    Math.min(Math.max(theta[3], b.logitAlpha[0]), b.logitAlpha[1]),
  ];
}

function residualVector(theta, freqArray, measuredRe, measuredIm) {
  const [Rs, Rct, Q, alpha] = transformedToParams(theta[0], theta[1], theta[2], theta[3]);
  const { re: predRe, im: predIm } = computeSpectrum(Rs, Rct, Q, alpha, freqArray);
  const r = new Array(2 * freqArray.length);
  for (let i = 0; i < freqArray.length; i++) {
    const w = 1.0 / (Math.hypot(measuredRe[i], measuredIm[i]) + EPS_WEIGHT);
    r[2 * i] = (predRe[i] - measuredRe[i]) * w;
    r[2 * i + 1] = (predIm[i] - measuredIm[i]) * w;
  }
  return r;
}

function costOf(r) {
  let s = 0;
  for (const v of r) s += v * v;
  return 0.5 * s;
}

// Numeric Jacobian via forward differences. theta: number[4]. Returns a
// (2*F) x 4 array of arrays.
function numericJacobian(theta, freqArray, measuredRe, measuredIm, r0) {
  const nRes = r0.length;
  const nParam = theta.length;
  const J = Array.from({ length: nRes }, () => new Array(nParam));
  for (let j = 0; j < nParam; j++) {
    const h = Math.max(1e-6, Math.abs(theta[j]) * 1e-5);
    const thetaPerturbed = theta.slice();
    thetaPerturbed[j] += h;
    const rPerturbed = residualVector(thetaPerturbed, freqArray, measuredRe, measuredIm);
    for (let i = 0; i < nRes; i++) {
      J[i][j] = (rPerturbed[i] - r0[i]) / h;
    }
  }
  return J;
}

function matMulTransposeSelf(J) {
  // Returns J^T J (nParam x nParam) and J^T r is computed separately.
  const nRes = J.length;
  const nParam = J[0].length;
  const JTJ = Array.from({ length: nParam }, () => new Array(nParam).fill(0));
  for (let a = 0; a < nParam; a++) {
    for (let b = 0; b < nParam; b++) {
      let s = 0;
      for (let i = 0; i < nRes; i++) s += J[i][a] * J[i][b];
      JTJ[a][b] = s;
    }
  }
  return JTJ;
}

function matVecTranspose(J, r) {
  const nRes = J.length;
  const nParam = J[0].length;
  const out = new Array(nParam).fill(0);
  for (let a = 0; a < nParam; a++) {
    let s = 0;
    for (let i = 0; i < nRes; i++) s += J[i][a] * r[i];
    out[a] = s;
  }
  return out;
}

// Solve A x = b for small dense A (Gaussian elimination, partial pivoting).
function solveLinearSystem(A, b) {
  const n = b.length;
  const M = A.map((row) => row.slice());
  const rhs = b.slice();
  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxAbs = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > maxAbs) {
        maxAbs = Math.abs(M[r][col]);
        pivotRow = r;
      }
    }
    if (pivotRow !== col) {
      [M[col], M[pivotRow]] = [M[pivotRow], M[col]];
      [rhs[col], rhs[pivotRow]] = [rhs[pivotRow], rhs[col]];
    }
    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-300) continue; // singular-ish; leave row as-is
    for (let r = col + 1; r < n; r++) {
      const factor = M[r][col] / pivot;
      for (let c = col; c < n; c++) M[r][c] -= factor * M[col][c];
      rhs[r] -= factor * rhs[col];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = rhs[r];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = Math.abs(M[r][r]) < 1e-300 ? 0 : s / M[r][r];
  }
  return x;
}

/**
 * Levenberg-Marquardt refinement.
 * @param {number[]} initialTheta [log10 Rs, log10 Rct, log10 Q, logit alpha]
 * @param {number[]} freqArray frequencies (Hz)
 * @param {number[]} measuredRe measured Re(Z), same length as freqArray
 * @param {number[]} measuredIm measured Im(Z), same length as freqArray
 * @param {object} [opts]
 * @returns {{theta:number[], params:number[], cost:number, weightedRmsResidual:number,
 *            iterations:number, converged:boolean}}
 */
export function lmRefine(initialTheta, freqArray, measuredRe, measuredIm, opts = {}) {
  const maxIterations = opts.maxIterations ?? 60;
  const costTol = opts.costTol ?? 1e-10;
  const stepTol = opts.stepTol ?? 1e-8;
  // Trust-region-style safeguard: transformed-space parameters are
  // log10/logit, so a step of this size already means a >=10x change in a
  // raw parameter. Without a cap, a near-singular JTJ (an insensitive
  // direction, e.g. Rct deep in the degenerate high-Rct regime) can produce
  // an enormous raw Gauss-Newton/LM step that the exponential
  // reparameterization then blows up into a nonsensical raw value.
  const maxStepPerIter = opts.maxStepPerIter ?? 1.0;

  let theta = clipToBounds(initialTheta.slice());
  let r = residualVector(theta, freqArray, measuredRe, measuredIm);
  let cost = costOf(r);
  let lambda = 1e-2;
  let converged = false;
  let iterations = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;
    const J = numericJacobian(theta, freqArray, measuredRe, measuredIm, r);
    const JTJ = matMulTransposeSelf(J);
    const JTr = matVecTranspose(J, r);
    const maxDiag = Math.max(...JTJ.map((row, i) => row[i]), 1e-8);

    let improved = false;
    for (let attempt = 0; attempt < 12; attempt++) {
      const A = JTJ.map((row, i) =>
        row.map((v, j) => (i === j ? v + lambda * Math.max(v, 1e-6 * maxDiag) : v))
      );
      const negJTr = JTr.map((v) => -v);
      let delta = solveLinearSystem(A, negJTr);

      // Clip the step to the trust-region radius (per-component, then
      // rescale the whole vector so direction is preserved).
      const stepNorm = Math.hypot(...delta);
      if (stepNorm > maxStepPerIter) {
        const scale = maxStepPerIter / stepNorm;
        delta = delta.map((v) => v * scale);
      }

      const thetaNew = clipToBounds(theta.map((v, i) => v + delta[i]));
      const rNew = residualVector(thetaNew, freqArray, measuredRe, measuredIm);
      const costNew = costOf(rNew);

      if (costNew < cost) {
        const stepSize = Math.hypot(...delta);
        const costDrop = cost - costNew;
        theta = thetaNew;
        r = rNew;
        cost = costNew;
        lambda = Math.max(lambda / 10, 1e-12);
        improved = true;
        if (costDrop < costTol || stepSize < stepTol) converged = true;
        break;
      } else {
        lambda *= 10;
      }
    }
    if (!improved || converged) break;
  }

  const params = transformedToParams(theta[0], theta[1], theta[2], theta[3]);
  const weightedRmsResidual = Math.sqrt((2 * cost) / r.length);
  return { theta, params, cost, weightedRmsResidual, iterations, converged };
}

export { residualVector, costOf };
