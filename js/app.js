import {
  FREQ_GRID_HZ,
  computeSpectrum,
  complexToMagPhaseDeg,
  paramsToTransformed,
} from "./forward_model.js";
import { lmRefine } from "./lm_refine.js";
import { classifyRctRegime, classifyResidual, checkFrequencyCoverage } from "./confidence.js";
import { configureOnnxRuntime, loadOnnxModel, runNnInference } from "./onnx_inference.js";
import { parseCsvText, convertToCanonical } from "./csv_parse.js";
import { renderNyquistPlot, renderBodePlot } from "./plots.js";

const PARAM_NAMES = ["Rs", "Rct", "Q", "alpha"];
const PARAM_UNITS = { Rs: "ohm", Rct: "ohm", Q: "S·s^α", alpha: "" };

let onnxSession = null;
let parsedUpload = null; // { freqHz, reOhm, imOhm } after Parse

// --------------------------- formatting helpers ---------------------------

function fmtParamValue(name, v) {
  if (name === "Rs" || name === "Rct") {
    if (!isFinite(v)) return String(v);
    return v.toLocaleString(undefined, { maximumFractionDigits: 0 }) + " Ω";
  }
  if (name === "Q") return v.toExponential(3) + " S·s^α";
  if (name === "alpha") return v.toFixed(4);
  return String(v);
}

function relDiffPct(a, b) {
  return (Math.abs(a - b) / Math.max(Math.abs(b), 1e-30)) * 100;
}

// ------------------------------ math helpers -------------------------------

function denseLogGrid(fMin, fMax, n = 80) {
  const lo = Math.log10(fMin),
    hi = Math.log10(fMax);
  return Array.from({ length: n }, (_, i) => Math.pow(10, lo + ((hi - lo) * i) / (n - 1)));
}

// Interpolates an arbitrary measured spectrum onto the canonical 20-point
// grid the NN expects, in log-frequency space (piecewise-linear on Re/Im).
// Points outside the measured range are clamped to the nearest boundary
// value rather than extrapolated. Task 4's coverage warning is the
// explicit signal for "this region wasn't actually measured"; this
// function just needs to not fabricate a trend past the data.
function interpolateToCanonicalGrid(freqHz, reOhm, imOhm) {
  const idx = freqHz.map((_, i) => i).sort((a, b) => freqHz[a] - freqHz[b]);
  const fSorted = idx.map((i) => freqHz[i]);
  const reSorted = idx.map((i) => reOhm[i]);
  const imSorted = idx.map((i) => imOhm[i]);
  const logF = fSorted.map((f) => Math.log10(f));

  function interpAt(targetF) {
    const lf = Math.log10(targetF);
    if (lf <= logF[0]) return { re: reSorted[0], im: imSorted[0] };
    if (lf >= logF[logF.length - 1]) return { re: reSorted.at(-1), im: imSorted.at(-1) };
    let j = 0;
    while (j < logF.length - 2 && logF[j + 1] < lf) j++;
    const t = (lf - logF[j]) / (logF[j + 1] - logF[j] || 1);
    return {
      re: reSorted[j] + t * (reSorted[j + 1] - reSorted[j]),
      im: imSorted[j] + t * (imSorted[j + 1] - imSorted[j]),
    };
  }

  const re = [],
    im = [];
  for (const f of FREQ_GRID_HZ) {
    const p = interpAt(f);
    re.push(p.re);
    im.push(p.im);
  }
  return { re, im };
}

function gaussianNoise() {
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function applyNoise(re, im, magPct, phaseDeg) {
  const outRe = [],
    outIm = [];
  for (let i = 0; i < re.length; i++) {
    const { magnitude, phaseDeg: ph } = complexToMagPhaseDeg(re[i], im[i]);
    const noisyMag = magnitude * (1 + (gaussianNoise() * magPct) / 100);
    const noisyPhase = ph + gaussianNoise() * phaseDeg;
    const rad = (noisyPhase * Math.PI) / 180;
    outRe.push(noisyMag * Math.cos(rad));
    outIm.push(noisyMag * Math.sin(rad));
  }
  return { re: outRe, im: outIm };
}

// ------------------------------ core pipeline -------------------------------

/**
 * Runs NN inference + LM refinement + all confidence checks on an arbitrary
 * (freqHz, reOhm, imOhm) spectrum. This is the single code path both
 * Synthesize and Upload modes call.
 */
async function runPipeline(freqHz, reOhm, imOhm) {
  const freqCoverage = checkFrequencyCoverage(freqHz);

  const nnGridSpectrum = interpolateToCanonicalGrid(freqHz, reOhm, imOhm);
  const nnRaw = await runNnInference(onnxSession, nnGridSpectrum.re, nnGridSpectrum.im);
  const nn = { Rs: nnRaw.Rs, Rct: nnRaw.Rct, Q: nnRaw.Q, alpha: nnRaw.alpha };

  const regimeFlag = classifyRctRegime(nn.Rct); // Task 1a: uses the NN's *initial* Rct estimate

  const initTheta = paramsToTransformed(nn.Rs, nn.Rct, nn.Q, nn.alpha);
  const lm = lmRefine(initTheta, freqHz, reOhm, imOhm);
  const refined = { Rs: lm.params[0], Rct: lm.params[1], Q: lm.params[2], alpha: lm.params[3] };

  const residualConfidence = classifyResidual(lm.weightedRmsResidual);

  const divergence = {};
  let anyDiverges = false;
  for (const p of PARAM_NAMES) {
    const pct = relDiffPct(nn[p], refined[p]);
    divergence[p] = pct;
    if (pct > 20) anyDiverges = true;
  }

  return {
    freqHz,
    reOhm,
    imOhm,
    nn,
    refined,
    lm,
    regimeFlag,
    residualConfidence,
    freqCoverage,
    divergence,
    anyDiverges,
  };
}

// ------------------------------- rendering ----------------------------------

// No color is used anywhere on this page, so severity (ok/caution/warning)
// is spelled out in words here, paired in the CSS with border weight and
// fill opacity. It's never conveyed by hue alone.
const SEVERITY_LABEL = { ok: "OK", warn: "CAUTION", bad: "WARNING" };

function flagBoxHtml(level, title, message) {
  const cls = level === "ok" ? "flag-ok" : level === "warn" ? "flag-warn" : "flag-bad";
  return `<div class="flag-box ${cls}"><strong>${SEVERITY_LABEL[level]}: ${title}</strong>${message}</div>`;
}

function buildResultsHtml(result, groundTruth) {
  const { nn, refined, residualConfidence, regimeFlag, freqCoverage, divergence, anyDiverges, lm } = result;

  let html = "<h2>Result</h2>";

  // Frequency-coverage warnings (Task 4). These fire independently of the
  // regime flag below; either, both, or neither can appear.
  for (const w of freqCoverage.warnings) {
    html += flagBoxHtml(
      "warn",
      `Frequency coverage: ${w.affectedParams.join(", ")} may be underdetermined`,
      w.message
    );
  }

  // Rct-regime flag (Task 1a), independent of the coverage warnings above.
  if (regimeFlag.regime !== "none") {
    html += flagBoxHtml(
      "warn",
      `Regime flag (from NN's initial Rct estimate = ${fmtParamValue("Rct", nn.Rct)})`,
      regimeFlag.message
    );
  }

  // General residual confidence (Task 1)
  html += flagBoxHtml(
    residualConfidence.level === "low" ? "ok" : residualConfidence.level === "moderate" ? "warn" : "bad",
    `${residualConfidence.label} (weighted RMS = ${lm.weightedRmsResidual.toExponential(3)})`,
    residualConfidence.message
  );

  if (anyDiverges) {
    html += flagBoxHtml(
      "warn",
      "NN estimate and refined fit diverge",
      "The network's quick first guess and the slower, more careful fine-tuning pass landed on noticeably " +
        "different answers for at least one number below. That gap is itself worth noticing: the neural " +
        "network estimate and the post-refinement fit disagree by more than 20% on at least one parameter " +
        "(see the table below)."
    );
  }

  html += '<p class="explain">';
  if (groundTruth) {
    html += "<strong>Ground truth</strong> is what you actually set with the sliders. ";
  }
  html +=
    "<strong>NN-only</strong> is the neural network's fast first guess. <strong>NN + refined</strong> is " +
    "that guess fine-tuned by a slower, more precise math pass that adjusts it to match the spectrum as " +
    "closely as possible; treat this column as the tool's real answer. <strong>NN vs refined</strong> is " +
    "how much that fine-tuning changed the first guess, as a percentage. A big change here is itself " +
    "worth noticing (see above).</p>";

  html += '<table class="data-table"><thead><tr><th>Parameter</th>';
  if (groundTruth) html += "<th>Ground truth</th>";
  html += "<th>NN-only</th><th>NN + refined</th><th>NN vs refined</th></tr></thead><tbody>";
  for (const p of PARAM_NAMES) {
    html += `<tr><td>${p}${PARAM_UNITS[p] ? ` (${PARAM_UNITS[p]})` : ""}</td>`;
    if (groundTruth) html += `<td>${fmtParamValue(p, groundTruth[p])}</td>`;
    html += `<td>${fmtParamValue(p, nn[p])}</td>`;
    html += `<td>${fmtParamValue(p, refined[p])}</td>`;
    html += `<td>${divergence[p].toFixed(1)}%</td>`;
    html += "</tr>";
  }
  html += "</tbody></table>";

  html += `<p class="hint">LM refinement: ${lm.iterations} iterations, ${
    lm.converged ? "converged" : "stopped (max iterations)"
  }.</p>`;

  return html;
}

function renderPlots(nyquistEl, bodeEl, result) {
  const { freqHz, reOhm, imOhm, refined } = result;
  const gridForFit = denseLogGrid(Math.min(...freqHz), Math.max(...freqHz), 80);
  const fit = computeSpectrum(refined.Rs, refined.Rct, refined.Q, refined.alpha, gridForFit);

  renderNyquistPlot(nyquistEl, {
    measured: { re: reOhm, im: imOhm },
    reconstructed: { re: fit.re, im: fit.im },
  });

  const measuredMagPhase = freqHz.map((_, i) => complexToMagPhaseDeg(reOhm[i], imOhm[i]));
  const fitMagPhase = gridForFit.map((_, i) => complexToMagPhaseDeg(fit.re[i], fit.im[i]));

  renderBodePlot(bodeEl, {
    measured: {
      freq: freqHz,
      mag: measuredMagPhase.map((m) => m.magnitude),
      phaseDeg: measuredMagPhase.map((m) => m.phaseDeg),
    },
    reconstructed: {
      freq: gridForFit,
      mag: fitMagPhase.map((m) => m.magnitude),
      phaseDeg: fitMagPhase.map((m) => m.phaseDeg),
    },
  });
}

// ------------------------------ synth mode ----------------------------------

const sliderIds = { Rs: "slider-rs", Rct: "slider-rct", Q: "slider-q", alpha: "slider-alpha" };

function readSynthParams() {
  const logRs = Number(document.getElementById(sliderIds.Rs).value);
  const logRct = Number(document.getElementById(sliderIds.Rct).value);
  const logQ = Number(document.getElementById(sliderIds.Q).value);
  const alpha = Number(document.getElementById(sliderIds.alpha).value);
  return { Rs: 10 ** logRs, Rct: 10 ** logRct, Q: 10 ** logQ, alpha };
}

function updateSliderOutputs() {
  const p = readSynthParams();
  document.getElementById("slider-rs-out").textContent = fmtParamValue("Rs", p.Rs);
  document.getElementById("slider-rct-out").textContent = fmtParamValue("Rct", p.Rct);
  document.getElementById("slider-q-out").textContent = fmtParamValue("Q", p.Q);
  document.getElementById("slider-alpha-out").textContent = p.alpha.toFixed(4);
}

function setSliders(params) {
  document.getElementById(sliderIds.Rs).value = Math.log10(params.Rs);
  document.getElementById(sliderIds.Rct).value = Math.log10(params.Rct);
  document.getElementById(sliderIds.Q).value = Math.log10(params.Q);
  document.getElementById(sliderIds.alpha).value = Math.min(params.alpha, 0.999);
  updateSliderOutputs();
}

const PRESETS = {
  case1: { Rs: 2000, Rct: 50000, Q: 2.0e-7, alpha: 0.999 },
  case2: { Rs: 5000, Rct: 1.0e7, Q: 3.0e-8, alpha: 0.5 },
  case3: { Rs: 39330.3, Rct: 2221380, Q: 1.03386e-7, alpha: 0.61869 },
};

let synthRunToken = 0;
async function runSynth() {
  const token = ++synthRunToken;
  const groundTruth = readSynthParams();
  const clean = computeSpectrum(groundTruth.Rs, groundTruth.Rct, groundTruth.Q, groundTruth.alpha, FREQ_GRID_HZ);

  let re = clean.re,
    im = clean.im;
  const noiseOn = document.getElementById("noise-toggle").checked;
  if (noiseOn) {
    const magPct = Number(document.getElementById("slider-noise-mag").value);
    const phaseDeg = Number(document.getElementById("slider-noise-phase").value);
    ({ re, im } = applyNoise(re, im, magPct, phaseDeg));
  }

  const resultsEl = document.getElementById("synth-results");
  resultsEl.innerHTML = "<p class='hint'>Running NN + refinement…</p>";
  const result = await runPipeline(FREQ_GRID_HZ, re, im);
  if (token !== synthRunToken) return; // a newer run superseded this one

  resultsEl.innerHTML = buildResultsHtml(result, groundTruth);
  renderPlots(document.getElementById("synth-nyquist"), document.getElementById("synth-bode"), result);
}

function wireSynthControls() {
  for (const id of Object.values(sliderIds)) {
    document.getElementById(id).addEventListener("input", () => {
      updateSliderOutputs();
    });
    document.getElementById(id).addEventListener("change", () => runSynth());
  }
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      setSliders(PRESETS[btn.dataset.preset]);
      runSynth();
    });
  });
  const noiseToggle = document.getElementById("noise-toggle");
  noiseToggle.addEventListener("change", () => {
    document.getElementById("noise-mag-row").hidden = !noiseToggle.checked;
    document.getElementById("noise-phase-row").hidden = !noiseToggle.checked;
    runSynth();
  });
  document.getElementById("slider-noise-mag").addEventListener("input", (e) => {
    document.getElementById("slider-noise-mag-out").textContent = Number(e.target.value).toFixed(1) + "%";
  });
  document.getElementById("slider-noise-mag").addEventListener("change", () => runSynth());
  document.getElementById("slider-noise-phase").addEventListener("input", (e) => {
    document.getElementById("slider-noise-phase-out").textContent = Number(e.target.value).toFixed(2) + "deg";
  });
  document.getElementById("slider-noise-phase").addEventListener("change", () => runSynth());
}

// ------------------------------ upload mode ---------------------------------

function wireUploadControls() {
  const reprSelect = document.getElementById("unit-repr");
  reprSelect.addEventListener("change", () => {
    document.getElementById("unit-phase-field").hidden = reprSelect.value !== "magphase";
  });

  document.getElementById("csv-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById("csv-text").value = await file.text();
  });

  document.getElementById("parse-btn").addEventListener("click", () => {
    const text = document.getElementById("csv-text").value;
    const { rows, errors } = parseCsvText(text);
    const opts = {
      freqUnit: document.getElementById("unit-freq").value,
      representation: document.getElementById("unit-repr").value,
      phaseUnit: document.getElementById("unit-phase").value,
    };
    const preview = document.getElementById("parsed-preview");
    const coverageEl = document.getElementById("coverage-warnings");
    const runBtn = document.getElementById("run-upload-btn");

    if (rows.length === 0) {
      preview.innerHTML = "<p class='hint'>No valid rows parsed.</p>";
      coverageEl.innerHTML = "";
      runBtn.disabled = true;
      parsedUpload = null;
      return;
    }

    const canonical = convertToCanonical(rows, opts);
    parsedUpload = canonical;

    let html = `<h3>Parsed values (${rows.length} rows, canonical units: Hz, ohm)</h3>`;
    if (errors.length) html += `<p class="hint">${errors.join("<br>")}</p>`;
    html +=
      '<table class="data-table"><thead><tr><th>freq (Hz)</th><th>Re(Z) (ohm)</th><th>Im(Z) (ohm)</th></tr></thead><tbody>';
    for (let i = 0; i < canonical.freqHz.length; i++) {
      html += `<tr><td>${canonical.freqHz[i].toPrecision(6)}</td><td>${canonical.reOhm[i].toPrecision(
        6
      )}</td><td>${canonical.imOhm[i].toPrecision(6)}</td></tr>`;
    }
    html += "</tbody></table>";
    preview.innerHTML = html;

    const coverage = checkFrequencyCoverage(canonical.freqHz);
    coverageEl.innerHTML = coverage.warnings
      .map((w) =>
        flagBoxHtml("warn", `Frequency coverage: ${w.affectedParams.join(", ")} may be underdetermined`, w.message)
      )
      .join("");
    if (coverage.warnings.length === 0) {
      coverageEl.innerHTML = flagBoxHtml(
        "ok",
        "Frequency coverage looks adequate",
        `This test swept both fast enough and slowly enough to give every number a fair chance of being ` +
          `pinned down. It spans ${coverage.decadesSpanned.toFixed(
            2
          )} decades (${coverage.minFreq.toPrecision(3)} Hz to ${coverage.maxFreq.toPrecision(3)} Hz).`
      );
    }

    runBtn.disabled = false;
  });

  document.getElementById("run-upload-btn").addEventListener("click", async () => {
    if (!parsedUpload) return;
    const resultsEl = document.getElementById("upload-results");
    resultsEl.innerHTML = "<p class='hint'>Running NN + refinement…</p>";
    const result = await runPipeline(parsedUpload.freqHz, parsedUpload.reOhm, parsedUpload.imOhm);
    resultsEl.innerHTML = buildResultsHtml(result, null);
    renderPlots(document.getElementById("upload-nyquist"), document.getElementById("upload-bode"), result);
  });
}

// -------------------------------- tabs & init --------------------------------

function wireTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });
      document.getElementById("synth-panel").hidden = btn.dataset.mode !== "synth";
      document.getElementById("upload-panel").hidden = btn.dataset.mode !== "upload";
    });
  });
}

async function init() {
  wireTabs();
  wireSynthControls();
  wireUploadControls();

  setSliders(PRESETS.case3); // default: a hand-checked case, so loading the page is itself a regression test

  const statusEl = document.getElementById("model-status");
  try {
    configureOnnxRuntime(new URL("vendor/onnxruntime-web/", document.baseURI).href);
    onnxSession = await loadOnnxModel("assets/model.onnx");
    statusEl.textContent = `Model loaded (inputs: ${onnxSession.inputNames.join(
      ", "
    )}; outputs: ${onnxSession.outputNames.join(", ")}).`;
  } catch (err) {
    statusEl.textContent = "Failed to load ONNX model: " + err.message;
    console.error(err);
    return;
  }

  await runSynth();
}

init();
