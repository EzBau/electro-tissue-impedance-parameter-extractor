// Hand-rolled SVG Nyquist + Bode plots. No chart library. Both plots take
// {re,im} spectra computed by forward_model.js elsewhere on the page (the
// caller passes the arrays in; this module only draws them), so there is a
// single code path for "what the spectrum values are" shared between the
// fit and the plot.

const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, children = []) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const c of children) e.appendChild(c);
  return e;
}

function linearScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

function niceLinearTicks(min, max, targetCount = 5) {
  if (min === max) return [min];
  const span = max - min;
  const rawStep = span / targetCount;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const residual = rawStep / magnitude;
  let step;
  if (residual > 5) step = 10 * magnitude;
  else if (residual > 2) step = 5 * magnitude;
  else if (residual > 1) step = 2 * magnitude;
  else step = magnitude;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + 1e-9 * span; v += step) ticks.push(Number(v.toPrecision(10)));
  return ticks;
}

function decadeTicks(min, max) {
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  const ticks = [];
  for (let e = lo; e <= hi; e++) ticks.push(Math.pow(10, e));
  return ticks.filter((t) => t >= min * 0.999 && t <= max * 1.001);
}

function fmtEngineering(v) {
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1e6) return (v / 1e6).toPrecision(3) + "M";
  if (abs >= 1e3) return (v / 1e3).toPrecision(3) + "k";
  if (abs >= 1) return v.toPrecision(3);
  if (abs >= 1e-3) return (v * 1e3).toPrecision(3) + "m";
  if (abs >= 1e-6) return (v * 1e6).toPrecision(3) + "u";
  if (abs >= 1e-9) return (v * 1e9).toPrecision(3) + "n";
  return v.toExponential(2);
}

function drawSeries(
  svgGroup,
  xs,
  ys,
  xScale,
  yScale,
  { color, drawLine = true, drawPoints = true, radius = 3, dashed = false }
) {
  // No color is used anywhere on this page, so measured vs. reconstructed
  // must be told apart by shape alone: measured is solid filled dots,
  // reconstructed is a dashed line (in addition to the dot-vs-line
  // distinction already in play).
  if (drawLine && xs.length > 1) {
    const d = xs.map((x, i) => `${i === 0 ? "M" : "L"} ${xScale(x)} ${yScale(ys[i])}`).join(" ");
    const attrs = { d, fill: "none", stroke: color, "stroke-width": 1.75, opacity: 0.9 };
    if (dashed) attrs["stroke-dasharray"] = "5 4";
    svgGroup.appendChild(el("path", attrs));
  }
  if (drawPoints) {
    for (let i = 0; i < xs.length; i++) {
      svgGroup.appendChild(
        el("circle", { cx: xScale(xs[i]), cy: yScale(ys[i]), r: radius, fill: color, opacity: 0.9 })
      );
    }
  }
}

/**
 * @param {HTMLElement} container
 * @param {{measured?: {re:number[], im:number[]}, reconstructed?: {re:number[], im:number[]}}} data
 */
export function renderNyquistPlot(container, data) {
  container.innerHTML = "";
  const W = 480,
    H = 460,
    pad = { l: 60, r: 20, t: 20, b: 50 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;

  const allRe = [];
  const allNegIm = [];
  for (const key of ["measured", "reconstructed"]) {
    const s = data[key];
    if (!s) continue;
    s.re.forEach((v) => allRe.push(v));
    s.im.forEach((v) => allNegIm.push(-v));
  }
  if (allRe.length === 0) return;

  const reMin = Math.min(0, ...allRe);
  const reMax = Math.max(...allRe);
  const imMin = Math.min(0, ...allNegIm);
  const imMax = Math.max(...allNegIm);
  // Equal aspect ratio so circular arcs aren't visually distorted.
  const span = Math.max(reMax - reMin, imMax - imMin) * 1.1 || 1;
  const reCenter = (reMax + reMin) / 2;
  const imCenter = (imMax + imMin) / 2;

  const xScale = linearScale([reCenter - span / 2, reCenter + span / 2], [pad.l, pad.l + plotW]);
  const yScale = linearScale([imCenter - span / 2, imCenter + span / 2], [pad.t + plotH, pad.t]); // flip y

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });

  // grid + axes
  const ticks = niceLinearTicks(reCenter - span / 2, reCenter + span / 2, 5);
  const gridGroup = el("g");
  for (const t of ticks) {
    gridGroup.appendChild(
      el("line", { x1: xScale(t), y1: pad.t, x2: xScale(t), y2: pad.t + plotH, stroke: "var(--grid-color)" })
    );
    gridGroup.appendChild(
      el("line", { x1: pad.l, y1: yScale(t), x2: pad.l + plotW, y2: yScale(t), stroke: "var(--grid-color)" })
    );
    const xLabel = el("text", { x: xScale(t), y: pad.t + plotH + 16, "text-anchor": "middle", class: "axis-label" });
    xLabel.textContent = fmtEngineering(t);
    gridGroup.appendChild(xLabel);
    const yLabel = el("text", { x: pad.l - 8, y: yScale(t) + 4, "text-anchor": "end", class: "axis-label" });
    yLabel.textContent = fmtEngineering(t);
    gridGroup.appendChild(yLabel);
  }
  svg.appendChild(gridGroup);
  const xTitle = el("text", { x: pad.l + plotW / 2, y: H - 8, "text-anchor": "middle", class: "axis-title" });
  xTitle.textContent = "Re(Z) [ohm]";
  svg.appendChild(xTitle);
  const yTitle = el("text", {
    x: 14,
    y: pad.t + plotH / 2,
    "text-anchor": "middle",
    class: "axis-title",
    transform: `rotate(-90 14 ${pad.t + plotH / 2})`,
  });
  yTitle.textContent = "-Im(Z) [ohm]";
  svg.appendChild(yTitle);

  const seriesGroup = el("g");
  if (data.measured) {
    drawSeries(seriesGroup, data.measured.re, data.measured.im.map((v) => -v), xScale, yScale, {
      color: "var(--series-measured)",
      drawLine: false,
      drawPoints: true,
      radius: 4,
    });
  }
  if (data.reconstructed) {
    drawSeries(seriesGroup, data.reconstructed.re, data.reconstructed.im.map((v) => -v), xScale, yScale, {
      color: "var(--series-fit)",
      drawLine: true,
      drawPoints: false,
      dashed: true,
    });
  }
  svg.appendChild(seriesGroup);
  container.appendChild(svg);
}

/**
 * @param {HTMLElement} container
 * @param {{measured?:{freq:number[],mag:number[],phaseDeg:number[]}, reconstructed?:{freq:number[],mag:number[],phaseDeg:number[]}}} data
 *   Each series carries its own frequency array (measured = sparse actual
 *   points, reconstructed = a dense smooth curve) rather than sharing one.
 */
export function renderBodePlot(container, data) {
  container.innerHTML = "";
  const W = 480,
    panelH = 190,
    gap = 30,
    pad = { l: 60, r: 20, t: 14, b: 30 };
  const plotW = W - pad.l - pad.r;
  const H = pad.t + panelH + gap + panelH + pad.b;

  const allFreq = [];
  const allMag = [];
  const allPhase = [];
  for (const key of ["measured", "reconstructed"]) {
    const s = data[key];
    if (!s) continue;
    s.freq.forEach((v) => allFreq.push(v));
    s.mag.forEach((v) => allMag.push(v));
    s.phaseDeg.forEach((v) => allPhase.push(v));
  }
  const fMin = Math.min(...allFreq),
    fMax = Math.max(...allFreq);
  const xScale = linearScale([Math.log10(fMin), Math.log10(fMax)], [pad.l, pad.l + plotW]);
  const magMin = Math.min(...allMag),
    magMax = Math.max(...allMag);
  const logMagMin = Math.log10(Math.max(magMin, 1e-12)) - 0.05;
  const logMagMax = Math.log10(magMax) + 0.05;
  const phaseMin = Math.min(-5, ...allPhase) - 5;
  const phaseMax = Math.max(5, ...allPhase) + 5;

  const yMagScale = linearScale([logMagMin, logMagMax], [pad.t + panelH, pad.t]);
  const yPhaseScale = linearScale(
    [phaseMin, phaseMax],
    [pad.t + panelH + gap + panelH, pad.t + panelH + gap]
  );

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });

  function drawPanel(yScale, yTicks, yFmt, panelTop, title, valuesKey, getY) {
    const g = el("g");
    for (const t of decadeTicks(fMin, fMax)) {
      g.appendChild(
        el("line", {
          x1: xScale(Math.log10(t)),
          y1: panelTop,
          x2: xScale(Math.log10(t)),
          y2: panelTop + panelH,
          stroke: "var(--grid-color)",
        })
      );
      const label = el("text", {
        x: xScale(Math.log10(t)),
        y: panelTop + panelH + 14,
        "text-anchor": "middle",
        class: "axis-label",
      });
      label.textContent = t >= 1000 ? fmtEngineering(t) : String(t);
      g.appendChild(label);
    }
    for (const t of yTicks) {
      g.appendChild(
        el("line", { x1: pad.l, y1: yScale(t), x2: pad.l + plotW, y2: yScale(t), stroke: "var(--grid-color)" })
      );
      const label = el("text", { x: pad.l - 8, y: yScale(t) + 4, "text-anchor": "end", class: "axis-label" });
      label.textContent = yFmt(t);
      g.appendChild(label);
    }
    const titleEl = el("text", { x: pad.l, y: panelTop - 2, class: "axis-title" });
    titleEl.textContent = title;
    g.appendChild(titleEl);

    if (data.measured) {
      drawSeries(
        g,
        data.measured.freq.map((f) => Math.log10(f)),
        data.measured[valuesKey].map(getY),
        xScale,
        yScale,
        { color: "var(--series-measured)", drawLine: false, drawPoints: true, radius: 3.5 }
      );
    }
    if (data.reconstructed) {
      drawSeries(
        g,
        data.reconstructed.freq.map((f) => Math.log10(f)),
        data.reconstructed[valuesKey].map(getY),
        xScale,
        yScale,
        { color: "var(--series-fit)", drawLine: true, drawPoints: false, dashed: true }
      );
    }
    return g;
  }

  const magTicks = decadeTicks(Math.pow(10, logMagMin), Math.pow(10, logMagMax));
  svg.appendChild(
    drawPanel(
      (logv) => yMagScale(Math.log10(logv)),
      magTicks,
      fmtEngineering,
      pad.t,
      "|Z| [ohm] vs frequency [Hz] (log-log)",
      "mag",
      (v) => v
    )
  );
  svg.appendChild(
    drawPanel(
      yPhaseScale,
      niceLinearTicks(phaseMin, phaseMax, 5),
      (v) => v.toFixed(0) + "deg",
      pad.t + panelH + gap,
      "phase [deg] vs frequency [Hz] (semi-log)",
      "phaseDeg",
      (v) => v
    )
  );
  const xAxisTitle = el("text", { x: pad.l + plotW / 2, y: H - 6, "text-anchor": "middle", class: "axis-title" });
  xAxisTitle.textContent = "frequency [Hz]";
  svg.appendChild(xAxisTitle);

  container.appendChild(svg);
}
