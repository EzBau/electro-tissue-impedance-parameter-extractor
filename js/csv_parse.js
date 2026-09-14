// CSV parsing + explicit unit conversion for upload mode (Task 3).
// Deliberately does NOT try to auto-detect units. The caller must pass the
// user's explicit selection; parseCsvText only tokenizes, and never
// guesses what a column means.

/**
 * Parses a 3-column CSV/TSV-ish text block (freq, col1, col2). Skips blank
 * lines and a single optional header row (detected by its first cell not
 * being parseable as a number). Splits on commas; falls back to
 * whitespace-splitting if a line has no comma.
 * @returns {{rows: number[][], skippedHeader: boolean, errors: string[]}}
 */
export function parseCsvText(text) {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  const errors = [];
  const rows = [];
  let skippedHeader = false;

  lines.forEach((line, idx) => {
    const cells = (line.includes(",") ? line.split(",") : line.split(/\s+/)).map((c) => c.trim());
    if (cells.length < 3) {
      if (idx === 0) {
        skippedHeader = true; // tolerate a short/odd header line
        return;
      }
      errors.push(`Line ${idx + 1}: expected 3 columns (freq, col1, col2), got ${cells.length}. Skipped.`);
      return;
    }
    const nums = cells.slice(0, 3).map(Number);
    if (nums.some((n) => Number.isNaN(n))) {
      if (idx === 0) {
        skippedHeader = true;
        return;
      }
      errors.push(`Line ${idx + 1}: non-numeric value. Skipped.`);
      return;
    }
    rows.push(nums);
  });

  return { rows, skippedHeader, errors };
}

/**
 * Converts parsed rows to canonical form: frequency in Hz, impedance as
 * Re(Z)/Im(Z) in ohms, the same representation used throughout Phase
 * 1/2/3.
 * @param {number[][]} rows each row = [freq, col1, col2] in the user's units
 * @param {{freqUnit:'Hz'|'rad_s', representation:'reim'|'magphase', phaseUnit:'deg'|'rad'}} opts
 * @returns {{freqHz:number[], reOhm:number[], imOhm:number[]}}
 */
export function convertToCanonical(rows, opts) {
  const freqHz = [];
  const reOhm = [];
  const imOhm = [];

  for (const [freqRaw, col1, col2] of rows) {
    const f = opts.freqUnit === "rad_s" ? freqRaw / (2 * Math.PI) : freqRaw;
    freqHz.push(f);

    if (opts.representation === "reim") {
      reOhm.push(col1);
      imOhm.push(col2);
    } else {
      const phaseRad = opts.phaseUnit === "deg" ? (col2 * Math.PI) / 180 : col2;
      reOhm.push(col1 * Math.cos(phaseRad));
      imOhm.push(col1 * Math.sin(phaseRad));
    }
  }

  return { freqHz, reOhm, imOhm };
}
