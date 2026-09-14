// Thin wrapper around onnxruntime-web for the Phase 2 exported model.
// Assumes the global `ort` object is already available (loaded via a plain
// <script src="vendor/onnxruntime-web/ort.wasm.min.js"> tag in index.html,
// before this module is imported).
//
// Input/output contract (must match PHASE2_NOTES.md section 9 exactly):
//   input  "re_im_spectrum": float32 (1,40) = [Re(Z)@freq[0..19], Im(Z)@freq[0..19]]
//   output "params_raw":     float32 (1,4)  = [Rs, Rct, Q, alpha], raw scale

/* global ort */

export function configureOnnxRuntime(wasmDirAbsoluteUrl) {
  // onnxruntime-web's WASM backend loads a same-named .mjs wrapper via
  // dynamic import() (resolved against the *page* URL, not the wrapper's
  // own module URL), which then fetches its sibling .wasm binary (resolved
  // against ort.wasm.min.js's *own* script URL, not the page). Those two
  // different resolution bases mean a single base-directory string for
  // wasmPaths can't correctly locate both files at once. The explicit
  // per-filename mapping form sidesteps the ambiguity entirely.
  ort.env.wasm.wasmPaths = {
    "ort-wasm-simd-threaded.wasm": `${wasmDirAbsoluteUrl}ort-wasm-simd-threaded.wasm`,
    "ort-wasm-simd-threaded.mjs": `${wasmDirAbsoluteUrl}ort-wasm-simd-threaded.mjs`,
  };
  // Avoids requiring cross-origin-isolation (COOP/COEP) headers, which a
  // plain static host (like a static HF Space) does not set by default.
  // ONNX Runtime Web falls back to running the WASM backend single-threaded.
  ort.env.wasm.numThreads = 1;
}

export async function loadOnnxModel(modelUrl) {
  const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
  return session;
}

/**
 * @param {ort.InferenceSession} session
 * @param {number[]} re20 Re(Z) at the 20 canonical frequency points (ohm)
 * @param {number[]} im20 Im(Z) at the 20 canonical frequency points (ohm)
 * @returns {Promise<{Rs:number, Rct:number, Q:number, alpha:number}>}
 */
export async function runNnInference(session, re20, im20) {
  const inputArray = Float32Array.from([...re20, ...im20]);
  const tensor = new ort.Tensor("float32", inputArray, [1, 40]);
  const feeds = {};
  feeds[session.inputNames[0]] = tensor;
  const results = await session.run(feeds);
  const out = results[session.outputNames[0]].data;
  return { Rs: out[0], Rct: out[1], Q: out[2], alpha: out[3] };
}
