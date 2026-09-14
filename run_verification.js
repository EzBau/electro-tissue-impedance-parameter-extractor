// Phase 2 Task 5: load the exported ONNX model with onnxruntime-web (the
// browser/wasm runtime, NOT onnxruntime-node's native binding) and confirm
// it runs and matches the Python-side onnxruntime predictions on 5 fixed
// test-set spectra.
//
// Usage: node run_verification.js
// Output: prints comparison to stdout AND writes
//         ../eval/browser_runtime_verification.txt

const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-web");

const WEBTEST_DIR = __dirname;
const ROOT = path.dirname(WEBTEST_DIR);
const EVAL_DIR = path.join(ROOT, "eval");

async function main() {
  const samplesPath = path.join(WEBTEST_DIR, "test_samples.json");
  const payload = JSON.parse(fs.readFileSync(samplesPath, "utf-8"));
  const modelPath = path.join(ROOT, "models", "model.onnx");

  const lines = [];
  const log = (s) => {
    console.log(s);
    lines.push(s);
  };

  log("Phase 2 Task 5: onnxruntime-web (WASM backend) browser-runtime verification");
  const ortPkg = JSON.parse(fs.readFileSync(path.join(WEBTEST_DIR, "node_modules", "onnxruntime-web", "package.json"), "utf-8"));
  log(`Runtime: onnxruntime-web version ${ortPkg.version}, Node ${process.version}`);
  log(`Model: ${modelPath}`);
  log("");

  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["wasm"],
  });
  log(`ONNX Runtime Web session created OK. Input names: ${session.inputNames}, output names: ${session.outputNames}`);
  log("");

  const paramNames = payload.param_names;
  let maxAbsDiff = 0;
  let maxRelDiff = 0;

  for (const sample of payload.samples) {
    const inputArray = Float32Array.from(sample.input_re_im_40);
    const tensor = new ort.Tensor("float32", inputArray, [1, 40]);
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;

    const results = await session.run(feeds);
    const output = results[session.outputNames[0]].data; // Float32Array length 4

    log(`sample index=${sample.index} (slice_label=${sample.slice_label}):`);
    for (let j = 0; j < paramNames.length; j++) {
      const name = paramNames[j];
      const webVal = output[j];
      const pyVal = sample.python_onnxruntime_prediction[name];
      const gtVal = sample.ground_truth_raw[name];
      const absDiff = Math.abs(webVal - pyVal);
      const relDiff = absDiff / Math.max(Math.abs(pyVal), 1e-12);
      maxAbsDiff = Math.max(maxAbsDiff, absDiff);
      maxRelDiff = Math.max(maxRelDiff, relDiff);
      log(
        `  ${name.padEnd(5)} onnxruntime-web=${webVal.toPrecision(8)}  python_onnxruntime=${pyVal.toPrecision(8)}  ` +
          `ground_truth=${gtVal.toPrecision(8)}  abs_diff=${absDiff.toExponential(3)}  rel_diff=${relDiff.toExponential(3)}`
      );
    }
    log("");
  }

  log(`max abs diff (onnxruntime-web vs python onnxruntime) across all samples/params: ${maxAbsDiff.toExponential(6)}`);
  log(`max rel diff (onnxruntime-web vs python onnxruntime) across all samples/params: ${maxRelDiff.toExponential(6)}`);
  if (maxRelDiff > 1e-3) {
    log("*** WARNING: onnxruntime-web output diverges from Python onnxruntime by more than 0.1% relative. ***");
  } else {
    log("PASS: onnxruntime-web output matches Python onnxruntime closely (browser runtime confirmed working).");
  }

  fs.mkdirSync(EVAL_DIR, { recursive: true });
  fs.writeFileSync(path.join(EVAL_DIR, "browser_runtime_verification.txt"), lines.join("\n"));
  console.log(`\nWrote ${path.join(EVAL_DIR, "browser_runtime_verification.txt")}`);
}

main().catch((err) => {
  console.error("FAILED:", err);
  process.exit(1);
});
