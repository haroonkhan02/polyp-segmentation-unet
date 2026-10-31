// Run the browser demo's pipeline (web/polyp.js + onnxruntime-web WASM) on frames dumped by
// scripts/web_parity.py and write the probability (or MC mean / std) maps it produces.
//
// Usage: node parity.mjs <dump dir> <model.onnx> [--mc]

import fs from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-web";

import { segment, segmentMC } from "../../web/polyp.js";

const [dir, modelPath, flag] = process.argv.slice(2);
const mc = flag === "--mc";
const meta = JSON.parse(fs.readFileSync(path.join(dir, "meta.json"), "utf8"));
const session = await ort.InferenceSession.create(fs.readFileSync(modelPath),
  { executionProviders: ["wasm"] });

let total = 0;
for (const item of meta.images) {
  const rgb = fs.readFileSync(path.join(dir, item.file));
  const start = performance.now();
  const out = mc
    ? await segmentMC(session, ort, rgb, item.width, item.height, 3)
    : await segment(session, ort, rgb, item.width, item.height, 3);
  total += performance.now() - start;
  const maps = mc ? [out.mean, out.std] : [out.probs];
  maps.forEach((m, k) => fs.writeFileSync(path.join(dir, `${item.file}.js${k}`), m));
}
const msPerFrame = total / meta.images.length;
fs.writeFileSync(path.join(dir, "js_meta.json"), JSON.stringify({
  ort_version: ort.env.versions.web, ms_per_frame: msPerFrame, mc,
}));
console.log(`${meta.images.length} frames, ${msPerFrame.toFixed(1)} ms/frame (WASM${mc ? ", MC" : ""})`);
