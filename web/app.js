import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";

import { outline, segment, segmentMC } from "./polyp.js";

// ?models=<url> loads the ONNX files from elsewhere (e.g. a local copy while developing).
const HUB = new URLSearchParams(location.search).get("models")
  || "https://huggingface.co/haroonx02/polyp-segmentation-unet-resnet34/resolve/main/onnx/";
const MODEL_URL = `${HUB}polyp_unet_int8.onnx`;
const MC_URL = `${HUB}polyp_unet_mc8_int8.onnx`;
const EXAMPLES = ["1_typical", "2_small", "3_large", "4_partial_miss", "5_confident_miss"];
const MAX_SIDE = 1280; // larger uploads are shrunk first; Kvasir frames are at most 1920 px
const STD_MAX = 0.25; // uncertainty colour scale (std of a probability is at most 0.5)
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];

const view = document.getElementById("view");
const ctx = view.getContext("2d", { willReadFrequently: true });
const status = document.getElementById("status");
const legend = document.getElementById("legend");
const truthBox = document.getElementById("truth");
let session = null;
let mcSession = null;
let mode = "mask";
let current = null; // {pixels, w, h, probs, mask, edge, truthEdge, mc, timing}

async function fetchWithProgress(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label} download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    status.textContent = total
      ? `Downloading the ${label}… ${Math.round((100 * got) / total)} %`
      : `Downloading the ${label}… ${(got / 1e6).toFixed(1)} MB`;
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

function viridis(t) {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  return VIRIDIS[i].map((c, k) => c + (VIRIDIS[i + 1][k] - c) * f);
}

function render() {
  if (!current) return;
  const { pixels, w, h } = current;
  const img = new ImageData(new Uint8ClampedArray(pixels), w, h);
  const d = img.data;
  const paint = (i, rgb, a) => {
    for (let k = 0; k < 3; k++) d[i * 4 + k] = d[i * 4 + k] * (1 - a) + rgb[k] * a;
  };
  if (mode === "mask" && current.mask) {
    for (let i = 0; i < w * h; i++) {
      if (current.edge[i]) paint(i, [0, 255, 0], 1);
      else if (current.mask[i]) paint(i, [0, 255, 0], 0.4);
    }
  } else if (mode === "probability" && current.probs) {
    for (let i = 0; i < w * h; i++) paint(i, viridis(current.probs[i]), 0.55);
  } else if (mode === "uncertainty" && current.mc) {
    for (let i = 0; i < w * h; i++) paint(i, viridis(current.mc.std[i] / STD_MAX), 0.55);
  }
  if (truthBox.checked && current.truthEdge) {
    for (let i = 0; i < w * h; i++) if (current.truthEdge[i]) paint(i, [255, 255, 255], 1);
  }
  ctx.putImageData(img, 0, 0);
  updateLegend();
  updateStatus();
}

function updateLegend() {
  const truth = current && current.truthEdge && truthBox.checked
    ? '<span style="background:#fff;border:1px solid #888"></span>expert mask' : "";
  if (mode === "mask") {
    legend.innerHTML = `<span style="background:#00ff00"></span>predicted polyp ${truth}`;
  } else if (mode === "probability") {
    legend.innerHTML = `probability 0<span class="ramp"></span>1 ${truth}`;
  } else {
    legend.innerHTML = `MC-dropout std 0<span class="ramp"></span>${STD_MAX}+ ${truth}`;
  }
}

function updateStatus() {
  const t = current.timing;
  const threads = `${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads === 1 ? "" : "s"}`;
  if (mode === "uncertainty") {
    status.textContent = current.mc
      ? `8 dropout samples · ${t.mc.toFixed(0)} ms (WASM, ${threads}) · mean std inside the `
        + `predicted polyp ${meanStd().toFixed(3)}`
      : "Loading the uncertainty model…";
    return;
  }
  let area = 0;
  for (let i = 0; i < current.mask.length; i++) area += current.mask[i];
  status.textContent = `polyp covers ${(100 * area / current.mask.length).toFixed(1)} % of the `
    + `frame · model + pre/post ${t.det.toFixed(0)} ms (WASM, ${threads})`;
}

function meanStd() {
  let s = 0, n = 0;
  for (let i = 0; i < current.mask.length; i++) {
    if (current.mask[i]) { s += current.mc.std[i]; n++; }
  }
  return n ? s / n : 0;
}

async function ensureMC() {
  if (!mcSession) {
    const model = await fetchWithProgress(MC_URL, "uncertainty model");
    status.textContent = "Starting the uncertainty model…";
    mcSession = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
  }
  if (current && !current.mc) {
    const t0 = performance.now();
    current.mc = await segmentMC(mcSession, ort, current.pixels, current.w, current.h, 4);
    current.timing.mc = performance.now() - t0;
    window.lastMC = current.mc;
  }
}

// Decode without colour management: OpenCV ignores embedded ICC profiles, and the model was
// trained on the raw pixel values.
async function decode(src) {
  const res = await fetch(src);
  if (!res.ok) throw new Error(`could not load ${src}`);
  return createImageBitmap(await res.blob(), { colorSpaceConversion: "none" });
}

function pixelsOf(bitmap) {
  const s = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  view.width = Math.round(bitmap.width * s);
  view.height = Math.round(bitmap.height * s);
  ctx.drawImage(bitmap, 0, 0, view.width, view.height);
  return ctx.getImageData(0, 0, view.width, view.height);
}

async function show(name, button) {
  document.querySelectorAll(".examples button").forEach((b) =>
    b.setAttribute("aria-pressed", String(b === button)));
  try {
    const bitmap = await decode(name.startsWith("blob:") ? name : `examples/${name}.jpg`);
    const { data, width, height } = pixelsOf(bitmap);
    current = { pixels: data, w: width, h: height, timing: {} };
    let truthEdge = null;
    if (!name.startsWith("blob:")) {
      const truth = pixelsOf(await decode(`examples/${name}_mask.png`)).data;
      const m = new Uint8Array(width * height);
      for (let i = 0; i < m.length; i++) m[i] = truth[i * 4] > 127 ? 1 : 0;
      truthEdge = outline(m, width, height);
    }
    current.truthEdge = truthEdge;
    if (!session) return;
    const t0 = performance.now();
    const { probs, mask } = await segment(session, ort, data, width, height, 4);
    current.timing.det = performance.now() - t0;
    Object.assign(current, { probs, mask, edge: outline(mask, width, height) });
    window.lastResult = { probs, mask }; // for automated checks against the Python pipeline
    if (mode === "uncertainty") { render(); await ensureMC(); }
    render();
  } catch (err) {
    status.textContent = String(err.message || err);
  }
}

const strip = document.getElementById("examples");
for (const name of EXAMPLES) {
  const b = document.createElement("button");
  b.type = "button";
  b.title = name.replace(/^\d_/, "").replaceAll("_", " ");
  b.innerHTML = `<img src="examples/${name}.jpg" alt="${b.title}">`;
  b.addEventListener("click", () => show(name, b));
  strip.appendChild(b);
}
document.getElementById("file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) show(URL.createObjectURL(file), null);
});
truthBox.addEventListener("change", render);
document.querySelectorAll(".views button").forEach((b) => b.addEventListener("click", async () => {
  mode = b.dataset.view;
  document.querySelectorAll(".views button").forEach((o) =>
    o.setAttribute("aria-pressed", String(o === b)));
  render();
  if (mode === "uncertainty") {
    try {
      await ensureMC();
      render();
    } catch (err) {
      status.textContent = `Could not run the uncertainty model: ${err.message || err}`;
    }
  }
}));

window.showExample = (name) => show(name, [...strip.children][EXAMPLES.indexOf(name)]);
window.setView = (v) => document.querySelector(`.views button[data-view="${v}"]`).click();

(async () => {
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
  try {
    const model = await fetchWithProgress(MODEL_URL, "model");
    status.textContent = "Starting the model…";
    session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
    await show(EXAMPLES[0], strip.children[0]);
  } catch (err) {
    status.textContent = `Could not start the model: ${err.message || err}`;
  }
})();
