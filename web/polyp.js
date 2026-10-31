// Pre- and post-processing of the polyp U-Net, ported from src/polypseg (no DOM, so Node can
// run it too: scripts/web_parity checks it against the Python pipeline).

export const SIZE = 320;
export const THRESHOLD = 0.5;

// OpenCV INTER_AREA downscale weights along one axis: each output pixel averages the source
// span it covers, with fractional weights at the edges.
function areaTable(src, dst) {
  const scale = src / dst;
  const tab = [];
  for (let d = 0; d < dst; d++) {
    const f0 = d * scale;
    const f1 = f0 + scale;
    const s0 = Math.ceil(f0);
    const s1 = Math.floor(f1);
    const taps = [];
    if (s0 - f0 > 1e-3) taps.push([s0 - 1, (s0 - f0) / scale]);
    for (let s = s0; s < s1; s++) taps.push([s, 1 / scale]);
    if (f1 - s1 > 1e-3 && s1 < src) taps.push([s1, Math.min(1, f1 - s1) / scale]);
    tab.push(taps);
  }
  return tab;
}

// OpenCV INTER_LINEAR weights along one axis (half-pixel centres, edges clamped).
function linearTable(src, dst) {
  const scale = src / dst;
  const tab = [];
  for (let d = 0; d < dst; d++) {
    const f = (d + 0.5) * scale - 0.5;
    let s = Math.floor(f);
    let t = f - s;
    if (s < 0) { s = 0; t = 0; }
    if (s >= src - 1) { s = src - 1; t = 0; }
    tab.push(t === 0 ? [[s, 1]] : [[s, 1 - t], [s + 1, t]]);
  }
  return tab;
}

// Separable resize of interleaved 8-bit pixels (RGB or RGBA) to an RGB uint8 SIZE x SIZE
// image, rounded like cv2.resize: INTER_AREA when shrinking (every Kvasir frame), bilinear
// for small uploads.
export function resizeToInput(pixels, srcW, srcH, channels) {
  const shrink = srcW >= SIZE && srcH >= SIZE;
  const xt = shrink ? areaTable(srcW, SIZE) : linearTable(srcW, SIZE);
  const yt = shrink ? areaTable(srcH, SIZE) : linearTable(srcH, SIZE);
  const rows = new Float64Array(SIZE * 3);
  const out = new Uint8Array(SIZE * SIZE * 3);
  for (let dy = 0; dy < SIZE; dy++) {
    const acc = new Float64Array(SIZE * 3);
    for (const [sy, wy] of yt[dy]) {
      const base = sy * srcW * channels;
      for (let dx = 0; dx < SIZE; dx++) {
        let r = 0, g = 0, b = 0;
        for (const [sx, wx] of xt[dx]) {
          const i = base + sx * channels;
          r += pixels[i] * wx;
          g += pixels[i + 1] * wx;
          b += pixels[i + 2] * wx;
        }
        rows[dx * 3] = r;
        rows[dx * 3 + 1] = g;
        rows[dx * 3 + 2] = b;
      }
      for (let k = 0; k < acc.length; k++) acc[k] += rows[k] * wy;
    }
    for (let k = 0; k < acc.length; k++) {
      out[dy * SIZE * 3 + k] = Math.min(255, Math.max(0, Math.round(acc[k])));
    }
  }
  return out;
}

// Resized RGB uint8 (SIZE, SIZE, 3) -> CHW float32 in [0, 1]: the exported graph's input.
export function toInput(rgb) {
  const plane = SIZE * SIZE;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = rgb[i * 3] / 255;
    out[plane + i] = rgb[i * 3 + 1] / 255;
    out[2 * plane + i] = rgb[i * 3 + 2] / 255;
  }
  return out;
}

// SIZE x SIZE float map -> (w, h) with cv2.INTER_LINEAR, as PolypSegmenter does.
export function upsample(map, w, h) {
  const xt = linearTable(SIZE, w);
  const yt = linearTable(SIZE, h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (const [sy, wy] of yt[y]) {
        for (const [sx, wx] of xt[x]) v += map[sy * SIZE + sx] * wx * wy;
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

export function binarize(probs, threshold = THRESHOLD) {
  const mask = new Uint8Array(probs.length);
  for (let i = 0; i < probs.length; i++) mask[i] = probs[i] >= threshold ? 1 : 0;
  return mask;
}

// Mask pixels with a 4-neighbour outside the mask: the outline drawn on the image.
export function outline(mask, w, h) {
  const edge = new Uint8Array(mask.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || !mask[i - 1] || !mask[i + 1]
        || !mask[i - w] || !mask[i + w]) edge[i] = 1;
    }
  }
  return edge;
}

// Full deterministic pipeline: pixels -> {probs, mask} at the input resolution.
export async function segment(session, ort, pixels, w, h, channels) {
  const input = toInput(resizeToInput(pixels, w, h, channels));
  const out = await session.run({ image: new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]) });
  const probs = upsample(out.probs.data, w, h);
  return { probs, mask: binarize(probs) };
}

// MC-dropout graph: pixels -> {mean, std} at the input resolution.
export async function segmentMC(session, ort, pixels, w, h, channels) {
  const input = toInput(resizeToInput(pixels, w, h, channels));
  const out = await session.run({ image: new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]) });
  return { mean: upsample(out.mean.data, w, h), std: upsample(out.std.data, w, h) };
}
