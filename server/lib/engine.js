/**
 * Pure-JavaScript inference engine for the trained CNN-LSTM HAR model.
 *
 * The weights in ./model/weights.bin were exported from `outputs/best_model.keras`
 * (Keras 3, 230,342 parameters). This file re-implements the forward pass of every
 * layer so the backend needs no Python / TensorFlow at runtime:
 *
 *   Input 128x9 -> [Conv1D(64,k5,same) -> BN -> ReLU -> MaxPool2] -> [Conv1D(128,k5,same) -> BN -> ReLU -> MaxPool2]
 *               -> LSTM(128, seq) -> LSTM(64) -> Dense(64, ReLU) -> Dense(6, softmax)
 *
 * Dropout layers are identity at inference time.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const modelDir = path.join(here, '..', 'model');

export const ACTIVITIES = [
  'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING',
];
export const CHANNELS = [
  'body_acc_x', 'body_acc_y', 'body_acc_z',
  'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
  'total_acc_x', 'total_acc_y', 'total_acc_z',
];
export const T = 128;
export const C = 9;

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function loadWeights() {
  const manifest = JSON.parse(fs.readFileSync(path.join(modelDir, 'manifest.json'), 'utf8'));
  const buf = fs.readFileSync(path.join(modelDir, 'weights.bin'));
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const w = {};
  for (const [name, m] of Object.entries(manifest.tensors)) {
    w[name] = all.subarray(m.offset, m.offset + m.length);
  }
  const norm = JSON.parse(fs.readFileSync(path.join(modelDir, 'norm.json'), 'utf8'));
  return { w, manifest, norm };
}

/** Conv1D, 'same' padding (Keras: pad_left = (k-1)>>1 ... for k=5 -> 2 left, 2 right). */
function conv1d(x, len, cin, wt, bias, k, cout) {
  const out = new Float32Array(len * cout);
  const pad = (k - 1) >> 1;
  for (let t = 0; t < len; t++) {
    for (let o = 0; o < cout; o++) out[t * cout + o] = bias[o];
    for (let i = 0; i < k; i++) {
      const s = t + i - pad;
      if (s < 0 || s >= len) continue;
      const xo = s * cin;
      for (let c = 0; c < cin; c++) {
        const xv = x[xo + c];
        if (xv === 0) continue;
        const wo = (i * cin + c) * cout;
        const oo = t * cout;
        for (let o = 0; o < cout; o++) out[oo + o] += xv * wt[wo + o];
      }
    }
  }
  return out;
}

function batchNormRelu(x, len, ch, g, b, m, v, eps = 1e-3) {
  const scale = new Float32Array(ch);
  const shift = new Float32Array(ch);
  for (let c = 0; c < ch; c++) {
    scale[c] = g[c] / Math.sqrt(v[c] + eps);
    shift[c] = b[c] - m[c] * scale[c];
  }
  for (let t = 0; t < len; t++) {
    for (let c = 0; c < ch; c++) {
      const y = x[t * ch + c] * scale[c] + shift[c];
      x[t * ch + c] = y > 0 ? y : 0;
    }
  }
  return x;
}

function maxPool2(x, len, ch) {
  const n = len >> 1;
  const out = new Float32Array(n * ch);
  for (let t = 0; t < n; t++) {
    for (let c = 0; c < ch; c++) {
      const a = x[2 * t * ch + c];
      const b = x[(2 * t + 1) * ch + c];
      out[t * ch + c] = a > b ? a : b;
    }
  }
  return out;
}

/** Keras LSTM (gate order i, f, c, o). Returns full sequence of hidden states (len x units). */
function lstm(x, len, inDim, W, U, b, units) {
  const seq = new Float32Array(len * units);
  const h = new Float32Array(units);
  const c = new Float32Array(units);
  const z = new Float32Array(4 * units);
  const g4 = 4 * units;
  for (let t = 0; t < len; t++) {
    for (let j = 0; j < g4; j++) z[j] = b[j];
    for (let k = 0; k < inDim; k++) {
      const xv = x[t * inDim + k];
      if (xv === 0) continue;
      const wo = k * g4;
      for (let j = 0; j < g4; j++) z[j] += xv * W[wo + j];
    }
    for (let k = 0; k < units; k++) {
      const hv = h[k];
      if (hv === 0) continue;
      const uo = k * g4;
      for (let j = 0; j < g4; j++) z[j] += hv * U[uo + j];
    }
    for (let j = 0; j < units; j++) {
      const i = sigmoid(z[j]);
      const f = sigmoid(z[units + j]);
      const g = Math.tanh(z[2 * units + j]);
      const o = sigmoid(z[3 * units + j]);
      c[j] = f * c[j] + i * g;
      h[j] = o * Math.tanh(c[j]);
      seq[t * units + j] = h[j];
    }
  }
  return seq;
}

function dense(x, inDim, W, b, outDim, relu) {
  const out = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) out[o] = b[o];
  for (let i = 0; i < inDim; i++) {
    const xv = x[i];
    if (xv === 0) continue;
    const wo = i * outDim;
    for (let o = 0; o < outDim; o++) out[o] += xv * W[wo + o];
  }
  if (relu) for (let o = 0; o < outDim; o++) if (out[o] < 0) out[o] = 0;
  return out;
}

function softmax(z) {
  let m = -Infinity;
  for (const v of z) if (v > m) m = v;
  const e = Array.from(z, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export class HarEngine {
  constructor() {
    const { w, manifest, norm } = loadWeights();
    this.w = w;
    this.norm = norm;
    this.paramCount = manifest.total;
  }

  /** Standardise a raw (physical-units) window with the training-set statistics. */
  normalise(raw) {
    const out = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < C; c++) {
        out[t * C + c] = (raw[t * C + c] - this.norm.mean[c]) / this.norm.std[c];
      }
    }
    return out;
  }

  /**
   * @param {Float32Array|number[]} raw flat row-major [t][c], 128*9 values, raw units (g, rad/s, g)
   * @param {{explain?: boolean}} opts
   */
  predict(raw, { explain = false } = {}) {
    const w = this.w;
    const x0 = this.normalise(raw);

    let x = conv1d(x0, 128, 9, w['conv1.w'], w['conv1.b'], 5, 64);
    batchNormRelu(x, 128, 64, w['bn1.gamma'], w['bn1.beta'], w['bn1.mean'], w['bn1.var']);
    const p1 = maxPool2(x, 128, 64); // 64 x 64

    x = conv1d(p1, 64, 64, w['conv2.w'], w['conv2.b'], 5, 128);
    batchNormRelu(x, 64, 128, w['bn2.gamma'], w['bn2.beta'], w['bn2.mean'], w['bn2.var']);
    const p2 = maxPool2(x, 64, 128); // 32 x 128

    const l1 = lstm(p2, 32, 128, w['lstm1.W'], w['lstm1.U'], w['lstm1.b'], 128); // 32 x 128
    const l2seq = lstm(l1, 32, 128, w['lstm2.W'], w['lstm2.U'], w['lstm2.b'], 64);
    const l2 = l2seq.subarray(31 * 64, 32 * 64); // last hidden state (64)

    const d1 = dense(l2, 64, w['dense1.w'], w['dense1.b'], 64, true);
    const logits = dense(d1, 64, w['dense2.w'], w['dense2.b'], 6, false);
    const probs = softmax(logits);

    let best = 0;
    for (let i = 1; i < 6; i++) if (probs[i] > probs[best]) best = i;

    const result = { probs, index: best, label: ACTIVITIES[best], confidence: probs[best] };
    if (explain) {
      result.activations = {
        conv1: { shape: [64, 64], data: p1 },
        conv2: { shape: [32, 128], data: p2 },
        lstm1: { shape: [32, 128], data: l1 },
        lstm2: { shape: [64], data: Float32Array.from(l2) },
        dense: { shape: [64], data: d1 },
        logits: Array.from(logits),
      };
    }
    return result;
  }
}
