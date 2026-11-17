/** "Typical values" per activity (port of compute_presets in the original app): gravity, movement strength, rotation strength, dominant frequency. */
import { ACTIVITIES, T, C } from './engine.js';
import { simulateWindow } from './simulator.js';

const BINS = 64;
const COS = [];
const SIN = [];
for (let b = 0; b <= BINS; b++) {
  COS.push(Float32Array.from({ length: T }, (_, t) => Math.cos((2 * Math.PI * b * t) / T)));
  SIN.push(Float32Array.from({ length: T }, (_, t) => Math.sin((2 * Math.PI * b * t) / T)));
}
const r = (v, d) => { const k = 10 ** d; return Math.round(v * k) / k; };
const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** @param {Float32Array[]} windows flat [128*9] windows of one activity */
function summarise(windows) {
  const g = [0, 0, 0]; const ba = [0, 0, 0]; const gy = [0, 0, 0];
  const spec = new Float64Array(BINS + 1);
  for (const w of windows) {
    for (let k = 0; k < 3; k++) {
      let mG = 0; let mB = 0; let mY = 0;
      for (let t = 0; t < T; t++) { mG += w[t * C + 6 + k]; mB += w[t * C + k]; mY += w[t * C + 3 + k]; }
      mG /= T; mB /= T; mY /= T;
      let vB = 0; let vY = 0;
      const x = new Float32Array(T);
      for (let t = 0; t < T; t++) { const d = w[t * C + k] - mB; x[t] = d; vB += d * d; const e = w[t * C + 3 + k] - mY; vY += e * e; }
      g[k] += mG; ba[k] += Math.sqrt(vB / T); gy[k] += Math.sqrt(vY / T);
      for (let b = 1; b <= BINS; b++) {
        let re = 0; let im = 0;
        for (let t = 0; t < T; t++) { re += x[t] * COS[b][t]; im += x[t] * SIN[b][t]; }
        spec[b] += re * re + im * im;
      }
    }
  }
  const n = windows.length;
  let best = 1;
  for (let b = 2; b <= BINS; b++) if (spec[b] > spec[best]) best = b;
  return {
    g: g.map((v) => r(clip(v / n, -1.5, 1.5), 2)),
    ba: ba.map((v) => r(clip(v / n, 0, 1), 3)),
    gy: gy.map((v) => r(clip(v / n, 0, 2), 3)),
    freq: r(clip((best * 50) / T, 0.5, 5), 1),
    noise: 0.005,
  };
}

/** @param {(k:number)=>Float32Array[]|null} realWindows returns windows of class k from the real dataset, if loaded */
export function computePresets(realWindows) {
  const out = {};
  let source = 'dataset';
  ACTIVITIES.forEach((id, k) => {
    let wins = realWindows ? realWindows(k) : null;
    if (!wins || !wins.length) {
      source = 'simulator';
      wins = Array.from({ length: 40 }, (_, s) => Float32Array.from(simulateWindow(id, { seed: s + 1 }).window.flat()));
    }
    out[id] = summarise(wins);
  });
  return { source, presets: out };
}
