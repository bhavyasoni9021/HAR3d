/** Long recordings: validate rows and classify overlapping 128-sample windows (stride 64 = 50% overlap). */
import { T, C, ACTIVITIES } from './engine.js';

export const STRIDE = 64;
export const MAX_ROWS = 25700;

export function parseRows(data) {
  if (!Array.isArray(data) || !data.length) throw new Error('"data" must be an array of rows with 9 numbers each.');
  if (data.length < T) throw new Error(`Need at least ${T} rows (2.56 s at 50 Hz), got ${data.length}.`);
  if (data.length > MAX_ROWS) throw new Error(`Recording too long: ${data.length} rows (maximum ${MAX_ROWS}, about 8.5 minutes).`);
  const flat = new Float32Array(data.length * C);
  for (let i = 0; i < data.length; i++) {
    const row = data[i];
    if (!Array.isArray(row) || row.length !== C) throw new Error(`Row ${i + 1} must have ${C} values (got ${Array.isArray(row) ? row.length : 'none'}).`);
    for (let c = 0; c < C; c++) {
      const v = Number(row[c]);
      if (!Number.isFinite(v)) throw new Error(`Row ${i + 1}, column ${c + 1} is not a finite number.`);
      flat[i * C + c] = v;
    }
  }
  return { flat, rows: data.length };
}

export function predictSequence(engine, { flat, rows }) {
  const n = Math.floor((rows - T) / STRIDE) + 1;
  const windows = [];
  const counts = new Array(ACTIVITIES.length).fill(0);
  for (let i = 0; i < n; i++) {
    const raw = flat.subarray(i * STRIDE * C, (i * STRIDE + T) * C);
    const r = engine.predict(raw);
    counts[r.index]++;
    windows.push({ index: r.index, label: r.label, confidence: +r.confidence.toFixed(4), start: +((i * STRIDE) / 50).toFixed(2), probs: r.probs.map((p) => +p.toFixed(4)) });
  }
  return { windows, summary: { windows: n, counts, seconds: +(rows / 50).toFixed(2) } };
}
