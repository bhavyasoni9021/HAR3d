/** Runs the engine over the real test split and builds the same metrics the report presents. */
import { ACTIVITIES, T, C } from './engine.js';

const tick = () => new Promise((r) => setImmediate(r));

export async function evaluateDataset(engine, ds, onProgress = () => {}, sink = null) {
  if (sink) { sink.pred = new Uint8Array(ds.n); sink.conf = new Float32Array(ds.n); }
  const K = ACTIVITIES.length;
  const confusion = Array.from({ length: K }, () => new Array(K).fill(0));
  const subjStats = new Map();
  let confCorrect = 0;
  let confWrong = 0;
  let nCorrect = 0;
  let nWrong = 0;

  for (let i = 0; i < ds.n; i++) {
    const raw = ds.X.subarray(i * T * C, (i + 1) * T * C);
    const r = engine.predict(raw);
    const truth = ds.y[i];
    confusion[truth][r.index]++;
    if (sink) { sink.pred[i] = r.index; sink.conf[i] = r.confidence; }
    const s = subjStats.get(ds.subjects[i]) || { ok: 0, n: 0 };
    s.n++;
    if (r.index === truth) { s.ok++; confCorrect += r.confidence; nCorrect++; } else { confWrong += r.confidence; nWrong++; }
    subjStats.set(ds.subjects[i], s);
    if (i % 25 === 0) { onProgress(i / ds.n); await tick(); }
  }

  const perClass = ACTIVITIES.map((id, k) => {
    const tp = confusion[k][k];
    const support = confusion[k].reduce((a, b) => a + b, 0);
    const predicted = confusion.reduce((a, row) => a + row[k], 0);
    const precision = predicted ? tp / predicted : 0;
    const recall = support ? tp / support : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { id, precision: +precision.toFixed(4), recall: +recall.toFixed(4), f1: +f1.toFixed(4), support };
  });
  const correct = confusion.reduce((a, row, k) => a + row[k], 0);
  const macroF1 = perClass.reduce((a, c) => a + c.f1, 0) / K;

  return {
    accuracy: +((100 * correct) / ds.n).toFixed(2),
    macroF1: +macroF1.toFixed(3),
    testWindows: ds.n,
    correct,
    errors: ds.n - correct,
    confusion,
    perClass,
    subjects: [...subjStats.entries()].sort((a, b) => a[0] - b[0])
      .map(([id, s]) => ({ id, accuracy: +((100 * s.ok) / s.n).toFixed(1) })),
    confidence: {
      correct: nCorrect ? +(confCorrect / nCorrect).toFixed(3) : null,
      wrong: nWrong ? +(confWrong / nWrong).toFixed(3) : null,
    },
  };
}
