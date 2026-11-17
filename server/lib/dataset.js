/**
 * Optional loader for the real UCI HAR test split (raw "Inertial Signals").
 * The 700 MB project folder is not needed: if the dataset is extracted anywhere below, the API
 * switches from simulated windows to genuine test windows and verifies the model's accuracy live.
 *
 * Looked up, in order: $HAR_DATA_DIR, ./data/UCI HAR Dataset, ./data, ../data/UCI HAR Dataset (project root).
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { T, C } from './engine.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.join(here, '..', '..');

const SIGNALS = [
  'body_acc_x', 'body_acc_y', 'body_acc_z',
  'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
  'total_acc_x', 'total_acc_y', 'total_acc_z',
];

export function findDatasetRoot() {
  const candidates = [
    process.env.HAR_DATA_DIR,
    path.join(projectRoot, 'data', 'UCI HAR Dataset'),
    path.join(projectRoot, 'data'),
    path.join(projectRoot, '..', 'data', 'UCI HAR Dataset'),
  ].filter(Boolean);
  for (const dir of candidates) {
    if (fs.existsSync(path.join(dir, 'test', 'y_test.txt'))) return dir;
  }
  return null;
}

function readNumberRows(file) {
  const text = fs.readFileSync(file, 'utf8');
  const lines = text.split('\n').filter((l) => l.trim().length);
  return lines.map((l) => l.trim().split(/\s+/).map(Number));
}

/** @returns {{n:number, X:Float32Array, y:Uint8Array, subjects:Uint8Array}} X is [n][128][9] flattened */
export function loadTestSet(root) {
  const base = path.join(root, 'test');
  const y = readNumberRows(path.join(base, 'y_test.txt')).map((r) => r[0] - 1);
  const subj = readNumberRows(path.join(base, 'subject_test.txt')).map((r) => r[0]);
  const n = y.length;
  const X = new Float32Array(n * T * C);
  SIGNALS.forEach((sig, c) => {
    const rows = readNumberRows(path.join(base, 'Inertial Signals', `${sig}_test.txt`));
    if (rows.length !== n) throw new Error(`${sig}: expected ${n} rows, got ${rows.length}`);
    for (let i = 0; i < n; i++) {
      const row = rows[i];
      if (row.length !== T) throw new Error(`${sig}: row ${i} has ${row.length} values, expected ${T}`);
      for (let t = 0; t < T; t++) X[(i * T + t) * C + c] = row[t];
    }
  });
  return { n, X, y: Uint8Array.from(y), subjects: Uint8Array.from(subj) };
}

export function windowAt(ds, i) {
  const out = [];
  for (let t = 0; t < T; t++) {
    const o = (i * T + t) * C;
    out.push(Array.from(ds.X.subarray(o, o + C), (v) => Math.round(v * 1e5) / 1e5));
  }
  return out;
}
