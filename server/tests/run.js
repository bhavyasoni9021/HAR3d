/* Zero-dependency test runner: `npm test` */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { HarEngine, ACTIVITIES, T, C } from '../lib/engine.js';
import { simulateWindow, synthWindow } from '../lib/simulator.js';
import { computePresets } from '../lib/presets.js';
import { parseRows, predictSequence } from '../lib/sequence.js';
import { loadTestSet, windowAt } from '../lib/dataset.js';
import { evaluateDataset } from '../lib/evaluation.js';

let passed = 0;
let failed = 0;
async function test(name, fn) {
  try { await fn(); passed++; console.log(`  ok   ${name}`); }
  catch (err) { failed++; console.log(`  FAIL ${name}\n       ${err.message}`); }
}

const flat = (w) => Float32Array.from(w.flat());
const engine = new HarEngine();

console.log('engine');
await test('loads all 230,342 parameters', () => assert.equal(engine.paramCount, 230342));
await test('softmax output sums to 1', () => {
  const r = engine.predict(flat(simulateWindow('WALKING', { seed: 3 }).window));
  assert.ok(Math.abs(r.probs.reduce((a, b) => a + b, 0) - 1) < 1e-5);
});
await test('explain=true returns activations with the documented shapes', () => {
  const a = engine.predict(flat(simulateWindow('SITTING').window), { explain: true }).activations;
  assert.deepEqual(a.conv1.shape, [64, 64]);
  assert.deepEqual(a.conv2.shape, [32, 128]);
  assert.deepEqual(a.lstm1.shape, [32, 128]);
  assert.equal(a.lstm2.data.length, 64);
  assert.equal(a.logits.length, 6);
});
await test('recognises simulated windows well above chance (>= 85%)', () => {
  let ok = 0; let n = 0;
  for (let k = 0; k < 6; k++) for (let s = 1; s <= 40; s++) {
    n++; if (engine.predict(flat(simulateWindow(ACTIVITIES[k], { seed: s }).window)).index === k) ok++;
  }
  assert.ok(ok / n >= 0.85, `accuracy was ${(100 * ok / n).toFixed(1)}%`);
});
await test('laying is separated from standing by gravity orientation', () => {
  assert.equal(engine.predict(flat(simulateWindow('LAYING', { seed: 2 }).window)).label, 'LAYING');
  assert.equal(engine.predict(flat(simulateWindow('STANDING', { seed: 2 }).window)).label, 'STANDING');
});
await test('inference takes under 100 ms', () => {
  const w = flat(simulateWindow('WALKING').window);
  engine.predict(w);
  const t0 = performance.now(); engine.predict(w);
  assert.ok(performance.now() - t0 < 100);
});

console.log('simulator');
await test('is deterministic for a given seed', () => {
  assert.deepEqual(simulateWindow('WALKING', { seed: 5 }).window, simulateWindow('WALKING', { seed: 5 }).window);
});
await test('returns 128 x 9', () => {
  const { window } = simulateWindow('LAYING');
  assert.equal(window.length, T); assert.equal(window[0].length, C);
});
await test('rejects unknown activities', () => assert.throws(() => simulateWindow('DANCING')));

console.log('set-values mode, presets, long recordings');
await test('synthWindow builds 128 x 9 with total acc = body acc + gravity', () => {
  const { window } = synthWindow({ g: [1, 0, 0], ba: [0.3, 0, 0], freq: 2, noise: 0 });
  assert.equal(window.length, T);
  assert.ok(Math.abs(window[5][6] - (1 + window[5][0])) < 1e-4);
});
await test('presets yield sensible predictions for static postures', () => {
  const { presets } = computePresets(null);
  for (const id of ['STANDING', 'LAYING']) {
    const p = presets[id];
    const { window } = synthWindow({ g: p.g, ba: p.ba, gy: p.gy, freq: p.freq, noise: p.noise });
    assert.equal(engine.predict(flat(window)).label, id);
  }
});
await test('long recordings are split into 50%-overlap windows', () => {
  const rows = [];
  for (let i = 0; i < 4; i++) rows.push(...simulateWindow('SITTING', { seed: i + 1 }).window.slice(0, 64));
  rows.push(...simulateWindow('SITTING', { seed: 9 }).window.slice(0, 64));
  const out = predictSequence(engine, parseRows(rows));
  assert.equal(out.summary.windows, Math.floor((rows.length - 128) / 64) + 1);
  assert.ok(out.windows.every((w) => w.label === 'SITTING'));
});
await test('recordings shorter than 128 rows are rejected', () => assert.throws(() => parseRows([[0, 0, 0, 0, 0, 0, 0, 0, 0]])));

console.log('dataset loader + live evaluation (fake dataset in a temp folder)');
await test('parses UCI layout and evaluates', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'har-'));
  const n = 12;
  fs.mkdirSync(path.join(root, 'test', 'Inertial Signals'), { recursive: true });
  const labels = []; const wins = [];
  for (let i = 0; i < n; i++) { const k = i % 6; labels.push(k + 1); wins.push(simulateWindow(ACTIVITIES[k], { seed: i + 1 }).window); }
  fs.writeFileSync(path.join(root, 'test', 'y_test.txt'), labels.join('\n') + '\n');
  fs.writeFileSync(path.join(root, 'test', 'subject_test.txt'), labels.map((_, i) => (i % 2 ? 2 : 4)).join('\n') + '\n');
  const sigs = ['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z', 'total_acc_x', 'total_acc_y', 'total_acc_z'];
  sigs.forEach((s, c) => fs.writeFileSync(
    path.join(root, 'test', 'Inertial Signals', `${s}_test.txt`),
    wins.map((w) => ' ' + w.map((row) => row[c].toExponential(8)).join('  ')).join('\n') + '\n'));
  const ds = loadTestSet(root);
  assert.equal(ds.n, n);
  assert.equal(windowAt(ds, 3).length, T);
  const res = await evaluateDataset(engine, ds);
  assert.equal(res.testWindows, n);
  assert.equal(res.confusion.flat().reduce((a, b) => a + b, 0), n);
  assert.deepEqual(res.subjects.map((s) => s.id), [2, 4]);
  assert.ok(res.accuracy >= 75);
  fs.rmSync(root, { recursive: true, force: true });
});

console.log('http api');
let app = null;
try { ({ createApp: app } = await import('../app.js')); } catch (err) {
  if (err.code === 'ERR_MODULE_NOT_FOUND') console.log('  skip express is not installed yet (run npm install)');
  else throw err;
}
if (app) {
  const server = app().listen(0);
  const base = `http://127.0.0.1:${server.address().port}`;
  const get = async (p) => { const r = await fetch(base + p); return { status: r.status, body: await r.json() }; };
  await test('GET /api/health', async () => { const r = await get('/api/health'); assert.equal(r.body.status, 'ok'); });
  await test('GET /api/metrics reproduces the report confusion matrix', async () => {
    const r = await get('/api/metrics'); assert.equal(r.body.confusion[4][3], 107);
  });
  await test('GET /api/simulate validates activity', async () => {
    assert.equal((await get('/api/simulate?activity=nope')).status, 400);
    assert.equal((await get('/api/simulate?activity=sitting&seed=4')).body.window.length, T);
  });
  await test('POST /api/predict classifies and explains', async () => {
    const { window } = (await get('/api/simulate?activity=LAYING&seed=1')).body;
    const r = await fetch(base + '/api/predict', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ window, explain: true }) });
    const j = await r.json();
    assert.equal(j.prediction.label, 'LAYING'); assert.ok(j.activations.conv1.data.length === 4096);
  });
  await test('POST /api/predict rejects bad windows', async () => {
    const r = await fetch(base + '/api/predict', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ window: [[1, 2]] }) });
    assert.equal(r.status, 400);
  });
  server.close();
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
