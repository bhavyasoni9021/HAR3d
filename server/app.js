import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { HarEngine, ACTIVITIES, T, C } from './lib/engine.js';
import { simulateWindow, synthWindow } from './lib/simulator.js';
import { computePresets } from './lib/presets.js';
import { parseRows, predictSequence } from './lib/sequence.js';
import { findDatasetRoot, loadTestSet, windowAt } from './lib/dataset.js';
import { evaluateDataset } from './lib/evaluation.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const report = JSON.parse(fs.readFileSync(path.join(here, 'data', 'report.json'), 'utf8'));

const round = (arr, d = 3) => {
  const k = 10 ** d;
  return Array.from(arr, (v) => Math.round(v * k) / k);
};

/** Accepts [128][9] or a flat 1152 array; returns Float32Array or throws a validation error. */
export function parseWindow(input) {
  let flat;
  if (Array.isArray(input) && input.length === T && Array.isArray(input[0])) {
    if (!input.every((r) => Array.isArray(r) && r.length === C)) throw new Error(`Every row must have ${C} values (one per channel).`);
    flat = input.flat();
  } else if (Array.isArray(input) && input.length === T * C) {
    flat = input;
  } else {
    throw new Error(`"window" must be ${T} rows × ${C} columns (or a flat array of ${T * C} numbers).`);
  }
  const out = new Float32Array(T * C);
  for (let i = 0; i < flat.length; i++) {
    const v = Number(flat[i]);
    if (!Number.isFinite(v)) throw new Error('The window contains a value that is not a finite number.');
    out[i] = v;
  }
  return out;
}

export function createApp() {
  const app = express();
  const engine = new HarEngine();
  const started = Date.now();

  const state = {
    dataset: null,
    datasetRoot: findDatasetRoot(),
    evaluation: { state: 'idle', progress: 0, result: null, error: null },
    preds: {},
    presets: null,
  };

  // Load the real test split (if present) and verify the model on it in the background.
  if (state.datasetRoot) {
    setTimeout(async () => {
      try {
        state.evaluation.state = 'running';
        state.dataset = loadTestSet(state.datasetRoot);
        console.log(`[har] real test set loaded: ${state.dataset.n} windows from ${state.datasetRoot}`);
        state.evaluation.result = await evaluateDataset(engine, state.dataset, (p) => { state.evaluation.progress = p; }, state.preds);
        state.evaluation.state = 'done';
        state.evaluation.progress = 1;
        console.log(`[har] live evaluation: accuracy ${state.evaluation.result.accuracy}% on ${state.dataset.n} windows`);
      } catch (err) {
        state.evaluation.state = 'error';
        state.evaluation.error = String(err.message || err);
        console.warn('[har] could not evaluate the real dataset:', state.evaluation.error);
      }
    }, 50);
  }

  app.disable('x-powered-by');
  app.use(express.json({ limit: '16mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'ok',
      uptimeSeconds: Math.round((Date.now() - started) / 1000),
      model: { loaded: true, parameters: engine.paramCount, runtime: 'node-js (pure JavaScript forward pass)' },
      dataset: {
        available: !!state.dataset,
        root: state.datasetRoot ? path.basename(state.datasetRoot) : null,
        windows: state.dataset ? state.dataset.n : 0,
      },
      evaluation: { state: state.evaluation.state, progress: +state.evaluation.progress.toFixed(2), error: state.evaluation.error },
    });
  });

  app.get('/api/report', (_req, res) => res.json(report));

  app.get('/api/model', (_req, res) => {
    res.json({
      name: 'CNN_LSTM_HAR',
      parameters: engine.paramCount,
      trainable: report.headline.trainable,
      inputShape: [T, C],
      classes: ACTIVITIES,
      channels: report.dataset.channels,
      normalisation: engine.norm,
      layers: report.layers,
      training: report.training,
    });
  });

  app.get('/api/metrics', (_req, res) => {
    if (state.evaluation.state === 'done') {
      return res.json({ source: 'computed', note: 'Computed live by this server on the real UCI HAR test split.', ...state.evaluation.result });
    }
    res.json({
      source: 'report',
      note: 'Values from the case study report. Add the UCI HAR dataset to ./data to verify them live.',
      accuracy: report.headline.accuracy,
      macroF1: report.headline.macroF1,
      testWindows: report.headline.testWindows,
      correct: report.headline.correct,
      errors: report.headline.errors,
      confusion: report.confusion,
      perClass: report.perClass,
      subjects: report.subjects,
      confidence: report.confidence,
    });
  });

  app.get('/api/simulate', (req, res) => {
    const activity = String(req.query.activity || 'WALKING').toUpperCase();
    if (!ACTIVITIES.includes(activity)) return res.status(400).json({ error: `activity must be one of ${ACTIVITIES.join(', ')}` });
    const num = (v, d, lo, hi) => {
      const x = v === undefined ? d : Number(v);
      return Number.isFinite(x) ? Math.min(hi, Math.max(lo, x)) : d;
    };
    const seed = Math.floor(num(req.query.seed, 1, 0, 1e9));
    const noise = num(req.query.noise, 1, 0, 6);
    const tilt = num(req.query.tilt, 0, -45, 60);
    res.json(simulateWindow(activity, { seed, noise, tilt }));
  });

  app.get('/api/presets', (_req, res) => {
    if (!state.presets || (state.dataset && state.presets.source !== 'dataset')) {
      const real = state.dataset
        ? (k) => {
          const out = [];
          for (let i = 0; i < state.dataset.n; i++) if (state.dataset.y[i] === k) out.push(state.dataset.X.subarray(i * T * C, (i + 1) * T * C));
          return out;
        }
        : null;
      state.presets = computePresets(real);
    }
    res.json(state.presets);
  });

  app.post('/api/synthesize', (req, res) => {
    try { res.json(synthWindow(req.body || {})); } catch (err) { res.status(400).json({ error: err.message }); }
  });

  app.post('/api/predict-sequence', (req, res) => {
    let parsed;
    try { parsed = parseRows(req.body && req.body.data); } catch (err) { return res.status(400).json({ error: err.message }); }
    const t0 = process.hrtime.bigint();
    const out = predictSequence(engine, parsed);
    out.latencyMs = +(Number(process.hrtime.bigint() - t0) / 1e6).toFixed(1);
    res.json(out);
  });

  const noDataset = { error: 'The real dataset is not available on this server. Extract the UCI HAR dataset into ./data (see README).' };

  app.get('/api/samples/index', (req, res) => {
    if (!state.dataset) return res.status(404).json(noDataset);
    const wanted = req.query.activity ? ACTIVITIES.indexOf(String(req.query.activity).toUpperCase()) : -1;
    const wrongOnly = String(req.query.wrong) === 'true';
    if (wrongOnly && state.evaluation.state !== 'done') {
      return res.status(409).json({ error: `Still verifying the model on the test set (${Math.round(state.evaluation.progress * 100)}%). Try again in a moment.` });
    }
    const indices = [];
    for (let i = 0; i < state.dataset.n; i++) {
      if (wanted >= 0 && state.dataset.y[i] !== wanted) continue;
      if (wrongOnly && state.preds.pred[i] === state.dataset.y[i]) continue;
      indices.push(i);
    }
    res.json({ count: indices.length, indices });
  });

  app.get('/api/samples/random', (req, res) => {
    if (!state.dataset) return res.status(404).json({ error: 'The real dataset is not available on this server. Use /api/simulate or add the UCI HAR dataset to ./data.' });
    const wanted = req.query.activity ? ACTIVITIES.indexOf(String(req.query.activity).toUpperCase()) : -1;
    const pool = [];
    for (let i = 0; i < state.dataset.n; i++) if (wanted < 0 || state.dataset.y[i] === wanted) pool.push(i);
    if (!pool.length) return res.status(400).json({ error: 'No windows for that activity.' });
    const i = pool[Math.floor(Math.random() * pool.length)];
    res.json({
      window: windowAt(state.dataset, i),
      meta: { index: i, activity: ACTIVITIES[state.dataset.y[i]], subject: state.dataset.subjects[i], simulated: false },
    });
  });

  app.get('/api/samples/:index', (req, res) => {
    if (!state.dataset) return res.status(404).json(noDataset);
    const i = Number(req.params.index);
    if (!Number.isInteger(i) || i < 0 || i >= state.dataset.n) return res.status(400).json({ error: `index must be an integer between 0 and ${state.dataset.n - 1}.` });
    res.json({ window: windowAt(state.dataset, i), meta: { index: i, activity: ACTIVITIES[state.dataset.y[i]], subject: state.dataset.subjects[i], simulated: false } });
  });

  app.post('/api/predict', (req, res) => {
    let raw;
    try {
      raw = parseWindow(req.body && req.body.window);
    } catch (err) {
      return res.status(400).json({ error: err.message });
    }
    const explain = !!(req.body && req.body.explain);
    const t0 = process.hrtime.bigint();
    const r = engine.predict(raw, { explain });
    const latencyMs = Number(process.hrtime.bigint() - t0) / 1e6;

    const body = {
      prediction: {
        index: r.index,
        label: r.label,
        confidence: +r.confidence.toFixed(4),
        probabilities: ACTIVITIES.map((label, i) => ({ label, probability: +r.probs[i].toFixed(4) })),
      },
      latencyMs: +latencyMs.toFixed(2),
    };
    if (explain) {
      const a = r.activations;
      const pack = (layer) => ({ shape: layer.shape, data: round(layer.data) });
      body.activations = {
        conv1: pack(a.conv1), conv2: pack(a.conv2), lstm1: pack(a.lstm1),
        lstm2: pack(a.lstm2), dense: pack(a.dense), logits: round(a.logits, 3),
      };
    }
    res.json(body);
  });

  app.use('/api', (_req, res) => res.status(404).json({ error: 'Unknown API route.' }));

  // Production: serve the built front end.
  const dist = path.join(here, '..', 'dist');
  if (fs.existsSync(path.join(dist, 'index.html'))) {
    app.use(express.static(dist));
    app.get('*', (_req, res) => res.sendFile(path.join(dist, 'index.html')));
  }

  // eslint-disable-next-line no-unused-vars
  app.use((err, _req, res, _next) => {
    const status = err.status || 500;
    res.status(status).json({ error: status === 400 ? 'Request body is not valid JSON.' : 'Internal server error.' });
  });

  return app;
}
