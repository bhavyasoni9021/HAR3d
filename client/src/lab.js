import { THREE, makeStage, ACT, actIndex, api } from './common.js';
import { Mannequin, Floor } from './mannequin.js';

const $ = (id) => document.getElementById(id);
const CHANNELS = ['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z', 'total_acc_x', 'total_acc_y', 'total_acc_z'];
const post = (body) => ({ method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });

/* ---------- file parsing ---------- */
function parseNpy(buf) {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.slice(1, 6)) !== 'NUMPY') throw new Error('That is not a valid .npy file.');
  const dv = new DataView(buf);
  const major = u8[6];
  const hl = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const off = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(u8.slice(off, off + hl));
  const descr = (/'descr':\s*'([^']+)'/.exec(header) || [])[1];
  if (/'fortran_order':\s*True/.test(header)) throw new Error('Fortran-ordered .npy files are not supported.');
  const body = buf.slice(off + hl);
  if (descr === '<f4') return Array.from(new Float32Array(body));
  if (descr === '<f8') return Array.from(new Float64Array(body));
  throw new Error(`Unsupported dtype ${descr}. Save the array as float32 or float64.`);
}

function parseText(text, name) {
  if (name.endsWith('.json')) {
    const j = JSON.parse(text);
    return j.window || j.data || j;
  }
  return text.split(/\r?\n/).map((l) => l.trim().split(/[,;\s]+/).map(Number)).filter((r) => r.length > 1 && r.every(Number.isFinite));
}

function toRows(win) {
  let rows = win;
  if (!Array.isArray(win[0])) {
    if (win.length % 9) throw new Error(`The file holds ${win.length} numbers, which is not a multiple of 9 columns.`);
    rows = [];
    for (let i = 0; i < win.length; i += 9) rows.push(win.slice(i, i + 9));
  }
  if (!rows.every((r) => r.length === 9)) throw new Error('Expected 9 columns: body acc xyz, body gyro xyz, total acc xyz.');
  if (rows.length < 128) throw new Error(`Need at least 128 rows (2.56 s at 50 Hz), found ${rows.length}.`);
  return rows;
}

/* ---------- drawing ---------- */
function setup(canvas, h) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = canvas.clientWidth;
  canvas.width = W * dpr; canvas.height = h * dpr;
  canvas.style.height = `${h}px`;
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, h);
  return { g, W };
}

function drawPlot(canvas, rows, color) {
  const H = 300;
  const { g, W } = setup(canvas, H);
  const groups = [['Body acceleration (g)', 0], ['Gyroscope (rad/s)', 3], ['Total acceleration (g)', 6]];
  const ph = H / 3;
  const axes = ['x', 'y', 'z'];
  const shades = [1, 0.7, 0.45];
  groups.forEach(([title, c0], gi) => {
    const top = gi * ph + 22;
    const bot = (gi + 1) * ph - 8;
    let lo = Infinity;
    let hi = -Infinity;
    for (const r of rows) for (let k = 0; k < 3; k++) { lo = Math.min(lo, r[c0 + k]); hi = Math.max(hi, r[c0 + k]); }
    if (hi - lo < 0.05) { const m = (hi + lo) / 2; lo = m - 0.05; hi = m + 0.05; }
    g.fillStyle = '#a9acd6'; g.font = '12px "Instrument Sans", sans-serif'; g.textAlign = 'left';
    g.fillText(`${title}   ${lo.toFixed(2)} to ${hi.toFixed(2)}`, 10, gi * ph + 14);
    g.strokeStyle = 'rgba(255,255,255,.08)'; g.beginPath(); g.moveTo(0, bot + 4); g.lineTo(W, bot + 4); g.stroke();
    for (let k = 0; k < 3; k++) {
      g.beginPath();
      g.strokeStyle = color; g.globalAlpha = shades[k]; g.lineWidth = 1.6;
      rows.forEach((r, t) => {
        const x = 8 + (t / (rows.length - 1)) * (W - 16);
        const y = bot - ((r[c0 + k] - lo) / (hi - lo)) * (bot - top);
        if (t) g.lineTo(x, y); else g.moveTo(x, y);
      });
      g.stroke();
      g.globalAlpha = 1;
      g.fillStyle = color; g.fillText(axes[k], W - 22, top + 12 + k * 13);
    }
  });
}

function drawSequence(seq, selected) {
  const { windows, summary } = seq.result;
  const N = seq.rows.length;
  // timeline strip
  let { g, W } = setup($('timelineCanvas'), 64);
  windows.forEach((w, i) => {
    const x0 = (i * 64 / N) * W;
    const x1 = i === windows.length - 1 ? W : ((i * 64 + 64) / N) * W;
    g.globalAlpha = 0.45 + 0.55 * w.confidence;
    g.fillStyle = ACT[w.index].color;
    g.fillRect(x0, 6, Math.max(1, x1 - x0 - 1), 40);
    g.globalAlpha = 1;
    if (i === selected) { g.strokeStyle = '#fff'; g.lineWidth = 2; g.strokeRect(x0, 4, Math.max(2, x1 - x0 - 1), 44); }
  });
  g.fillStyle = '#a9acd6'; g.font = '11px "Instrument Sans", sans-serif';
  g.textAlign = 'left'; g.fillText('0 s', 2, 60);
  g.textAlign = 'right'; g.fillText(`${summary.seconds} s`, W - 2, 60);
  // probabilities over time
  ({ g, W } = setup($('probCanvas'), 170));
  const H = 170;
  g.strokeStyle = 'rgba(255,255,255,.08)';
  for (const f of [0, 0.5, 1]) { const y = 10 + (1 - f) * (H - 30); g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
  ACT.forEach((a, k) => {
    g.beginPath(); g.strokeStyle = a.color; g.lineWidth = 1.8;
    windows.forEach((w, i) => {
      const x = windows.length > 1 ? 8 + (i / (windows.length - 1)) * (W - 16) : W / 2;
      const y = 10 + (1 - w.probs[k]) * (H - 30);
      if (i) g.lineTo(x, y); else g.moveTo(x, y);
    });
    g.stroke();
  });
  g.fillStyle = '#a9acd6'; g.font = '11px "Instrument Sans", sans-serif'; g.textAlign = 'left';
  g.fillText('1.0', 2, 8); g.fillText('0', 2, H - 22);
  g.fillText(`window start (s): 0 … ${windows[windows.length - 1].start}`, 8, H - 6);
  // summary
  const mx = Math.max(...summary.counts, 1);
  $('seqSummary').innerHTML = ACT.map((a, k) => `<div class="row"><span><i class="dot" style="background:${a.color}"></i>${a.name}</span><div><div class="tr" style="width:${100 * summary.counts[k] / mx}%;background:${a.color}"></div><div class="cap">${summary.counts[k]} of ${summary.windows} windows (${(100 * summary.counts[k] / summary.windows).toFixed(0)}%)</div></div></div>`).join('');
}

/* ---------- the lab ---------- */
export function initLab() {
  const stage = makeStage($('labCanvas'), { fov: 40, pos: [2.6, 1.5, 4.2], target: [0, 0.8, 0], orbit: true });
  stage.controls.maxPolarAngle = Math.PI * 0.5;
  stage.controls.minDistance = 2.5;
  stage.controls.maxDistance = 9;
  stage.scene.add(new THREE.AmbientLight(0xffffff, 1.0));
  const sun = new THREE.DirectionalLight(0xffffff, 2);
  sun.position.set(3, 6, 4);
  stage.scene.add(sun);
  const man = new Mannequin('#ffb84d');
  stage.scene.add(man.group);
  const floor = new Floor();
  stage.scene.add(floor.group);
  let shown = 'STANDING';
  let lastIdx = -1;
  stage.onFrame((dt, now) => { man.update(shown, stage.reduce ? 0 : now, dt); floor.update(shown, dt * (stage.reduce ? 0 : 1)); });

  const state = {
    mode: 'sim', act: 'WALKING', seed: 3, noise: 1, tilt: 0, window: null, meta: null, ticket: 0,
    vals: { g_x: 0.97, g_y: -0.2, g_z: 0.06, ba_x: 0, ba_y: 0, ba_z: 0, gy_x: 0, gy_y: 0, gy_z: 0, freq: 2, noise: 0.005 },
    presets: null, real: { pool: [], pos: 0, ready: false }, seq: null, selected: -1,
  };
  const msg = (t) => { $('fileMsg').textContent = t || ''; };

  /* ----- shared pipeline ----- */
  async function setWindow(rows, meta) {
    state.window = rows; state.meta = meta;
    await classify();
  }
  async function classify() {
    const ticket = ++state.ticket;
    const r = await api('/predict', post({ window: state.window, explain: true }));
    if (ticket !== state.ticket) return;
    render(r);
    window.dispatchEvent(new CustomEvent('har:predict', { detail: { result: r, window: state.window } }));
  }
  const guard = (fn) => async (...a) => { try { msg(''); await fn(...a); } catch (err) { msg(err.message); } };

  function sourceText(m) {
    if (m.manual) return 'Window built from your slider values.';
    if (m.seqIndex !== undefined) return `Window ${m.seqIndex + 1} of ${m.seqCount} from “${m.upload}”, starting at ${(m.seqIndex * 64 / 50).toFixed(2)} s.`;
    if (m.upload) return `Your window “${m.upload}”.`;
    if (m.index !== undefined) return `Real test window #${m.index} from subject ${m.subject}, labelled ${ACT[actIndex(m.activity)].name.toLowerCase()}.`;
    return `Synthetic window generated for ${ACT[actIndex(m.activity)].name.toLowerCase()}, subject ${m.seed}.`;
  }

  function render(r) {
    const p = r.prediction;
    const a = ACT[p.index];
    shown = a.id;
    man.setColor(a.color);
    if (lastIdx !== p.index) { man.burst(); lastIdx = p.index; }
    $('labTag').textContent = `Model sees: ${a.name}`;
    $('labTag').style.borderColor = a.color;
    const m = state.meta || {};
    let verdict = '';
    if (m.activity) {
      const ok = m.activity === a.id;
      verdict = m.simulated ? (ok ? ' The model agrees.' : ' The model disagrees with the generator, the kind of mistake it makes on real people too.')
        : (ok ? ' Correct.' : ' The model got this one wrong.');
    }
    const ranked = [...p.probabilities].map((q, i) => ({ q, i })).sort((x, y) => y.q.probability - x.q.probability);
    const second = ranked[1];
    $('verdict').innerHTML = `<div class="big" style="color:${a.color}">${a.name}</div><div class="sm">${(p.confidence * 100).toFixed(1)}% confident. Runner-up: ${ACT[second.i].name} (${(second.q.probability * 100).toFixed(1)}%). ${sourceText(m)}${verdict} Computed in ${r.latencyMs} ms on the server.</div>`;
    $('probs').innerHTML = ranked.map(({ q, i }, rank) => `<div class="prow ${rank === 0 ? 'top' : ''}"><span>${ACT[i].name}</span><span class="bar"><i style="width:${(q.probability * 100).toFixed(1)}%;background:${ACT[i].color}"></i></span><span>${(q.probability * 100).toFixed(1)}%</span></div>`).join('');
    drawPlot($('plotCanvas'), state.window, a.color);
    if (state.seq) drawSequence(state.seq, state.selected);
  }

  /* ----- mode: simulate ----- */
  const seg = $('actSeg');
  ACT.forEach((a) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.style.setProperty('--c', a.color);
    b.innerHTML = `<i style="background:${a.color}"></i>${a.name}`;
    b.setAttribute('aria-pressed', String(a.id === state.act));
    b.dataset.id = a.id;
    b.addEventListener('click', () => { state.act = a.id; seg.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.id === state.act))); refreshSim(); });
    seg.appendChild(b);
  });
  const refreshSim = guard(async () => {
    const q = new URLSearchParams({ activity: state.act, seed: state.seed, noise: state.noise, tilt: state.tilt });
    const r = await api(`/simulate?${q}`);
    await setWindow(r.window, r.meta);
  });
  let timer = null;
  const debounced = (fn) => { clearTimeout(timer); timer = setTimeout(fn, 160); };
  $('noise').addEventListener('input', (e) => { state.noise = +e.target.value; $('noiseOut').textContent = `${state.noise.toFixed(1)}×`; debounced(refreshSim); });
  $('tilt').addEventListener('input', (e) => { state.tilt = +e.target.value; $('tiltOut').textContent = `${state.tilt}°`; debounced(refreshSim); });
  $('seed').addEventListener('change', (e) => { state.seed = Math.max(1, Math.floor(+e.target.value || 1)); refreshSim(); });
  $('shuffle').addEventListener('click', () => { state.seed = 1 + Math.floor(Math.random() * 9999); $('seed').value = state.seed; refreshSim(); });

  /* ----- mode: set values ----- */
  const GROUPS = [
    ['Posture: gravity direction (g)', [['g_x', 'Total acc X', -1.5, 1.5, 0.01], ['g_y', 'Total acc Y', -1.5, 1.5, 0.01], ['g_z', 'Total acc Z', -1.5, 1.5, 0.01]]],
    ['Body acceleration strength (g)', [['ba_x', 'Body acc X', 0, 1, 0.001], ['ba_y', 'Body acc Y', 0, 1, 0.001], ['ba_z', 'Body acc Z', 0, 1, 0.001]]],
    ['Rotation strength (rad/s)', [['gy_x', 'Gyro X', 0, 2, 0.001], ['gy_y', 'Gyro Y', 0, 2, 0.001], ['gy_z', 'Gyro Z', 0, 2, 0.001]]],
    ['Movement', [['freq', 'Frequency (Hz)', 0.5, 5, 0.1], ['noise', 'Sensor noise', 0, 0.05, 0.001]]],
  ];
  const sl = $('valueSliders');
  GROUPS.forEach(([title, rows]) => {
    const h = document.createElement('div'); h.className = 'sgroup'; h.textContent = title; sl.appendChild(h);
    rows.forEach(([key, label, lo, hi, step]) => {
      const row = document.createElement('div'); row.className = 'srow';
      const dec = step < 0.01 ? 3 : step < 0.1 ? 2 : 1;
      row.innerHTML = `<label for="v_${key}" style="margin:0;font-weight:500">${label}</label><input type="range" id="v_${key}" min="${lo}" max="${hi}" step="${step}" value="${state.vals[key]}"><output id="o_${key}">${state.vals[key].toFixed(dec)}</output>`;
      sl.appendChild(row);
      row.querySelector('input').addEventListener('input', (e) => { state.vals[key] = +e.target.value; $(`o_${key}`).textContent = (+e.target.value).toFixed(dec); debounced(refreshValues); });
    });
  });
  const applyVals = (p) => {
    const v = { g_x: p.g[0], g_y: p.g[1], g_z: p.g[2], ba_x: p.ba[0], ba_y: p.ba[1], ba_z: p.ba[2], gy_x: p.gy[0], gy_y: p.gy[1], gy_z: p.gy[2], freq: p.freq, noise: p.noise };
    Object.entries(v).forEach(([k, val]) => {
      state.vals[k] = val;
      const el = $(`v_${k}`); el.value = val;
      const step = +el.step; $(`o_${k}`).textContent = val.toFixed(step < 0.01 ? 3 : step < 0.1 ? 2 : 1);
    });
  };
  const refreshValues = guard(async () => {
    const v = state.vals;
    const r = await api('/synthesize', post({ g: [v.g_x, v.g_y, v.g_z], ba: [v.ba_x, v.ba_y, v.ba_z], gy: [v.gy_x, v.gy_y, v.gy_z], freq: v.freq, noise: v.noise, seed: 1 }));
    await setWindow(r.window, r.meta);
  });
  const presetSeg = $('presetSeg');
  ACT.forEach((a) => {
    const b = document.createElement('button');
    b.type = 'button'; b.style.setProperty('--c', a.color);
    b.innerHTML = `<i style="background:${a.color}"></i>${a.name}`;
    b.addEventListener('click', guard(async () => { if (!state.presets) return; applyVals(state.presets.presets[a.id]); await refreshValues(); }));
    presetSeg.appendChild(b);
  });
  async function enterValues() {
    if (!state.presets) {
      try {
        state.presets = await api('/presets');
        $('presetNote').textContent = state.presets.source === 'dataset' ? 'Typical values computed from the real UCI test windows.' : 'Typical values computed from the built-in simulator (add the UCI dataset to use real averages).';
        applyVals(state.presets.presets.STANDING);
      } catch (err) { msg(err.message); }
    }
    await refreshValues();
  }

  /* ----- mode: real samples ----- */
  const realAct = $('realAct');
  realAct.innerHTML = `<option value="ALL">All activities</option>${ACT.map((a) => `<option value="${a.id}">${a.name}</option>`).join('')}`;
  const realMsg = (t) => { $('realMsg').textContent = t || ''; };
  const realLoad = guard(async (pos) => {
    const { pool } = state.real;
    if (!pool.length) return;
    state.real.pos = pos;
    $('realOut').textContent = `${pos + 1} of ${pool.length}`;
    const r = await api(`/samples/${pool[pos]}`);
    await setWindow(r.window, r.meta);
  });
  async function realPool() {
    const q = new URLSearchParams();
    if (realAct.value !== 'ALL') q.set('activity', realAct.value);
    if ($('realWrong').checked) q.set('wrong', 'true');
    try {
      const r = await api(`/samples/index?${q}`);
      state.real.pool = r.indices;
      const pos = $('realPos');
      pos.max = Math.max(0, r.count - 1); pos.value = 0;
      realMsg(r.count ? '' : 'No samples match this filter.');
      $('realOut').textContent = r.count ? `1 of ${r.count}` : '–';
      if (r.count) await realLoad(0);
    } catch (err) { realMsg(err.message); }
  }
  async function enterReal() {
    let h;
    try { h = await api('/health'); } catch (err) { realMsg(err.message); return; }
    const off = !h.dataset.available;
    [realAct, $('realWrong'), $('realRandom'), $('realPos')].forEach((el) => { el.disabled = off; });
    if (off) { realMsg('Real samples need the UCI HAR dataset on the server. Extract it so that data/UCI HAR Dataset/test/y_test.txt exists, then restart (see the README).'); return; }
    await realPool();
  }
  realAct.addEventListener('change', realPool);
  $('realWrong').addEventListener('change', realPool);
  $('realPos').addEventListener('input', (e) => debounced(() => realLoad(+e.target.value)));
  $('realRandom').addEventListener('click', () => {
    const n = state.real.pool.length;
    if (!n) return;
    const p = Math.floor(Math.random() * n);
    $('realPos').value = p;
    realLoad(p);
  });

  /* ----- mode: upload ----- */
  $('file').addEventListener('change', guard(async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    e.target.value = '';
    const name = f.name.toLowerCase();
    const raw = name.endsWith('.npy') ? parseNpy(await f.arrayBuffer()) : parseText(await f.text(), name);
    const rows = toRows(raw);
    if (rows.length === 128) {
      state.seq = null; $('seqPanel').hidden = true;
      msg(`Loaded ${f.name}: one window.`);
      await setWindow(rows, { upload: f.name });
      return;
    }
    msg(`Loaded ${f.name}: ${rows.length} rows (${(rows.length / 50).toFixed(1)} s). Classifying every window…`);
    const result = await api('/predict-sequence', post({ data: rows }));
    state.seq = { rows, result, name: f.name };
    $('seqPanel').hidden = false;
    $('seqTitle').textContent = `Recording timeline: ${result.summary.windows} windows, ${result.summary.seconds} s`;
    msg(`${result.summary.windows} windows classified in ${result.latencyMs} ms.`);
    await selectWindow(0);
  }));
  async function selectWindow(i) {
    const { rows, result, name } = state.seq;
    state.selected = i;
    await setWindow(rows.slice(i * 64, i * 64 + 128), { upload: name, seqIndex: i, seqCount: result.windows.length });
  }
  $('timelineCanvas').addEventListener('click', guard(async (e) => {
    if (!state.seq) return;
    const r = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - r.left) / r.width;
    const i = Math.min(state.seq.result.windows.length - 1, Math.max(0, Math.floor(frac * state.seq.rows.length / 64)));
    await selectWindow(i);
  }));
  $('dlExample').addEventListener('click', () => {
    if (!state.window) return;
    const csv = `${CHANNELS.join(',')}\n${state.window.map((r) => r.join(',')).join('\n')}\n`;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'example_window.csv';
    a.click();
    URL.revokeObjectURL(a.href);
  });

  /* ----- tabs ----- */
  const MODES = [['sim', 'Simulate'], ['values', 'Set values'], ['real', 'Real samples'], ['upload', 'Your recording']];
  const tabs = $('labTabs');
  function switchMode(m) {
    state.mode = m;
    msg('');
    tabs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.m === m)));
    MODES.forEach(([k]) => { $(`p-${k}`).hidden = k !== m; });
    $('seqPanel').hidden = !(m === 'upload' && state.seq);
    if (m === 'sim') refreshSim();
    else if (m === 'values') enterValues();
    else if (m === 'real') enterReal();
    else if (state.seq) selectWindow(state.selected < 0 ? 0 : state.selected);
  }
  MODES.forEach(([k, label]) => {
    const b = document.createElement('button');
    b.type = 'button'; b.setAttribute('role', 'tab'); b.dataset.m = k; b.textContent = label;
    b.addEventListener('click', () => switchMode(k));
    tabs.appendChild(b);
  });

  window.addEventListener('resize', () => {
    if (!state.window) return;
    drawPlot($('plotCanvas'), state.window, ACT.find((x) => x.id === shown).color);
    if (state.seq) drawSequence(state.seq, state.selected);
  });
  switchMode('sim');
}
