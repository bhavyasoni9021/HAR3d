import './styles.css';
import { ACT, api } from './common.js';
import { initHero } from './hero.js';
import { initArch } from './arch.js';
import { initLab } from './lab.js';
import { initConfusion } from './confusion.js';

const $ = (id) => document.getElementById(id);
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const col = (id) => ACT.find((a) => a.id === id).color;
const nm = (id) => ACT.find((a) => a.id === id).name;

/* ---------- presentation helpers ---------- */
function countUp(el, to, dec, suffix) {
  const fmt = (v) => (dec ? v.toFixed(dec) : Math.round(v).toLocaleString()) + suffix;
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { el.textContent = fmt(to); return; }
  const t0 = performance.now();
  const step = (n) => {
    const u = Math.min(1, (n - t0) / 1800);
    el.textContent = fmt(to * (1 - Math.pow(1 - u, 3)));
    if (u < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function initReveal() {
  if (!('IntersectionObserver' in window)) return;
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { rootMargin: '0px 0px -8% 0px', threshold: 0.04 });
  document.querySelectorAll('.sec .wrap > *').forEach((el) => { el.classList.add('reveal'); io.observe(el); });
}

function initProgress() {
  const bar = $('progress');
  let queued = false;
  const update = () => {
    queued = false;
    const max = document.documentElement.scrollHeight - innerHeight;
    bar.style.transform = `scaleX(${max > 0 ? Math.min(1, scrollY / max) : 0})`;
  };
  addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(update); } }, { passive: true });
  update();
}

/* ---------- theory ---------- */
const THEORY = [
  { k: 'CNN', title: 'Convolutional neural networks', body: `<p>A CNN learns small filters that slide over the input and fire wherever a local pattern appears. For sensor data the input is one-dimensional in time with several channels, so a kernel of size k spans k time steps across all C channels and yields one value per position.</p><p>Two properties make it efficient: <b>local connectivity</b> (each output sees a short stretch) and <b>weight sharing</b> (the same filter is used everywhere, so a step is detected wherever it happens). Pooling downsamples and adds shift robustness; stacking blocks builds simple motions into abstract ones. Batch normalisation steadies training and dropout fights over-fitting.</p><p><b>In this project:</b> two Conv1D blocks (64 then 128 filters, kernel 5) turn 128 raw steps into 32 rich feature vectors.</p>`, eq: ['y_f(t) = σ( Σᵢ Σ_c w_{f,i,c} · x(t+i, c) + b_f )', 'σ = ReLU,  i = 0…k−1,  c = 1…C'] },
  { k: 'LSTM', title: 'Recurrent networks and LSTM', body: `<p>Activities are defined by how motion evolves over seconds, and a CNN’s receptive field is short. Recurrent networks carry a hidden state through the sequence, but plain RNNs suffer vanishing and exploding gradients.</p><p>The LSTM adds a cell state and three gates. The <b>forget gate</b> discards memory, the <b>input gate</b> stores new information, and the <b>output gate</b> exposes the hidden state. Because the cell state updates additively, gradients flow across many steps.</p><p><b>In this project:</b> LSTM(128) returns a state at every step, then LSTM(64) summarises the window.</p>`, eq: ['fₜ = σ(W_f[hₜ₋₁, xₜ] + b_f)', 'iₜ = σ(W_i[hₜ₋₁, xₜ] + b_i)', 'oₜ = σ(W_o[hₜ₋₁, xₜ] + b_o)', 'c̃ₜ = tanh(W_c[hₜ₋₁, xₜ] + b_c)', 'cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ', 'hₜ = oₜ ⊙ tanh(cₜ)'] },
  { k: 'Transfer', title: 'Transfer learning', body: `<p>Lower layers learn generic features (edges in images, short motion primitives in sensor signals) while higher layers become task-specific. The recipe: copy the pretrained convolutional layers, freeze the earliest, fine-tune the later ones with a small learning rate and attach a new classification head.</p><p><b>For HAR:</b> a CNN pretrained on many subjects can be fine-tuned with minutes of labelled data from a new device, sensor placement (wrist instead of waist) or user. This project trains from scratch because UCI HAR is large enough, but its CNN extractor is a natural candidate for reuse, and per-user head fine-tuning targets the weak subjects 9 and 10.</p>`, eq: ['pretrained conv blocks → freeze early, fine-tune late → new 6-class head'] },
  { k: 'Attention', title: 'Attention mechanism', body: `<p>The baseline LSTM summarises the whole window in its last hidden state. Attention instead scores every time step, normalises the scores with a softmax and forms a weighted sum, so the network can focus on the most informative moments such as the transition into a sit or a stair step.</p><p>The weights are interpretable: plotting αᵢ over time shows which part of the signal drove the decision. Self-attention, the heart of Transformers, lets every step attend to every other.</p><p><b>Proposed here:</b> attention pooling over the 32 LSTM outputs to sharpen the sitting/standing boundary.</p>`, eq: ['eᵢ = vᵀ tanh(W hᵢ + b)', 'αᵢ = exp(eᵢ) / Σⱼ exp(eⱼ)', 'context = Σᵢ αᵢ hᵢ'] },
  { k: 'GAN', title: 'Generative adversarial networks', body: `<p>A generator maps random noise to synthetic samples while a discriminator tries to tell real from fake. They play a minimax game until generated samples are hard to distinguish from real data.</p><p><b>For HAR:</b> conditional or time-series GANs can create realistic windows for under-represented activities or subjects, a form of augmentation, or help adapt between devices. GANs are less stable on time series than on images, so jitter, scaling, rotation and time-warping are tried first.</p>`, eq: ['min_G max_D  E_x[log D(x)] + E_z[log(1 − D(G(z)))]'] },
];

function initTheory() {
  const tabs = $('theoryTabs');
  const show = (i) => {
    tabs.querySelectorAll('button').forEach((b, j) => b.setAttribute('aria-selected', String(i === j)));
    const t = THEORY[i];
    $('theoryBody').innerHTML = `<div><h3 style="margin-top:0">${t.title}</h3>${t.body}</div><div class="eq" aria-label="Equations">${t.eq.map((l) => `<div>${l}</div>`).join('')}</div>`;
  };
  THEORY.forEach((t, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.setAttribute('role', 'tab'); b.textContent = t.k;
    b.addEventListener('click', () => show(i));
    tabs.appendChild(b);
  });
  show(0);
}

/* ---------- sliding-window canvas ---------- */
function initSlicing() {
  const cv = $('sliceCanvas');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const SEC = 9;
  const sig = (t) => 0.55 * Math.sin(2 * Math.PI * 1.9 * t) + 0.2 * Math.sin(2 * Math.PI * 3.8 * t + 1) + 0.1 * Math.sin(2 * Math.PI * 7.1 * t + 2);
  let step = 0;
  let pos = 0;
  let last = performance.now();
  let visible = true;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(cv);
  function draw(now) {
    requestAnimationFrame(draw);
    if (!visible) return;
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (!reduce) step += dt;
    const target = (Math.floor(step / 1.6) % 5) * 1.28;
    pos += (target - pos) * Math.min(1, dt * 7);
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const W = cv.clientWidth; const H = 190;
    if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const x = (t) => 10 + (t / SEC) * (W - 20);
    const prev = pos - 1.28;
    if (prev >= 0) { g.fillStyle = 'rgba(255,184,77,.10)'; g.fillRect(x(prev), 8, x(2.56) - x(0), H - 36); }
    g.fillStyle = 'rgba(255,184,77,.22)'; g.fillRect(x(pos), 8, x(2.56) - x(0), H - 36);
    g.strokeStyle = '#ffb84d'; g.lineWidth = 1.5; g.strokeRect(x(pos), 8, x(2.56) - x(0), H - 36);
    g.beginPath(); g.strokeStyle = '#eceaff'; g.lineWidth = 1.6;
    for (let i = 0; i <= 450; i++) { const t = (i / 450) * SEC; const px = x(t); const py = 8 + (H - 36) / 2 - sig(t) * (H - 50) / 1.6; if (i) g.lineTo(px, py); else g.moveTo(px, py); }
    g.stroke();
    g.fillStyle = '#a9acd6'; g.font = '12px "Instrument Sans", sans-serif'; g.textAlign = 'center';
    for (let s = 0; s <= SEC; s += 1) g.fillText(`${s}s`, x(s), H - 8);
    g.textAlign = 'left'; g.fillStyle = '#ffb84d';
    g.fillText('current window (2.56 s, 128 samples)', Math.min(x(pos) + 4, W - 230), 22);
  }
  requestAnimationFrame(draw);
}

/* ---------- content from the API ---------- */
function fillReport(r, metrics) {
  countUp($('hAcc'), metrics.accuracy, 2, '%');
  countUp($('hParams'), r.headline.parameters, 0, '');
  $('datasetTable').innerHTML = `<tbody>${r.dataset.rows.map(([a, b]) => `<tr><th scope="row">${a}</th><td>${b}</td></tr>`).join('')}</tbody>`;
  const mx = Math.max(...r.dataset.trainCounts, ...r.dataset.testCounts);
  $('classBars').innerHTML = ACT.map((a, i) => `<div class="row"><span><i class="dot" style="background:${a.color}"></i>${a.name}</span><div><div class="tr" style="width:${100 * r.dataset.trainCounts[i] / mx}%;background:${a.color}"></div><div class="tr" style="width:${100 * r.dataset.testCounts[i] / mx}%;background:${a.color};opacity:.45"></div><div class="cap">train ${r.dataset.trainCounts[i].toLocaleString()} · test ${r.dataset.testCounts[i]}</div></div></div>`).join('');
  $('trainTable').innerHTML = `<tbody>${r.training.map(([a, b]) => `<tr><th scope="row">${a}</th><td>${b}</td></tr>`).join('')}</tbody>`;
  $('layerTable').innerHTML = `<thead><tr><th>Layer</th><th>Output</th><th>Params</th></tr></thead><tbody>${r.layers.map((l) => `<tr><td>${l.name}</td><td>${l.shape.join(' × ')}</td><td>${l.params.toLocaleString()}</td></tr>`).join('')}<tr><th>Total</th><td></td><th>${r.headline.parameters.toLocaleString()}</th></tr></tbody>`;
  $('errTable').innerHTML = `<thead><tr><th>True → predicted</th><th>Count</th><th>% of class</th><th>% of errors</th></tr></thead><tbody>${r.topErrors.map((e) => `<tr><td><i class="dot" style="background:${col(e.from)}"></i>${nm(e.from)} → ${nm(e.to)}</td><td>${e.count}</td><td>${e.pctOfClass}%</td><td>${e.pctOfErrors}%</td></tr>`).join('')}</tbody>`;
  $('roadmapList').innerHTML = `<div class="head"><span>Strategy</span><span>Weakness addressed</span><span>Expected impact</span><span>Effort</span></div>${r.improvements.map((s) => `<details><summary><b>${s.strategy}</b><span>${s.weakness}</span><span>${s.impact}</span><span>${s.effort}</span></summary><div class="how">${s.how}</div></details>`).join('')}`;
  $('futureList').innerHTML = r.futureWork.map((f) => `<li>${f}</li>`).join('');
  $('refList').innerHTML = r.references.map((f) => `<li>${f}</li>`).join('');
  const p = r.project;
  $('teamCourse').textContent = `${p.course}. ${p.degree}.`;
  $('teamTable').innerHTML = `<thead><tr><th>Name</th><th>Register number</th></tr></thead><tbody>${p.team.map((t) => `<tr><td>${t.name}</td><td>${t.reg}</td></tr>`).join('')}<tr><td>${p.supervisor.name}<br><small>${p.supervisor.role}</small></td><td>Supervisor</td></tr><tr><td>${p.head.name}<br><small>${p.head.role}</small></td><td>Head of department</td></tr></tbody>`;
  $('teamInst').textContent = `${p.department}, ${p.institution}. ${p.session}, ${p.month}.`;
  $('repoLink').href = p.repo;
  $('apiTable').innerHTML = `<tbody>${[['GET /api/report', 'Project content and tables'], ['GET /api/model', 'Layers, shapes, normalisation'], ['GET /api/metrics', 'Confusion matrix, per-class, per-subject'], ['GET /api/simulate', 'Synthetic sensor window'], ['POST /api/predict', 'Runs the trained CNN-LSTM, returns activations']].map(([a, b]) => `<tr><th scope="row"><code>${a}</code></th><td>${b}</td></tr>`).join('')}</tbody>`;
}

function fillMetrics(m) {
  $('resultsLead').innerHTML = `The saved best model scores <b>${m.accuracy.toFixed(2)}%</b> (${m.correct.toLocaleString()} of ${m.testWindows.toLocaleString()} windows) with a macro F1 of <b>${m.macroF1}</b>, on subjects that were never part of training.`;
  $('classTable').innerHTML = `<thead><tr><th>Activity</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>${m.perClass.map((c) => `<tr><td><i class="dot" style="background:${col(c.id)}"></i>${nm(c.id)}</td><td>${c.precision.toFixed(3)}</td><td>${c.recall.toFixed(3)}</td><td>${c.f1.toFixed(3)}</td><td>${c.support}</td></tr>`).join('')}</tbody>`;
  $('subjectBars').innerHTML = m.subjects.map((s) => `<div class="r"><span>Subject ${s.id}</span><div class="b"><i class="${s.accuracy < 90 ? 'low' : ''}" style="width:${Math.max(0, (s.accuracy - 50) * 2)}%"></i></div><span>${s.accuracy.toFixed(1)}%${s.approx ? '*' : ''}</span></div>`).join('');
  $('subjectNote').textContent = m.source === 'computed' ? 'Computed by the server on the real test split. The scale starts at 50%.' : 'Subjects marked * are read from the report’s Figure 3.5 and are approximate. The scale starts at 50%.';
  $('metricSource').textContent = m.note;
  const c = m.confidence || { correct: 0.964, wrong: 0.806 };
  $('confBars').innerHTML = [['Correct', c.correct, '#4dd6c1'], ['Wrong', c.wrong, '#ff6b8a']].map(([l, v, cc]) => `<div class="r"><span>${l}</span><div class="b"><i style="width:${v * 100}%;background:${cc}"></i></div><span>${v.toFixed(3)}</span></div>`).join('');
}

function layerPanel(layers) {
  const bar = $('layerBar');
  const info = $('layerInfo');
  let arch;
  const paint = (i, st) => {
    const l = layers[i];
    bar.querySelectorAll('button').forEach((b, j) => b.setAttribute('aria-selected', String(i === j)));
    const live = st ? `<div class="kv"><div><b>${st.mean.toFixed(3)}</b>mean activation</div><div><b>${(st.active * 100).toFixed(0)}%</b>units firing</div></div>` : '';
    info.innerHTML = `<h3>${l.name}</h3><div class="kv"><div><b>${l.shape.join(' × ')}</b>output shape</div><div><b>${l.params.toLocaleString()}</b>parameters</div></div><p>${l.detail}</p>${l.paramBreakdown ? `<p class="note">${l.paramBreakdown}</p>` : ''}${live}`;
  };
  layers.forEach((l, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.setAttribute('role', 'tab'); b.textContent = l.name;
    b.addEventListener('click', () => arch && arch.select(i));
    bar.appendChild(b);
  });
  arch = initArch(layers, paint);
}

function watchStatus() {
  const el = $('status');
  const tick = async () => {
    try {
      const h = await api('/health');
      el.className = 'status ok';
      el.querySelector('b').textContent = h.dataset.available ? 'API online · real data' : 'API online';
    } catch { el.className = 'status bad'; el.querySelector('b').textContent = 'API offline'; }
  };
  tick(); setInterval(tick, 15000);
}

function watchNav() {
  const links = [...document.querySelectorAll('.nav nav a')];
  const map = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { links.forEach((l) => l.classList.remove('on')); (map.get(e.target.id) || { classList: { add() {} } }).classList.add('on'); } }), { rootMargin: '-45% 0px -50% 0px' });
  map.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });
}

async function boot() {
  initReveal();
  initProgress();
  initTheory();
  initSlicing();
  watchNav();
  watchStatus();
  try { initHero(); } catch (e) { console.warn('3D hero unavailable', e); }
  try {
    const [report, metrics, model] = await Promise.all([api('/report'), api('/metrics'), api('/model')]);
    fillReport(report, metrics);
    fillMetrics(metrics);
    try { layerPanel(model.layers); } catch (e) { console.warn('3D architecture unavailable', e); }
    try { initConfusion(metrics); } catch (e) { console.warn('3D confusion matrix unavailable', e); }
  } catch (err) {
    document.querySelectorAll('.lead').forEach((n, i) => { if (i === 0) n.insertAdjacentHTML('afterend', `<p class="note">The backend is not reachable (${err.message}). Start it with <code>npm run dev</code>.</p>`); });
  }
  try { initLab(); } catch (e) { console.warn('3D lab unavailable', e); }
}
boot();
