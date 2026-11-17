import { THREE, makeStage, makeLabel, ACT, hexNum, softTexture } from './common.js';

const KIND_COLOR = { input: 0xeceaff, cnn: 0x4dd6c1, lstm: 0xb38cff, dense: 0x6fa8ff, output: 0xffb84d };
// Where each layer lives in the scene and how large its plate is.
const SPEC = [
  { id: 'input', x: 0, size: [3.4, 1.2], tex: [128, 9] },
  { id: 'conv1', x: 5.2, size: [2.3, 2.3], tex: [64, 64] },
  { id: 'conv2', x: 9.8, size: [1.5, 2.9], tex: [32, 128] },
  { id: 'lstm1', x: 14, size: [1.5, 2.9], tex: [32, 128] },
  { id: 'lstm2', x: 18, size: [0.45, 2.3], tex: [1, 64] },
  { id: 'dense', x: 21, size: [0.45, 2.3], tex: [1, 64] },
  { id: 'softmax', x: 24.6, size: [3, 2.3], tex: null },
];

function ramp(u) {
  const stops = [[0, [26, 31, 74]], [0.35, [58, 90, 160]], [0.6, [77, 214, 193]], [0.85, [255, 184, 77]], [1, [255, 245, 214]]];
  u = Math.min(1, Math.max(0, u));
  for (let i = 1; i < stops.length; i++) {
    if (u <= stops[i][0]) {
      const [u0, a] = stops[i - 1];
      const [u1, b] = stops[i];
      const f = (u - u0) / (u1 - u0);
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
    }
  }
  return stops[4][1];
}

export function initArch(layers, onSelect) {
  const canvas = document.getElementById('archCanvas');
  const tip = document.getElementById('archTip');
  const home = { pos: new THREE.Vector3(12.3, 3.2, 21), tgt: new THREE.Vector3(12.3, 0, 0) };
  const stage = makeStage(canvas, { fov: 40, pos: home.pos.toArray(), target: home.tgt.toArray(), orbit: true });
  const { scene, camera, controls } = stage;
  controls.maxDistance = 45;
  controls.minDistance = 4;
  let goal = null;
  controls.addEventListener('start', () => { goal = null; });

  const meta = Object.fromEntries(layers.map((l) => [l.id, l]));
  const plates = [];
  const pickables = [];
  const stats = {};
  let selected = -1;

  SPEC.forEach((sp, i) => {
    const info = meta[sp.id] || { name: sp.id, kind: 'dense', shape: [] };
    const color = KIND_COLOR[info.kind] || 0xffffff;
    const group = new THREE.Group();
    group.position.x = sp.x;
    scene.add(group);
    const label = makeLabel(info.name, { size: 0.42, sub: info.shape.join(' × ') });
    label.position.set(0, sp.size[1] / 2 + 0.7, 0);
    group.add(label);

    if (sp.tex) {
      const [w, h] = sp.tex;
      const data = new Uint8Array(w * h * 4);
      for (let p = 0; p < w * h; p++) { const v = ramp(0.04 + 0.08 * Math.abs(Math.sin(p * 12.9898))); data.set([v[0], v[1], v[2], 255], p * 4); }
      const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
      tex.magFilter = THREE.NearestFilter;
      tex.minFilter = THREE.NearestFilter;
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      const geo = new THREE.PlaneGeometry(sp.size[0], sp.size[1]);
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, toneMapped: false }));
      group.add(mesh);
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color }));
      group.add(edge);
      const halo = new THREE.Mesh(new THREE.PlaneGeometry(sp.size[0] * 1.8 + 1.2, sp.size[1] * 1.6 + 1.2), new THREE.MeshBasicMaterial({ map: softTexture(), color, transparent: true, opacity: 0.28, blending: THREE.AdditiveBlending, depthWrite: false }));
      halo.position.z = -0.08;
      group.add(halo);
      mesh.userData.layer = i;
      pickables.push(mesh);
      plates.push({ i, group, mesh, edge, halo, tex, w, h, id: sp.id });
    } else {
      // softmax: six bars
      const bars = [];
      ACT.forEach((a, k) => {
        const bar = new THREE.Mesh(new THREE.BoxGeometry(0.34, 1, 0.34), new THREE.MeshBasicMaterial({ color: hexNum(a.color) }));
        bar.position.set((k - 2.5) * 0.52, -1.15, 0);
        bar.scale.y = 0.03;
        bar.userData.layer = i;
        group.add(bar);
        pickables.push(bar);
        const lab = makeLabel(a.short, { size: 0.26, color: a.color });
        lab.position.set((k - 2.5) * 0.52, -1.5, 0);
        group.add(lab);
        bars.push({ bar, goal: 0.03 });
      });
      const base = new THREE.Mesh(new THREE.PlaneGeometry(3.1, 0.02), new THREE.MeshBasicMaterial({ color }));
      base.position.y = -1.15;
      group.add(base);
      const halo = new THREE.Mesh(new THREE.PlaneGeometry(6, 5), new THREE.MeshBasicMaterial({ map: softTexture(), color, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false }));
      halo.position.z = -0.08;
      group.add(halo);
      plates.push({ i, group, bars, halo, id: sp.id });
    }
  });

  // faint wires between consecutive layers
  const wire = new THREE.LineBasicMaterial({ color: 0x6f78c8, transparent: true, opacity: 0.25 });
  for (let i = 0; i < SPEC.length - 1; i++) {
    const a = SPEC[i];
    const b = SPEC[i + 1];
    const pts = [];
    for (const sy of [-1, 1]) {
      pts.push(new THREE.Vector3(a.x + a.size[0] / 2, sy * a.size[1] / 2, 0), new THREE.Vector3(b.x - b.size[0] / 2, sy * b.size[1] / 2, 0));
    }
    scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), wire));
  }

  // pulses travelling through the network
  const pulses = [];
  for (let k = 0; k < 18; k++) {
    const m = new THREE.Sprite(new THREE.SpriteMaterial({ map: softTexture(), color: 0xffb84d, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    m.scale.set(0.45, 0.45, 1);
    scene.add(m);
    pulses.push({ m, seg: k % (SPEC.length - 1), off: Math.random(), lane: (Math.random() - 0.5) * 1.6 });
  }

  // starfield behind the network
  const starPos = new Float32Array(600 * 3);
  for (let k = 0; k < 600; k++) { starPos[k * 3] = -8 + Math.random() * 42; starPos[k * 3 + 1] = -9 + Math.random() * 18; starPos[k * 3 + 2] = -16 + Math.random() * 10; }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0x9aa6ff, size: 0.12, map: softTexture(), transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false })));

  let waveT = 99;

  function heat(plate, values, shape, signed) {
    const [T, CH] = shape.length === 2 ? shape : [1, shape[0]];
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-9) hi = lo + 1;
    const out = plate.tex.image.data;
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < CH; c++) {
        const u = (values[t * CH + c] - lo) / (hi - lo);
        const col = ramp(signed ? u : Math.pow(u, 0.8));
        out.set([col[0], col[1], col[2], 255], (c * T + t) * 4);
      }
    }
    plate.tex.needsUpdate = true;
    let sum = 0;
    let act = 0;
    for (const v of values) { sum += Math.abs(v); if (Math.abs(v) > 1e-3) act++; }
    stats[plate.id] = { mean: sum / values.length, max: hi, active: act / values.length };
  }

  function setPrediction({ result, window }) {
    waveT = 0;
    const byId = Object.fromEntries(plates.map((p) => [p.id, p]));
    // input: per-channel normalised window
    if (window) {
      const rows = window.length;
      const flat = new Float32Array(rows * 9);
      for (let c = 0; c < 9; c++) {
        let lo = Infinity;
        let hi = -Infinity;
        for (let t = 0; t < rows; t++) { const v = window[t][c]; if (v < lo) lo = v; if (v > hi) hi = v; }
        const span = hi - lo < 1e-6 ? 1 : hi - lo;
        for (let t = 0; t < rows; t++) flat[t * 9 + c] = (window[t][c] - lo) / span;
      }
      heat(byId.input, flat, [rows, 9], true);
    }
    const a = result.activations;
    if (a) {
      heat(byId.conv1, a.conv1.data, a.conv1.shape, false);
      heat(byId.conv2, a.conv2.data, a.conv2.shape, false);
      heat(byId.lstm1, a.lstm1.data, a.lstm1.shape, true);
      heat(byId.lstm2, a.lstm2.data, a.lstm2.shape, true);
      heat(byId.dense, a.dense.data, a.dense.shape, false);
    }
    const sm = byId.softmax;
    result.prediction.probabilities.forEach((p, k) => { sm.bars[k].goal = Math.max(0.03, p.probability * 2.2); });
    stats.softmax = { mean: result.prediction.confidence, max: result.prediction.confidence, active: 1 };
    if (selected >= 0) onSelect(selected, stats[SPEC[selected].id]);
  }

  window.addEventListener('har:predict', (e) => setPrediction(e.detail));

  function select(i, fly = true) {
    selected = i;
    plates.forEach((p) => { if (p.edge) p.edge.material.color.setHex(p.i === i ? 0xffffff : KIND_COLOR[(meta[p.id] || {}).kind] || 0xffffff); });
    if (fly) {
      const sp = SPEC[i];
      goal = { pos: new THREE.Vector3(sp.x + 1.2, 0.8, Math.max(6, sp.size[1] * 2.1)), tgt: new THREE.Vector3(sp.x, 0, 0) };
    }
    onSelect(i, stats[SPEC[i].id]);
  }
  function reset() { goal = { pos: home.pos.clone(), tgt: home.tgt.clone() }; }
  document.getElementById('archReset').addEventListener('click', reset);

  // hover + click
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let down = null;
  const pick = (ev) => {
    const r = canvas.getBoundingClientRect();
    ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObjects(pickables, false)[0];
    return hit ? hit.object.userData.layer : -1;
  };
  canvas.addEventListener('pointermove', (ev) => {
    const i = pick(ev);
    canvas.style.cursor = i >= 0 ? 'pointer' : 'grab';
    if (i < 0) { tip.hidden = true; return; }
    const info = meta[SPEC[i].id] || {};
    const r = canvas.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = `${Math.min(ev.clientX - r.left + 14, r.width - 270)}px`;
    tip.style.top = `${ev.clientY - r.top + 14}px`;
    tip.innerHTML = `<b>${info.name || SPEC[i].id}</b><br>Output ${info.shape ? info.shape.join(' × ') : ''} · ${(info.params || 0).toLocaleString()} parameters`;
  });
  canvas.addEventListener('pointerleave', () => { tip.hidden = true; });
  canvas.addEventListener('pointerdown', (ev) => { down = [ev.clientX, ev.clientY]; });
  canvas.addEventListener('pointerup', (ev) => {
    if (down && Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) < 5) { const i = pick(ev); if (i >= 0) select(i); }
    down = null;
  });

  stage.onFrame((dt, now) => {
    if (goal) {
      const k = 1 - Math.exp(-4 * dt);
      camera.position.lerp(goal.pos, k);
      controls.target.lerp(goal.tgt, k);
      if (camera.position.distanceTo(goal.pos) < 0.05) goal = null;
    }
    waveT += dt;
    plates.forEach((p) => {
      const d = waveT - p.i * 0.3;
      const f = d > 0 ? Math.exp(-d * 3.2) : 0;
      if (p.mesh) p.mesh.material.color.setScalar(1 + 0.9 * f);
      if (p.halo) p.halo.material.opacity = (p.bars ? 0.22 : 0.28) + 0.55 * f;
    });
    plates.forEach((p) => {
      if (p.bars) p.bars.forEach((b) => { b.bar.scale.y += (b.goal - b.bar.scale.y) * (1 - Math.exp(-6 * dt)); b.bar.position.y = -1.15 + b.bar.scale.y / 2; });
    });
    for (const p of pulses) {
      const u = (now * (stage.reduce ? 0.05 : 0.22) + p.off) % 1;
      const a = SPEC[p.seg];
      const b = SPEC[p.seg + 1];
      const ax = a.x + a.size[0] / 2;
      const bx = b.x - b.size[0] / 2;
      p.m.position.set(ax + (bx - ax) * u, p.lane * (a.size[1] + (b.size[1] - a.size[1]) * u) * 0.4, 0);
    }
  });

  select(1, false);
  return { select, stats, count: SPEC.length };
}
