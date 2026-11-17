import { THREE, makeStage, ACT, api, periodic, fallbackWindow, hexNum, softTexture } from './common.js';
import { Mannequin, Floor } from './mannequin.js';

export function initHero() {
  const canvas = document.getElementById('heroCanvas');
  const chip = document.getElementById('heroChip');
  const modelLine = document.getElementById('heroModel');
  const stage = makeStage(canvas, { fov: 42, pos: [2.9, 1.7, 5.4], target: [1.3, 1, 0] });
  const { scene, camera } = stage;
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));
  const sun = new THREE.DirectionalLight(0xffffff, 2.0);
  sun.position.set(3, 6, 4);
  scene.add(sun);
  const rim = new THREE.PointLight(0xffb84d, 18, 9);
  rim.position.set(0.2, 1.6, -1.2);
  scene.add(rim);

  const man = new Mannequin(ACT[0].color);
  man.group.position.x = 1.3;
  scene.add(man.group);
  const floor = new Floor();
  floor.group.position.x = 1.3;
  scene.add(floor.group);

  // drifting particle field
  const PN = 700;
  const ppos = new Float32Array(PN * 3);
  const pspd = new Float32Array(PN);
  for (let i = 0; i < PN; i++) { ppos[i * 3] = -6 + Math.random() * 16; ppos[i * 3 + 1] = Math.random() * 6; ppos[i * 3 + 2] = -5 + Math.random() * 8; pspd[i] = 0.12 + Math.random() * 0.35; }
  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute('position', new THREE.BufferAttribute(ppos, 3));
  const pmat = new THREE.PointsMaterial({ color: hexNum(ACT[0].color), size: 0.07, map: softTexture(), transparent: true, opacity: 0.75, blending: THREE.AdditiveBlending, depthWrite: false });
  scene.add(new THREE.Points(pgeo, pmat));
  const pTarget = new THREE.Color(ACT[0].color);

  // nine sensor traces: a thin line plus glowing beads
  const N = 112;
  const ribbons = [];
  const beads = [];
  const X0 = 1.9;
  for (let k = 0; k < 9; k++) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: hexNum(ACT[0].color), transparent: true, opacity: 0.3 + 0.5 * (1 - k / 12) }));
    scene.add(line);
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: hexNum(ACT[0].color), size: 0.12, map: softTexture(), transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
    scene.add(pts);
    ribbons.push(line);
    beads.push(pts);
  }
  // scanning beam that sweeps across the traces
  const beam = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 3.4), new THREE.MeshBasicMaterial({ map: softTexture(), color: 0xffffff, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false }));
  beam.position.set(X0, 1.4, -0.3);
  scene.add(beam);

  const scale = [0.5, 0.5, 0.5, 0.22, 0.22, 0.22, 0.5, 0.5, 0.5];
  let seq = periodic(fallbackWindow('WALKING'));
  const means = new Array(9).fill(0);
  const setSeq = (rows) => {
    seq = periodic(rows);
    for (let c = 0; c < 9; c++) means[c] = seq.reduce((a, r) => a + r[c], 0) / seq.length;
  };
  setSeq(fallbackWindow('WALKING'));

  let idx = 0;
  let tAct = 0;
  const load = async (i) => {
    modelLine.textContent = 'Model is reading the window…';
    let rows;
    try { rows = (await api(`/simulate?activity=${ACT[i].id}&seed=7`)).window; setSeq(rows); } catch { setSeq(fallbackWindow(ACT[i].id)); modelLine.textContent = 'Backend offline'; return; }
    try {
      const r = await api('/predict', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ window: rows }) });
      if (idx !== i) return;
      const p = r.prediction;
      modelLine.textContent = `Model says ${ACT[p.index].name} · ${(p.confidence * 100).toFixed(1)}%`;
    } catch { modelLine.textContent = ''; }
  };
  const apply = (i) => {
    idx = i;
    const col = ACT[i].color;
    man.setColor(col);
    man.burst();
    rim.color.set(col);
    pTarget.set(col);
    ribbons.forEach((l) => l.material.color.setHex(hexNum(col)));
    beads.forEach((b) => b.material.color.setHex(hexNum(col)));
    chip.querySelector('i').style.background = col;
    chip.querySelector('span').textContent = ACT[i].name;
    load(i);
  };
  apply(0);
  const slow = stage.reduce ? 0.25 : 1;

  // mouse parallax + intro dolly
  let mx = 0; let my = 0; let tx = 0; let ty = 0; let elapsed = 0;
  window.addEventListener('pointermove', (e) => { tx = (e.clientX / window.innerWidth) * 2 - 1; ty = (e.clientY / window.innerHeight) * 2 - 1; });

  stage.onFrame((dt, now) => {
    elapsed += dt;
    tAct += dt;
    if (tAct > 4.6) { tAct = 0; apply((idx + 1) % ACT.length); }
    const t = now * slow;
    man.update(ACT[idx].id, t, dt);
    floor.update(ACT[idx].id, dt * slow);

    mx += (tx - mx) * (1 - Math.exp(-3 * dt));
    my += (ty - my) * (1 - Math.exp(-3 * dt));
    const u = stage.reduce ? 1 : Math.min(1, elapsed / 2.6);
    const e = 1 - Math.pow(1 - u, 3);
    const bz = camera.aspect < 1 ? 8.5 : 5.4;
    camera.position.set(2.9 + mx * 0.55 - (1 - e) * 2.4, 1.7 - my * 0.25 + (1 - e) * 1.1, bz + (1 - e) * 3.2);
    camera.lookAt(1.3 + mx * 0.15, 1.0, 0);

    pmat.color.lerp(pTarget, 1 - Math.exp(-3 * dt));
    for (let i = 0; i < PN; i++) {
      let y = ppos[i * 3 + 1] + pspd[i] * dt * slow;
      if (y > 6) y = 0;
      ppos[i * 3 + 1] = y;
    }
    pgeo.attributes.position.needsUpdate = true;

    beam.position.x = X0 + ((now * 0.9 * slow) % 1) * 5.4;
    const off = Math.floor(now * 50 * slow);
    for (let k = 0; k < 9; k++) {
      const pos = ribbons[k].geometry.attributes.position;
      const y0 = 0.35 + k * 0.24;
      for (let i = 0; i < N; i++) {
        const v = seq[(i + off) % N][k] - means[k];
        pos.setXYZ(i, X0 + (i / (N - 1)) * 5.4, y0 + v * scale[k], -0.4 - (k % 3) * 0.12);
      }
      pos.needsUpdate = true;
    }
  });
}
