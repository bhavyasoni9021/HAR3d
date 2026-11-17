import { THREE, makeStage, makeLabel, ACT, hexNum } from './common.js';

export function initConfusion(m) {
  const canvas = document.getElementById('cmCanvas');
  const tip = document.getElementById('cmTip');
  const stage = makeStage(canvas, { fov: 40, pos: [5.8, 6, 7.4], target: [0, 0.7, 0], orbit: true });
  const { scene, camera, controls } = stage;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 5;
  controls.maxDistance = 20;
  scene.add(new THREE.AmbientLight(0xffffff, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 2);
  sun.position.set(4, 8, 5);
  scene.add(sun);

  const S = 1.1;
  const off = -2.5 * S;
  const conf = m.confusion;
  const rowSum = conf.map((r) => r.reduce((a, b) => a + b, 0));
  const maxC = Math.max(...conf.flat());
  let sqrtMode = true;
  const cells = [];
  const labels = [];

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(7.6, 7.6), new THREE.MeshBasicMaterial({ color: 0x1a2050 }));
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);
  const grid = new THREE.GridHelper(6.6, 6, 0x4a54a8, 0x4a54a8);
  grid.position.y = 0.01;
  scene.add(grid);

  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      const diag = r === c;
      const col = diag ? hexNum(ACT[r].color) : 0xff4d6d;
      const mat = new THREE.MeshStandardMaterial({ color: col, roughness: 0.45, emissive: col, emissiveIntensity: 0.12 });
      const bar = new THREE.Mesh(new THREE.BoxGeometry(0.85, 1, 0.85), mat);
      bar.position.set(off + c * S, 0.01, off + r * S);
      bar.userData = { r, c };
      scene.add(bar);
      let lab = null;
      if (conf[r][c] > 0) {
        lab = makeLabel(String(conf[r][c]), { size: 0.3, color: diag ? '#ffffff' : '#ffc2cf' });
        scene.add(lab);
      }
      cells.push({ bar, r, c, lab, h: 0.02 });
    }
  }
  ACT.forEach((a, i) => {
    const x = makeLabel(a.short, { size: 0.34, color: a.color });
    x.position.set(off + i * S, 0.05, off + 5 * S + 0.95);
    const y = makeLabel(a.short, { size: 0.34, color: a.color });
    y.position.set(off - 0.95, 0.05, off + i * S);
    scene.add(x, y);
  });
  const px = makeLabel('Predicted', { size: 0.4, color: '#a9acd6' });
  px.position.set(0, 0.05, 4.15);
  const py = makeLabel('True', { size: 0.4, color: '#a9acd6' });
  py.position.set(-4.0, 0.05, 0);
  scene.add(px, py);

  let since = 0;
  stage.onShow(() => { since = 0; cells.forEach((c) => { c.h = 0.02; }); });
  const heightOf = (v) => 3.1 * (sqrtMode ? Math.sqrt(v / maxC) : v / maxC);
  const goalOf = (cell) => (since > (cell.r * 6 + cell.c) * 0.04 ? Math.max(0.03, heightOf(conf[cell.r][cell.c])) : 0.02);

  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let hover = null;
  canvas.addEventListener('pointermove', (ev) => {
    const rect = canvas.getBoundingClientRect();
    ndc.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObjects(cells.map((c) => c.bar), false)[0];
    hover = hit ? hit.object.userData : null;
    if (!hover) { tip.hidden = true; return; }
    const { r, c } = hover;
    const n = conf[r][c];
    tip.hidden = false;
    tip.style.left = `${Math.min(ev.clientX - rect.left + 14, rect.width - 270)}px`;
    tip.style.top = `${ev.clientY - rect.top + 14}px`;
    tip.innerHTML = r === c
      ? `<b>${ACT[r].name}</b><br>${n} of ${rowSum[r]} windows correct (${(100 * n / rowSum[r]).toFixed(1)}%)`
      : `<b>True ${ACT[r].name} → predicted ${ACT[c].name}</b><br>${n} windows (${(100 * n / rowSum[r]).toFixed(1)}% of ${ACT[r].short})`;
  });
  canvas.addEventListener('pointerleave', () => { hover = null; tip.hidden = true; });

  document.getElementById('cmScale').addEventListener('click', (e) => {
    sqrtMode = !sqrtMode;
    e.currentTarget.textContent = `Height: ${sqrtMode ? 'square root' : 'linear'}`;
  });

  stage.onFrame((dt) => {
    since += dt;
    const k = 1 - Math.exp(-5 * dt);
    for (const cell of cells) {
      cell.h += (goalOf(cell) - cell.h) * k;
      cell.bar.scale.y = cell.h;
      cell.bar.position.y = cell.h / 2;
      const on = hover && hover.r === cell.r && hover.c === cell.c;
      cell.bar.material.emissiveIntensity += ((on ? 0.8 : 0.12) - cell.bar.material.emissiveIntensity) * 0.3;
      if (cell.lab) cell.lab.position.set(cell.bar.position.x, cell.h + 0.3, cell.bar.position.z);
    }
  });
}
