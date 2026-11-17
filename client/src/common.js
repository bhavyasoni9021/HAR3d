import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export const ACT = [
  { id: 'WALKING', short: 'Walk', name: 'Walking', color: '#ffb84d' },
  { id: 'WALKING_UPSTAIRS', short: 'Up', name: 'Walking upstairs', color: '#ff6b8a' },
  { id: 'WALKING_DOWNSTAIRS', short: 'Down', name: 'Walking downstairs', color: '#b38cff' },
  { id: 'SITTING', short: 'Sit', name: 'Sitting', color: '#4dd6c1' },
  { id: 'STANDING', short: 'Stand', name: 'Standing', color: '#6fa8ff' },
  { id: 'LAYING', short: 'Lay', name: 'Laying', color: '#c6f26b' },
];
export const actIndex = (id) => ACT.findIndex((a) => a.id === id);
export const hexNum = (c) => parseInt(c.slice(1), 16);

export async function api(path, opts) {
  const r = await fetch('/api' + path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `Request failed (${r.status})`);
  return j;
}

/** Shared render-loop scaffold: resizes with the canvas and pauses while off-screen. */
export function makeStage(canvas, { fov = 45, pos = [0, 1.4, 5], target = [0, 1, 0], orbit = false } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 300);
  camera.position.set(...pos);
  const tgt = new THREE.Vector3(...target);
  camera.lookAt(tgt);
  let controls = null;
  if (orbit) {
    controls = new OrbitControls(camera, canvas);
    controls.target.copy(tgt);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.update();
  }
  const frames = [];
  const showCbs = [];
  let visible = true;
  let running = false;
  let last = performance.now();
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  function loop() {
    if (!visible) { running = false; return; }
    running = true;
    requestAnimationFrame(loop);
    const now = performance.now();
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    for (const f of frames) f(dt, now / 1000);
    if (controls) controls.update();
    renderer.render(scene, camera);
  }
  new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (visible && !running) { last = performance.now(); loop(); }
    if (visible) showCbs.forEach((f) => f());
  }).observe(canvas);
  loop();

  return { scene, camera, renderer, controls, target: tgt, onFrame: (f) => frames.push(f), onShow: (f) => showCbs.push(f), reduce };
}

export function makeLabel(text, { size = 0.5, color = '#eceaff', sub = '' } = {}) {
  const c = document.createElement('canvas');
  c.width = 512; c.height = 128;
  const g = c.getContext('2d');
  g.textAlign = 'center';
  g.fillStyle = color;
  g.font = '700 46px "Bricolage Grotesque", "Segoe UI", sans-serif';
  g.fillText(text, 256, sub ? 56 : 80);
  if (sub) { g.font = '500 30px "Instrument Sans", "Segoe UI", sans-serif'; g.fillStyle = '#a9acd6'; g.fillText(sub, 256, 100); }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  s.scale.set(size * 4, size, 1);
  return s;
}

/** Make a 128x9 window loop seamlessly: cross-fade the last 16 samples into the first 16. */
export function periodic(rows) {
  const K = 16;
  const N = rows.length - K;
  const out = [];
  for (let i = 0; i < N; i++) {
    if (i < K) { const a = i / K; out.push(rows[i].map((v, c) => v * a + rows[N + i][c] * (1 - a))); } else out.push(rows[i]);
  }
  return out;
}

/** Offline stand-in used only if the API is unreachable. */
export function fallbackWindow(actId) {
  const dyn = actId.startsWith('WALKING');
  const f = actId === 'WALKING_UPSTAIRS' ? 1.8 : actId === 'WALKING_DOWNSTAIRS' ? 2.1 : 2.3;
  const rows = [];
  for (let t = 0; t < 128; t++) {
    const p = 2 * Math.PI * f * t / 50;
    const b = dyn ? 0.35 * Math.sin(p) + 0.1 * Math.sin(2 * p) : 0.004 * Math.sin(t * 7);
    const g = dyn ? 0.7 * Math.sin(p / 2 + 1) : 0.005 * Math.sin(t * 5);
    rows.push([b, b * 0.6, b * 0.5, g, g * 0.8, g * 0.5, 0.97 + b, -0.2 + b * 0.6, 0.05 + b * 0.5]);
  }
  return rows;
}

let softTex = null;
/** Soft radial sprite used for glows, beads and particles. */
export function softTexture() {
  if (softTex) return softTex;
  const c = document.createElement('canvas');
  c.width = 64; c.height = 64;
  const g = c.getContext('2d');
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)');
  gr.addColorStop(0.35, 'rgba(255,255,255,0.55)');
  gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr;
  g.fillRect(0, 0, 64, 64);
  softTex = new THREE.CanvasTexture(c);
  return softTex;
}

export const lerp = (a, b, k) => a + (b - a) * k;
export { THREE };
