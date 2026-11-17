/**
 * Physics-inspired generator of 2.56 s inertial-sensor windows (128 samples @ 50 Hz, 9 channels).
 *
 * These are NOT recordings from the UCI HAR dataset. They are synthetic windows that mimic the
 * structure of a waist-mounted smartphone: gait harmonics for the walking classes, a gravity vector
 * whose orientation separates the static postures, and sensor noise. They exist so the live demo
 * works out of the box. (If the real dataset is placed in ./data the API serves genuine test windows too.)
 *
 * Channel order: body_acc xyz (g), body_gyro xyz (rad/s), total_acc xyz (g).
 */
import { T, C, ACTIVITIES } from './engine.js';

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const rad = (d) => (d * Math.PI) / 180;

/** Nominal gait / posture parameters per class; each window jitters them to mimic different people. */
const PROFILES = {
  WALKING:            { freq: 2.30, amp: 0.40, gyro: 0.62, h2: 0.30, h3: 0.30, pitch: 9,  roll: 0.5, ry: 0.75, rz: 0.65 },
  WALKING_UPSTAIRS:   { freq: 1.85, amp: 0.38, gyro: 0.74, h2: 0.29, h3: 0.23, pitch: 17, roll: 0.3, ry: 0.67, rz: 0.64 },
  WALKING_DOWNSTAIRS: { freq: 2.05, amp: 0.31, gyro: 1.05, h2: 0.33, h3: 0.27, pitch: 11, roll: 0.8, ry: 0.59, rz: 0.67 },
  SITTING:            { pitch: 15, roll: 1.5 },
  STANDING:           { pitch: 11, roll: 0.3 },
  LAYING:             {},
};

/**
 * @param {string} activity one of ACTIVITIES
 * @param {{seed?: number, noise?: number, tilt?: number}} opts
 *   noise: sensor-noise multiplier (1 = nominal), tilt: extra phone pitch in degrees
 * @returns {{window: number[][], meta: object}} window[t][c]
 */
export function simulateWindow(activity, { seed = 1, noise = 1, tilt = 0 } = {}) {
  if (!ACTIVITIES.includes(activity)) throw new Error(`Unknown activity "${activity}"`);
  const rng = mulberry32(seed * 9973 + ACTIVITIES.indexOf(activity) * 131 + 7);
  const out = new Float32Array(T * C);
  const dt = 1 / 50;
  const P = PROFILES[activity];
  const dynamic = P.freq !== undefined;

  // "Subject" traits derived from the seed: gait speed, vigour and how the phone sits on the waist.
  const jit = (v, rel) => v * (1 + rel * gauss(rng));
  const phase = rng() * Math.PI * 2;
  let pitch;
  let roll;
  if (activity === 'LAYING') { pitch = 90 + 12 * gauss(rng); roll = rng() * Math.PI * 2; }
  else { pitch = P.pitch + 2.5 * gauss(rng); roll = P.roll + (activity === 'SITTING' ? 0.08 : 0.25) * gauss(rng); }
  pitch += tilt;

  const th = rad(pitch);
  const gx = Math.cos(th);
  const gy = -Math.sin(th) * Math.cos(roll);
  const gz = Math.sin(th) * Math.sin(roll);

  const freq = dynamic ? jit(P.freq, 0.05) : 0;
  const amp = dynamic ? jit(P.amp, 0.08) : 0;
  const gyro = dynamic ? jit(P.gyro, 0.08) : 0;
  const n = noise;
  const settle = activity === 'LAYING' && rng() < 0.5 ? 0.25 + 0.2 * rng() : 0; // posture-settling transient

  for (let t = 0; t < T; t++) {
    const time = t * dt;
    const ba = [0, 0, 0];
    const bg = [0, 0, 0];

    if (dynamic) {
      const ph = 2 * Math.PI * freq * time + phase;
      const wave = (p) => Math.sin(ph + p) + P.h2 * Math.sin(2 * ph + p * 1.7 + 0.6) + P.h3 * Math.sin(3 * ph + p * 2.3 + 1.1);
      ba[0] = amp * wave(0);
      ba[1] = amp * P.ry * wave(1.3);
      ba[2] = amp * P.rz * wave(2.5);
      bg[0] = gyro * 0.9 * Math.sin(ph / 2 + 0.4);
      bg[1] = gyro * Math.sin(ph + 1.1) + 0.3 * gyro * Math.sin(2 * ph);
      bg[2] = gyro * 0.6 * Math.sin(ph / 2 + 2.0);
      for (let k = 0; k < 3; k++) {
        ba[k] += 0.02 * n * gauss(rng);
        bg[k] += 0.03 * n * gauss(rng);
      }
    } else {
      for (let k = 0; k < 3; k++) {
        ba[k] = 0.004 * n * gauss(rng);
        bg[k] = 0.006 * n * gauss(rng);
      }
      if (settle) {
        const d = Math.exp(-time / settle);
        ba[0] += 0.3 * d * Math.sin(14 * time + 1);
        ba[1] += -0.2 * d * Math.sin(11 * time);
        ba[2] += 0.35 * d * Math.cos(9 * time);
      }
    }

    const o = t * C;
    out[o] = ba[0]; out[o + 1] = ba[1]; out[o + 2] = ba[2];
    out[o + 3] = bg[0]; out[o + 4] = bg[1]; out[o + 5] = bg[2];
    out[o + 6] = gx + ba[0]; out[o + 7] = gy + ba[1]; out[o + 8] = gz + ba[2];
  }

  const window = [];
  for (let t = 0; t < T; t++) window.push(Array.from(out.subarray(t * C, (t + 1) * C), (v) => Math.round(v * 1e5) / 1e5));
  return { window, meta: { activity, seed, noise, tilt, pitchDeg: Math.round(pitch * 10) / 10, simulated: true } };
}

const clamp = (v, lo, hi, d) => { const x = Number(v); return Number.isFinite(x) ? Math.min(hi, Math.max(lo, x)) : d; };
const vec3 = (v, lo, hi, d) => [0, 1, 2].map((k) => clamp(Array.isArray(v) ? v[k] : undefined, lo, hi, d[k]));

/**
 * Build a window from a handful of numbers (port of synth_window in the original Streamlit app):
 * gravity gives the posture, a sine at `freq` gives body movement and rotation, total acc = body acc + gravity.
 */
export function synthWindow({ g, ba, gy, freq, noise, seed } = {}) {
  const G = vec3(g, -1.5, 1.5, [0.97, -0.2, 0.06]);
  const BA = vec3(ba, 0, 1, [0, 0, 0]);
  const GY = vec3(gy, 0, 2, [0, 0, 0]);
  const f = clamp(freq, 0.5, 5, 2);
  const nz = clamp(noise, 0, 0.05, 0.005);
  const rng = mulberry32(Math.floor(clamp(seed, 0, 1e9, 0)) + 1);
  const phase = [0, 1.3, 2.6];
  const window = [];
  for (let t = 0; t < T; t++) {
    const time = t / 50;
    const body = [0, 1, 2].map((k) => BA[k] * Math.SQRT2 * Math.sin(2 * Math.PI * f * time + phase[k]) + nz * gauss(rng));
    const gyro = [0, 1, 2].map((k) => GY[k] * Math.SQRT2 * Math.sin(2 * Math.PI * f * time + phase[k] + 0.7) + nz * gauss(rng));
    const total = body.map((b, k) => b + G[k]);
    window.push([...body, ...gyro, ...total].map((v) => Math.round(v * 1e5) / 1e5));
  }
  return { window, meta: { manual: true, simulated: true } };
}
