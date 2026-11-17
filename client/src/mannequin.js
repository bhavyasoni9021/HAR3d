import * as THREE from 'three';
import { hexNum, softTexture } from './common.js';

const s = Math.sin;
const c = Math.cos;
const POSE_KEYS = ['y', 'rx', 'z', 'spine', 'tL', 'tR', 'kL', 'kR', 'aL', 'aR', 'eL', 'eR'];

/** Target joint angles for each activity at time t (positive x-rotation swings a hanging limb backwards). */
function target(act, t) {
  switch (act) {
    case 'WALKING': {
      const p = t * 2 * Math.PI * 1.1;
      return { y: 0.88 + 0.025 * Math.abs(c(p)), rx: 0, z: 0, spine: 0.05, tL: -0.55 * s(p), tR: 0.55 * s(p), kL: 0.1 + 0.55 * Math.max(0, c(p)), kR: 0.1 + 0.55 * Math.max(0, -c(p)), aL: 0.5 * s(p), aR: -0.5 * s(p), eL: -0.35, eR: -0.35 };
    }
    case 'WALKING_UPSTAIRS': {
      const p = t * 2 * Math.PI * 0.85;
      return { y: 0.86 + 0.045 * Math.abs(c(p)), rx: 0, z: 0, spine: 0.24, tL: -0.25 - 0.7 * s(p), tR: -0.25 + 0.7 * s(p), kL: 0.3 + 0.95 * Math.max(0, c(p)), kR: 0.3 + 0.95 * Math.max(0, -c(p)), aL: 0.35 * s(p), aR: -0.35 * s(p), eL: -0.5, eR: -0.5 };
    }
    case 'WALKING_DOWNSTAIRS': {
      const p = t * 2 * Math.PI * 1.15;
      return { y: 0.86 + 0.055 * Math.abs(s(p)), rx: 0, z: 0, spine: -0.06, tL: -0.5 * s(p), tR: 0.5 * s(p), kL: 0.12 + 0.3 * Math.max(0, c(p)), kR: 0.12 + 0.3 * Math.max(0, -c(p)), aL: 0.35 * s(p) + 0.2, aR: -0.35 * s(p) + 0.2, eL: -0.2, eR: -0.2 };
    }
    case 'SITTING':
      return { y: 0.52, rx: 0, z: 0, spine: 0.08 + 0.01 * s(t * 1.3), tL: -1.5, tR: -1.5, kL: 1.5, kR: 1.5, aL: -0.55, aR: -0.55, eL: -1.0, eR: -1.0 };
    case 'STANDING':
      return { y: 0.88, rx: 0, z: 0, spine: 0.02 + 0.012 * s(t * 1.4), tL: 0, tR: 0, kL: 0.02, kR: 0.02, aL: 0.06, aR: 0.06, eL: -0.15, eR: -0.15 };
    default: // LAYING
      return { y: 0.19, rx: -Math.PI / 2, z: 0, spine: 0.01 * s(t * 1.2), tL: 0.03, tR: 0.03, kL: 0.02, kR: 0.02, aL: 0.12, aR: 0.12, eL: -0.1, eR: -0.1 };
  }
}

export class Mannequin {
  constructor(color = '#ffb84d') {
    this.group = new THREE.Group();
    this.root = new THREE.Group();
    this.group.add(this.root);
    const body = new THREE.MeshStandardMaterial({ color: 0xdfe3ff, roughness: 0.55, metalness: 0.05 });
    this.accent = new THREE.MeshStandardMaterial({ color: hexNum(color), emissive: hexNum(color), emissiveIntensity: 0.9, roughness: 0.4 });
    const cap = (r, l, m) => new THREE.Mesh(new THREE.CapsuleGeometry(r, l, 6, 12), m);

    const pelvis = cap(0.13, 0.1, body);
    pelvis.rotation.z = Math.PI / 2;
    this.root.add(pelvis);
    this.spine = new THREE.Group();
    this.root.add(this.spine);
    const torso = cap(0.15, 0.3, body);
    torso.position.y = 0.33;
    this.spine.add(torso);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.12, 24, 16), body);
    head.position.y = 0.79;
    this.spine.add(head);

    const limb = (parent, pos, upLen, loLen, r, foot) => {
      const hip = new THREE.Group();
      hip.position.set(...pos);
      parent.add(hip);
      const up = cap(r, upLen, body);
      up.position.y = -(upLen / 2 + r);
      hip.add(up);
      const knee = new THREE.Group();
      knee.position.y = -(upLen + 2 * r);
      hip.add(knee);
      const lo = cap(r * 0.9, loLen, body);
      lo.position.y = -(loLen / 2 + r * 0.9);
      knee.add(lo);
      if (foot) {
        const f = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.05, 0.2), body);
        f.position.set(0, -(loLen + 1.8 * r) + 0.01, 0.05);
        knee.add(f);
      }
      return { hip, knee };
    };
    this.legL = limb(this.root, [0.1, -0.02, 0], 0.3, 0.3, 0.065, true);
    this.legR = limb(this.root, [-0.1, -0.02, 0], 0.3, 0.3, 0.065, true);
    this.armL = limb(this.spine, [0.23, 0.6, 0], 0.2, 0.2, 0.045, false);
    this.armR = limb(this.spine, [-0.23, 0.6, 0], 0.2, 0.2, 0.045, false);

    // Phone clipped to the belt on the left hip, flat against the body (as in the UCI HAR setup).
    const belt = new THREE.Mesh(new THREE.TorusGeometry(0.158, 0.014, 8, 36), new THREE.MeshStandardMaterial({ color: 0x2a3170, roughness: 0.7 }));
    belt.rotation.x = Math.PI / 2;
    belt.position.y = 0.13;
    this.root.add(belt);
    this.phone = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.16, 0.09), this.accent);
    this.phone.position.set(0.172, 0.13, 0.02);
    this.root.add(this.phone);
    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.115, 40), new THREE.MeshBasicMaterial({ color: hexNum(color), transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
    this.ring.rotation.y = Math.PI / 2;
    this.ring.position.set(0.19, 0.13, 0.02);
    this.root.add(this.ring);

    // props
    const prop = new THREE.MeshStandardMaterial({ color: 0x2a3170, roughness: 0.8 });
    this.chair = new THREE.Group();
    const seat = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.05, 0.55), prop);
    seat.position.set(0, 0.4, 0.1);
    const back = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.55, 0.05), prop);
    back.position.set(0, 0.7, -0.17);
    this.chair.add(seat, back);
    for (const [x, z] of [[-0.24, -0.15], [0.24, -0.15], [-0.24, 0.35], [0.24, 0.35]]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.4, 0.04), prop);
      leg.position.set(x, 0.2, z);
      this.chair.add(leg);
    }
    this.mat = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.06, 2.1), prop);
    this.mat.position.set(0, 0.03, 0);
    this.group.add(this.chair, this.mat);

    // floor glow and a shockwave ring that fires whenever the activity changes
    this.glow = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 3.4), new THREE.MeshBasicMaterial({ map: softTexture(), color: hexNum(color), transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.glow.rotation.x = -Math.PI / 2;
    this.glow.position.y = 0.012;
    this.group.add(this.glow);
    this.shock = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.57, 72), new THREE.MeshBasicMaterial({ color: hexNum(color), transparent: true, opacity: 0, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.shock.rotation.x = -Math.PI / 2;
    this.shock.position.y = 0.02;
    this.shock.visible = false;
    this.shockT = 9;
    this.group.add(this.shock);

    this.cur = {};
    POSE_KEYS.forEach((k) => { this.cur[k] = 0; });
    Object.assign(this.cur, target('STANDING', 0));
  }

  setColor(hex) {
    const n = hexNum(hex);
    this.accent.color.setHex(n);
    this.accent.emissive.setHex(n);
    this.ring.material.color.setHex(n);
    this.glow.material.color.setHex(n);
    this.shock.material.color.setHex(n);
  }

  burst() { this.shockT = 0; this.shock.visible = true; }

  update(act, t, dt) {
    const tg = target(act, t);
    const k = 1 - Math.exp(-9 * dt);
    for (const key of POSE_KEYS) this.cur[key] += (tg[key] - this.cur[key]) * k;
    const p = this.cur;
    this.root.position.set(0, p.y, p.z);
    this.root.rotation.x = p.rx;
    this.spine.rotation.x = p.spine;
    this.legL.hip.rotation.x = p.tL; this.legR.hip.rotation.x = p.tR;
    this.legL.knee.rotation.x = p.kL; this.legR.knee.rotation.x = p.kR;
    this.armL.hip.rotation.x = p.aL; this.armR.hip.rotation.x = p.aR;
    this.armL.knee.rotation.x = p.eL; this.armR.knee.rotation.x = p.eR;
    this.chair.visible = act === 'SITTING';
    this.mat.visible = act === 'LAYING';
    this.glow.material.opacity = 0.5 + 0.12 * Math.sin(t * 2.2);
    if (this.shockT < 1.4) {
      this.shockT += dt;
      const u = Math.min(1, this.shockT / 1.4);
      this.shock.scale.setScalar(1 + u * 6);
      this.shock.material.opacity = 0.9 * (1 - u) * (1 - u);
      if (u >= 1) this.shock.visible = false;
    }
    const pulse = (t * 1.6) % 1;
    this.ring.scale.setScalar(1 + pulse * 1.8);
    this.ring.material.opacity = 0.7 * (1 - pulse);
  }
}

/** Grid floor that scrolls under walking activities. */
export class Floor {
  constructor(color = 0x3b4390) {
    this.group = new THREE.Group();
    this.grid = new THREE.GridHelper(24, 48, color, color);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.45;
    this.group.add(this.grid);
    this.shift = 0;
  }

  update(act, dt) {
    const sp = act === 'WALKING' ? 0.95 : act === 'WALKING_UPSTAIRS' ? 0.65 : act === 'WALKING_DOWNSTAIRS' ? 1.15 : 0;
    this.shift = (this.shift + sp * dt) % 0.5;
    this.grid.position.z = -this.shift;
  }
}
