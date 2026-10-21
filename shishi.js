// Shishi-odoshi on the pond's left bank: a kakehi pipe trickles into a bamboo arm until it tips, pours into the
// pond, swings back and knocks its stone. One cycle every PERIOD seconds, driven by timeU like everything else.
//
//   const shishi = createShishi({ THREE, scene, camera, timeU, groundAt, material });
//   render():  shishi.update();          // before camera.updateMatrixWorld(), so the listener follows the camera
//   pond.pour = shishi.pour;             // ripple rings where the spill lands
//   shishi.unlock() on the first click or key (browsers keep audio off until then); shishi.muted = true / false
//
// The sounds are synthesised here (no recordings): the knock is a few decaying bamboo and air-column modes plus a
// stone click, run through a small comb reverb; the pour and the trickle are filtered noise with bubble chirps.

const PERIOD = 30, PHASE = 18;              // first tip about ten seconds after the page opens
const REST = 22, TIP = -26;                 // arm angle in degrees: mouth up on the stone, mouth down pouring
const T_TIP = 27.0, T_POUR = 27.45, T_BACK = 28.3, T_KNOCK = 28.62;
const PIVOT = { x: -2.2, y: 7.75 }, DIR = { x: 0.88, y: 0.47 };   // Blender x/y; the arm points into the pond
const WY = -0.55 - 0.07;                    // pond surface height
const PIVOT_H = 0.40, ARM_R = 0.05, TAIL = 0.36, MOUTH = 0.66, SPOUT_X = 0.57, SPOUT_H = 0.84;

const deg = Math.PI / 180;
const ease = (s) => s * s;
function armAngle(tc) {
  if (tc < 26.4) return REST;
  if (tc < T_TIP) return REST - 2 * ((tc - 26.4) / 0.6) ** 2;                       // the balance goes
  if (tc < T_POUR) return REST - 2 + (TIP - REST + 2) * ease((tc - T_TIP) / (T_POUR - T_TIP));
  if (tc < T_BACK) { const x = tc - T_POUR; return TIP + 2.5 * Math.exp(-5 * x) * Math.sin(14 * x); }
  if (tc < T_KNOCK) return TIP + (REST - TIP) * ease((tc - T_BACK) / (T_KNOCK - T_BACK));
  if (tc < 29.2) { const x = tc - T_KNOCK; return REST - 5 * Math.exp(-9 * x) * Math.abs(Math.sin(x * Math.PI / 0.16)); }
  return REST;
}
const cycle = (t) => ((t + PHASE) % PERIOD + PERIOD) % PERIOD;

const STREAM_VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const STREAM_FRAG = /* glsl */`
uniform float uTime, uLen, uFrom, uTo, uSpeed;
varying vec2 vUv;
float h11(float n) { return fract(sin(n * 91.3) * 43758.5453); }
void main() {
  float v = vUv.y;
  if (v < uFrom || v > uTo) discard;
  // streaks running down the stream, a little broken toward its end
  float s = vUv.y * uLen * 9.0 + uTime * uSpeed;
  float k = floor(s), f = fract(s);
  float streak = mix(h11(k + floor(vUv.x * 5.0) * 17.0), h11(k + 1.0 + floor(vUv.x * 5.0) * 17.0), f * f * (3.0 - 2.0 * f));
  float side = abs(vUv.x - 0.5) * 2.0;
  vec3 c = mix(vec3(0.30, 0.36, 0.36), vec3(0.85, 0.90, 0.88), streak * (0.6 + 0.4 * side));
  gl_FragColor = vec4(c, 4.0);   // pond mask: the painter hands water through untouched
}`;

export function createShishi({ THREE, scene, camera, timeU, groundAt, material }) {
  const g0 = groundAt(PIVOT.x, PIVOT.y);
  const base = new THREE.Group();
  base.position.set(PIVOT.x, g0, -PIVOT.y);
  base.rotation.y = Math.atan2(DIR.y, DIR.x);   // local +X along the arm, local -Z away from the cushion
  scene.add(base);

  const bamboo = material({ c1: [0.30, 0.31, 0.10], mask: 1 });            // fresh green-gold cane
  const bambooOld = material({ c1: [0.26, 0.20, 0.09], mask: 1 });         // weathered posts and pipe
  const inside = material({ c1: [0.06, 0.045, 0.02], mask: 1 });
  const stone = material({ c1: [0.11, 0.105, 0.1], c2: [0.27, 0.26, 0.24], ramp: [0.35, 0.65], scale: 6.5, mask: 1 });
  const mesh = (geo, mat, parent = base) => {
    const m = new THREE.Mesh(geo, mat);
    m.castShadow = m.receiveShadow = true;
    parent.add(m);
    return m;
  };

  // the arm: a cane closed by a node behind the mouth, its mouth cut on the slant so it opens upward
  const arm = new THREE.Group();
  arm.position.set(0, PIVOT_H, 0);
  base.add(arm);
  const cane = new THREE.CylinderGeometry(ARM_R, ARM_R, 1, 20, 1, true).rotateZ(-Math.PI / 2);
  const pos = cane.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    pos.setX(i, pos.getX(i) > 0 ? 0.60 - 0.08 * (y / ARM_R) : -TAIL);
  }
  cane.computeVertexNormals();
  const caneMat = material({ c1: [0.30, 0.31, 0.10], mask: 1 });
  caneMat.side = THREE.DoubleSide;
  mesh(cane, caneMat, arm);
  mesh(new THREE.CircleGeometry(ARM_R * 0.98, 20).rotateY(-Math.PI / 2).translate(-TAIL, 0, 0), bamboo, arm);
  mesh(new THREE.CircleGeometry(ARM_R * 0.97, 20).rotateY(-Math.PI / 2).translate(0.30, 0, 0), inside, arm);
  for (const x of [-0.12, 0.30]) mesh(new THREE.CylinderGeometry(ARM_R * 1.07, ARM_R * 1.07, 0.014, 20).rotateZ(Math.PI / 2).translate(x, 0, 0), bamboo, arm);

  // trestle: two posts and the axle
  for (const z of [-0.085, 0.085]) mesh(new THREE.CylinderGeometry(0.028, 0.032, PIVOT_H + 0.05, 10).translate(0, (PIVOT_H + 0.05) / 2, z), bambooOld);
  mesh(new THREE.CylinderGeometry(0.011, 0.011, 0.22, 8).rotateX(Math.PI / 2).translate(0, PIVOT_H, 0), bambooOld);

  // the stone the tail knocks on
  const tailX = -TAIL * Math.cos(REST * deg) + 0.02, tailY = PIVOT_H - TAIL * Math.sin(REST * deg) - ARM_R / Math.cos(REST * deg);
  const rock = mesh(new THREE.IcosahedronGeometry(0.15, 2), stone);
  rock.scale.set(1.15, (tailY - 0.005) / 0.15, 0.95);
  rock.position.set(tailX, 0, 0);
  rock.rotation.y = 0.6;
  mesh(new THREE.IcosahedronGeometry(0.1, 1), stone).position.set(-0.08, 0.0, 0.22);

  // kakehi: a post behind the arm and a pipe reaching out over the mouth
  const postAt = new THREE.Vector3(0.12, 0, -0.34), postH = SPOUT_H + 0.1;
  mesh(new THREE.CylinderGeometry(0.036, 0.04, postH, 12).translate(postAt.x, postH / 2, postAt.z), bambooOld);
  const pA = new THREE.Vector3(postAt.x, SPOUT_H + 0.04, postAt.z), pB = new THREE.Vector3(SPOUT_X, SPOUT_H, 0);
  const pipe = mesh(new THREE.CylinderGeometry(0.021, 0.021, 1, 10, 1, true), bambooOld);
  pipe.position.copy(pA).lerp(pB, 0.5);
  pipe.scale.y = pA.distanceTo(pB) + 0.04;
  pipe.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pB.clone().sub(pA).normalize());

  // water: the trickle from the pipe, and the spill when the arm tips
  const streamMat = (speed) => new THREE.ShaderMaterial({
    vertexShader: STREAM_VERT, fragmentShader: STREAM_FRAG,
    uniforms: { uTime: timeU, uLen: { value: 1 }, uFrom: { value: 0 }, uTo: { value: 1 }, uSpeed: { value: speed } },
  });
  const trickle = new THREE.Mesh(new THREE.CylinderGeometry(0.0045, 0.006, 1, 8, 1, true).translate(0, -0.5, 0), streamMat(14));
  trickle.position.set(SPOUT_X, SPOUT_H - 0.012, 0);
  base.add(trickle);

  const pourAngle = TIP * deg;
  const lip = new THREE.Vector3(0.62, -0.035, 0).applyAxisAngle(new THREE.Vector3(0, 0, 1), pourAngle).add(new THREE.Vector3(0, PIVOT_H, 0));
  const waterY = WY - g0;
  const v0 = new THREE.Vector3(Math.cos(pourAngle), Math.sin(pourAngle), 0).multiplyScalar(0.7);
  const fall = (Math.sqrt(v0.y * v0.y + 2 * 9.8 * Math.max(0.02, lip.y - waterY)) + v0.y) / 9.8;
  const pts = [];
  for (let i = 0; i <= 12; i++) {
    const t = fall * i / 12;
    pts.push(new THREE.Vector3(lip.x + v0.x * t, lip.y + v0.y * t - 4.9 * t * t, 0));
  }
  const spill = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 16, 0.016, 8), streamMat(-10));
  spill.material.uniforms.uLen.value = 0.4;
  base.add(spill);
  // swap the tube's uv so v runs along it, as the stream shader expects
  { const uv = spill.geometry.attributes.uv; for (let i = 0; i < uv.count; i++) { const a = uv.getX(i); uv.setXY(i, uv.getY(i), a); } }
  const landing = base.localToWorld(pts[12].clone());
  const pour = { x: landing.x, y: -landing.z, last: (t) => { const c = T_POUR + fall; return Math.floor((t + PHASE - c) / PERIOD) * PERIOD + c - PHASE; } };

  // ---------------------------------------------------------------- sound
  const listener = new THREE.AudioListener();
  camera.add(listener);
  const ctx = listener.context;
  const sr = ctx.sampleRate;
  const rand = (() => { let s = 7; return () => ((s = (s * 16807) % 2147483647) / 2147483647) * 2 - 1; })();
  function buffer(seconds, fill) {
    const b = ctx.createBuffer(1, Math.floor(sr * seconds), sr);
    const d = b.getChannelData(0);
    fill(d);
    let peak = 0;
    for (let i = 0; i < d.length; i++) peak = Math.max(peak, Math.abs(d[i]));
    for (let i = 0; i < d.length; i++) d[i] *= 0.9 / (peak || 1);
    return b;
  }
  function reverb(d, mix) {   // a few feedback combs: the knock rings off the wall and the engawa
    const wet = new Float32Array(d.length);
    for (const [ms, fb] of [[29.7, 0.62], [37.1, 0.6], [41.1, 0.58], [43.7, 0.56], [53.3, 0.5]]) {
      const n = Math.floor(ms * sr / 1000);
      const c = new Float32Array(d.length);
      let lp = 0;
      for (let i = 0; i < d.length; i++) {
        const back = i >= n ? c[i - n] : 0;
        lp += (back - lp) * 0.45;
        c[i] = d[i] + lp * fb;
        wet[i] += c[i] - d[i];
      }
    }
    for (let i = 0; i < d.length; i++) d[i] += wet[i] * mix;
  }
  const knockBuf = buffer(1.4, (d) => {
    const hit = (t0, amp, bright) => {
      // air column of the closed cane, then the cane's own wall modes, then the click of bamboo on stone
      const modes = [[232, 0.07, 0.75], [780, 0.05, 0.85], [1655, 0.03, 0.45 * bright], [2890, 0.018, 0.3 * bright], [4310, 0.009, 0.15 * bright]];
      const i0 = Math.floor(t0 * sr);
      for (let i = i0; i < Math.min(d.length, i0 + sr * 0.5); i++) {
        const t = i / sr - t0;
        let v = 0;
        for (const [f, tau, a] of modes) v += a * Math.exp(-t / tau) * Math.sin(2 * Math.PI * f * t * (1 + 0.015 * Math.exp(-t / 0.01)));
        v += rand() * 0.9 * bright * Math.exp(-t / 0.0022);
        d[i] += v * amp * Math.min(1, t / 0.0004);
      }
    };
    hit(0, 1, 1); hit(0.155, 0.22, 0.6); hit(0.245, 0.06, 0.4);
    reverb(d, 0.22);
  });
  function water(d, env, bubbles, lo, hi) {
    let a = 0, b = 0;
    for (let i = 0; i < d.length; i++) {
      const x = rand();
      a += (x - a) * hi; b += (a - b) * lo;           // band of noise between two one-pole filters
      d[i] = (a - b) * env(i / sr);
    }
    for (let k = 0; k < bubbles.n; k++) {                // bubbles: short rising chirps
      const t0 = bubbles.at(k), f0 = 500 + 900 * Math.abs(rand()), i0 = Math.floor(t0 * sr);
      let ph = 0;
      for (let i = i0; i < Math.min(d.length, i0 + sr * 0.03); i++) {
        const t = i / sr - t0;
        ph += 2 * Math.PI * f0 * (1 + 12 * t) / sr;
        d[i] += Math.sin(ph) * Math.exp(-t / 0.008) * bubbles.amp * env(i / sr);
      }
    }
  }
  const splashBuf = buffer(1.3, (d) => water(d, (t) => Math.min(1, t / 0.03) * Math.exp(-Math.max(0, t - 0.5) / 0.15),
    { n: 26, at: (k) => 0.02 + 0.7 * (k / 26) + 0.02 * rand(), amp: 0.5 }, 0.05, 0.5));
  const trickleBuf = buffer(4, (d) => water(d, () => 0.35, { n: 30, at: (k) => (k + 0.5) * 4 / 30 + 0.04 * rand(), amp: 0.35 }, 0.08, 0.6));

  const voice = (buf, gain, ref, at) => {
    const s = new THREE.PositionalAudio(listener);
    s.setBuffer(buf); s.setRefDistance(ref); s.setRolloffFactor(1.2); s.setVolume(gain);
    s.position.copy(at);
    base.add(s);
    return s;
  };
  const knock = voice(knockBuf, 1.0, 2.5, new THREE.Vector3(tailX, 0.15, 0));
  const splash = voice(splashBuf, 0.45, 1.5, pts[12]);
  const trickleS = voice(trickleBuf, 0.12, 1.0, new THREE.Vector3(SPOUT_X, 0.6, 0));
  trickleS.setLoop(true);

  let unlocked = false, muted = false, prevT = timeU.value;
  function unlock() {
    if (unlocked) return;
    unlocked = true;
    ctx.resume().then(() => { if (!muted && !trickleS.isPlaying) trickleS.play(); });
  }
  const play = (s) => { if (!unlocked || muted || ctx.state !== 'running') return; if (s.isPlaying) s.stop(); s.play(); };

  // ---------------------------------------------------------------- per frame
  function update() {
    const t = timeU.value, tc = cycle(t);
    const a = armAngle(tc) * deg;
    arm.rotation.z = a;
    // the trickle falls onto the arm (or, while it's down, past it into the pond)
    const top = PIVOT_H + SPOUT_X * Math.tan(a) - 0.01;
    const end = a > 0 ? top : Math.max(waterY, top);
    trickle.scale.y = SPOUT_H - 0.012 - end;
    trickle.material.uniforms.uLen.value = trickle.scale.y;
    // the spill: runs from the lip once the arm is down, then thins out from the top as the arm empties
    const u = spill.material.uniforms;
    const on = tc > T_POUR - 0.08 && tc < T_BACK + 0.15;
    spill.visible = on;
    if (on) {
      u.uTo.value = Math.min(1, (tc - T_POUR + 0.08) / 0.18);
      u.uFrom.value = Math.max(0, (tc - T_BACK + 0.1) / 0.25);
    }
    // sounds fire when time crosses their moment (a scripted jump in time doesn't count)
    const dt = t - prevT;
    if (dt > 0 && dt < 0.5) {
      const crossed = (at) => { const c0 = cycle(prevT); return c0 < at && (c0 + dt >= at || c0 > cycle(t)); };
      if (crossed(T_KNOCK)) play(knock);
      if (crossed(T_POUR + fall * 0.5)) play(splash);
    }
    prevT = t;
  }

  return {
    update, unlock, pour,
    get muted() { return muted; },
    set muted(m) {
      muted = m;
      if (m && trickleS.isPlaying) trickleS.pause();
      if (!m && unlocked) ctx.resume().then(() => { if (!trickleS.isPlaying) trickleS.play(); });
    },
  };
}
