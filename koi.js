// Koi behaviour for the garden pond: a small fixed-step simulation, kept free of three.js so it can be tested on its own.
//
//   const sim = createKoiSim({ clearance, floorDepth, bounds, fish });
//   sim.advance(t)          // steps (or replays from the start) until sim time reaches t; same t, same pond
//   sim.pose(i, out)        // interpolated pose of koi i for drawing: spine points, tail phase, fins
//   sim.feed(x, y)          // scatter a handful of food on the water at Blender (x, y)
//   sim.visitor = [x, y]    // where the viewer stands (Blender x, y), or null; curious koi come over
//
// Everything is in Blender's ground plane (x, y metres) with depth measured down from the water surface.
// Each koi steers by heading and speed only, never by teleporting: it turns through arcs whose radius
// grows with speed, its turn rate eases in and out, and it swims in bursts of tail beats followed by
// long glides. The body is drawn along the trail the head left behind, so every turn bends the whole
// fish the way it actually swam. All randomness comes from a seeded generator and the clock is a fixed
// step from the start, so garden.view({ t }) replays to the same picture.

export const DT = 1 / 30;
const PREROLL = 25;              // seconds simulated before t = 0, so the pond opens mid-swim
const TRAIL_DS = 0.012;          // metres between remembered head positions
const TRAIL_N = 96;
export const SPINE_N = 12;

const TAU = Math.PI * 2;
const wrap = (a) => a - TAU * Math.floor((a + Math.PI) / TAU);
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
function rng32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const MODES = { WANDER: 0, FOLLOW: 1, KISS: 2, REST: 3, DART: 4, FEED: 5, VISIT: 6, CHASE: 7 };
export { MODES };

// fish: [{ len (m), social 0..1, bold 0..1, pace (cruise m/s) }]
// clearance(x, y): metres of open water around a point (negative outside the pond)
// floorDepth(x, y): depth of the pond bed below the surface
export function createKoiSim({ clearance, floorDepth, bounds, fish: specs, seed = 7 }) {
  // ---------------------------------------------------------------- open-water field, sampled once
  const [x0, y0, x1, y1] = bounds, CELL = 0.04;
  const gw = Math.ceil((x1 - x0) / CELL) + 1, gh = Math.ceil((y1 - y0) / CELL) + 1;
  const field = new Float32Array(gw * gh), bed = new Float32Array(gw * gh);
  for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
    const x = x0 + i * CELL, y = y0 + j * CELL;
    field[j * gw + i] = clearance(x, y);
    bed[j * gw + i] = floorDepth(x, y);
  }
  function sample(arr, x, y) {
    const fx = clamp((x - x0) / CELL, 0, gw - 1.001), fy = clamp((y - y0) / CELL, 0, gh - 1.001);
    const i = Math.floor(fx), j = Math.floor(fy), u = fx - i, v = fy - j, k = j * gw + i;
    return (arr[k] * (1 - u) + arr[k + 1] * u) * (1 - v) + (arr[k + gw] * (1 - u) + arr[k + gw + 1] * u) * v;
  }
  const clear = (x, y) => sample(field, x, y);
  const bedAt = (x, y) => Math.max(0.04, sample(bed, x, y));
  const grad = (x, y, out) => {
    const e = 0.06;
    out[0] = (clear(x + e, y) - clear(x - e, y)) / (2 * e);
    out[1] = (clear(x, y + e) - clear(x, y - e)) / (2 * e);
    const l = Math.hypot(out[0], out[1]) || 1;
    out[0] /= l; out[1] /= l;
    return out;
  };
  // a random point of open water, at least c metres from rim and trunk
  function openPoint(rand, c) {
    for (let k = 0; k < 400; k++) {
      const x = x0 + rand() * (x1 - x0), y = y0 + rand() * (y1 - y0);
      if (clear(x, y) > c) return [x, y];
    }
    return [(x0 + x1) / 2, (y0 + y1) / 2];
  }

  // the open-water point (at least c from rim and trunk) nearest to (x, y)
  function nearestOpen(x, y, c) {
    let best = null, bd = 1e9;
    for (let j = 0; j < gh; j += 2) for (let i = 0; i < gw; i += 2) {
      if (field[j * gw + i] < c) continue;
      const px = x0 + i * CELL, py = y0 + j * CELL, d = (px - x) ** 2 + (py - y) ** 2;
      if (d < bd) { bd = d; best = [px, py]; }
    }
    return best || [(x0 + x1) / 2, (y0 + y1) / 2];
  }

  // ---------------------------------------------------------------- state
  let S;            // the whole simulation state; rebuilt on rewind
  const feeds = []; // { t, x, y } in sim time, kept so a rewind replays them
  let visitor = null;

  function reset() {
    const rand = rng32(seed);
    const fish = specs.map((sp, i) => {
      const [x, y] = openPoint(rand, 0.8);
      const h = rand() * TAU;
      const f = {
        i, sp, x, y, d: 0.12 + rand() * 0.1, h, w: 0, v: sp.pace * 0.8, vd: 0,
        hw: h, wr: 0, beating: false, lo: 0.6, beats: 0, hz: 1, acc: 0, beatPh: 0, amp: 0.2, phase: rand() * TAU, flutter: rand() * TAU,
        roll: 0, pitch: 0, scull: 0.5, side: 0, sideHold: 0, depthGoal: 0.15, sx: 0, sy: 0, dz: 0,
        mode: MODES.WANDER, modeT: 5 + rand() * 20, leader: -1, slot: [0, 0], target: -1, gulp: 0, rings: 0, ringT: 0,
        trail: new Float32Array(TRAIL_N * 3), tn: 0, th: 0, travelled: 0, prev: null,
      };
      // the trail starts as a straight line behind the head
      for (let k = 0; k < TRAIL_N; k++) {
        const s = (TRAIL_N - 1 - k) * TRAIL_DS;
        f.trail.set([x - Math.cos(h) * s, y - Math.sin(h) * s, f.d], k * 3);
      }
      f.tn = TRAIL_N; f.th = TRAIL_N - 1;
      return f;
    });
    S = { t: -PREROLL, rand, fish, pellets: [], rings: [], feedIdx: 0, chaseT: 60 + rand() * 60 };
    for (const f of fish) snapshot(f);
  }

  // ---------------------------------------------------------------- per-step helpers
  function trailPush(f) {
    f.th = (f.th + 1) % TRAIL_N;
    f.trail[f.th * 3] = f.x; f.trail[f.th * 3 + 1] = f.y; f.trail[f.th * 3 + 2] = f.d;
  }
  // point on f's trail s metres behind its head
  function trailAt(f, s, out) {
    let px = f.x, py = f.y, pd = f.d, acc = 0;
    for (let k = 0; k < TRAIL_N - 1; k++) {
      const idx = ((f.th - k) % TRAIL_N + TRAIL_N) % TRAIL_N * 3;
      const qx = f.trail[idx], qy = f.trail[idx + 1], qd = f.trail[idx + 2];
      const l = Math.hypot(qx - px, qy - py);
      if (acc + l >= s && l > 1e-6) {
        const u = (s - acc) / l;
        out[0] = px + (qx - px) * u; out[1] = py + (qy - py) * u; out[2] = pd + (qd - pd) * u;
        return out;
      }
      acc += l; px = qx; py = qy; pd = qd;
    }
    // ran out of trail: carry on straight
    const idx = ((f.th - (TRAIL_N - 1)) % TRAIL_N + TRAIL_N) % TRAIL_N * 3, last = (f.th - (TRAIL_N - 2) + TRAIL_N) % TRAIL_N * 3;
    let dx = f.trail[idx] - f.trail[last], dy = f.trail[idx + 1] - f.trail[last + 1];
    const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
    out[0] = px + dx * (s - acc); out[1] = py + dy * (s - acc); out[2] = pd;
    return out;
  }
  function inChain(j, i) {   // would following j lead back to i?
    for (let n = 0; j >= 0 && n < 20; n++) { if (j === i) return true; j = S.fish[j].leader; }
    return false;
  }
  function followers(j) { let n = 0; for (const g of S.fish) if (g.leader === j && g.mode === MODES.FOLLOW) n++; return n; }
  function root(j) { for (let n = 0; n < 20 && S.fish[j].mode === MODES.FOLLOW && S.fish[j].leader >= 0; n++) j = S.fish[j].leader; return j; }
  function groupSize(j) { const r = root(j); let n = 0; for (const g of S.fish) if (root(g.i) === r) n++; return n; }
  // someone worth falling in with: not resting or busy at the surface, and not a group that is already big
  const joinable = (f) => (g) => !inChain(g.i, f.i) && g.mode !== MODES.REST && g.mode !== MODES.KISS &&
    g.mode !== MODES.VISIT && followers(g.i) < 2 && groupSize(g.i) < 4;
  function ring(x, y, s) { S.rings.push({ x, y, t: S.t, s }); if (S.rings.length > 40) S.rings.shift(); }

  function setMode(f, mode, dur) {
    if (f.mode === MODES.FOLLOW && mode !== MODES.FOLLOW) {
      // peeling off: drift away from the group on a gentle new bearing
      f.hw = f.h + (S.rand() < 0.5 ? -1 : 1) * (0.5 + S.rand() * 0.6);
    }
    f.mode = mode; f.modeT = dur;
    if (mode !== MODES.FOLLOW && mode !== MODES.CHASE) f.leader = -1;
    if (mode === MODES.WANDER) f.hw = f.h;
    if (mode === MODES.DART) {
      // bolt for open water: the bearing nearest the intended one with a clear run ahead
      const base = f.hw;
      let best = base, score = -1e9;
      for (let k = -8; k <= 8; k++) {
        const a = base + k * 0.25;
        let room = 9;
        for (const s of [0.4, 0.8, 1.3]) room = Math.min(room, clear(f.x + Math.cos(a) * s, f.y + Math.sin(a) * s));
        const sc = Math.min(room, 0.6) * 4 - Math.abs(k) * 0.1;
        if (sc > score) { score = sc; best = a; }
      }
      f.hw = best;
    }
  }

  function nearestFish(f, range, pick) {
    let best = -1, bd = range;
    for (const g of S.fish) {
      if (g === f) continue;
      const d = Math.hypot(g.x - f.x, g.y - f.y);
      if (d < bd && pick(g, d)) { bd = d; best = g.i; }
    }
    return best;
  }

  function startFollow(f, j) {
    const n = followers(j);
    const side = n % 2 ? 1 : -1;
    // slots fan out behind the leader: close pairs, then a loose V
    f.slot = [0.32 + 0.22 * Math.floor(n / 2) + S.rand() * 0.12, side * (0.14 + 0.1 * Math.floor(n / 2) + S.rand() * 0.06)];
    f.leader = j;
    setMode(f, MODES.FOLLOW, (12 + S.rand() * 40) * (0.4 + f.sp.social));
  }

  // what to do when the current activity runs out
  function chooseMode(f) {
    const r = S.rand(), sp = f.sp;
    if (r < 0.10) return setMode(f, MODES.KISS, 6 + S.rand() * 8);
    if (r < 0.18) return setMode(f, MODES.REST, 10 + S.rand() * 20);
    if (r < 0.22 + sp.bold * 0.04) return setMode(f, MODES.DART, 0.6 + S.rand() * 0.6);
    if (r < 0.22 + sp.social * 0.3) {
      const j = nearestFish(f, 2.2, joinable(f));
      if (j >= 0) return startFollow(f, j);
    }
    setMode(f, MODES.WANDER, 8 + S.rand() * 25 * (1.4 - sp.social));
  }

  // ---------------------------------------------------------------- one step for one koi
  const g2 = [0, 0], tp = [0, 0, 0];
  function stepFish(f, dt) {
    const sp = f.sp, L = sp.len, rand = S.rand;
    let gx, gy;   // where it wants to go
    let vd = sp.pace, depthGoal = f.depthGoal, wGoal = 1, turnR = 0.45 + L * 0.8;   // comfortable turn radius
    const cosH = Math.cos(f.h), sinH = Math.sin(f.h);

    f.modeT -= dt;
    if (f.modeT <= 0) chooseMode(f);

    // wandering koi cruise toward somewhere across the pond, on a bearing that sways in slow arcs
    // (close enough, or passed it by: never orbit a spot)
    const ddx = f.dest ? f.dest[0] - f.x : 0, ddy = f.dest ? f.dest[1] - f.y : 0, dd = Math.hypot(ddx, ddy);
    if (!f.dest || dd < 1.0 || (dd < 2.0 && ddx * cosH + ddy * sinH < 0) || rand() < dt / 40) {
      for (let k = 0; k < 12; k++) {
        f.dest = openPoint(rand, 0.48);
        if (Math.hypot(f.dest[0] - f.x, f.dest[1] - f.y) > 2.2) break;
      }
    }
    f.wr += (-f.wr / 3.0) * dt + 0.1 * Math.sqrt(dt) * (rand() + rand() + rand() - 1.5) * 2;
    const toDest = Math.atan2(f.dest[1] - f.y, f.dest[0] - f.x);
    f.hw += (f.wr + clamp(wrap(toDest - f.hw), -1, 1) * 0.12) * dt;
    // the wander bearing never strays far from where the fish is actually facing
    const lag = wrap(f.hw - f.h);
    if (Math.abs(lag) > 1.2) f.hw = f.h + Math.sign(lag) * 1.2;
    gx = Math.cos(f.hw); gy = Math.sin(f.hw);

    switch (f.mode) {
      case MODES.WANDER:
        if (f.depthGoal === undefined || rand() < dt / 12) f.depthGoal = 0.1 + rand() * 0.2;
        depthGoal = f.depthGoal;
        // a chance meeting: a sociable koi falls in behind one passing close by
        if (rand() < dt * 0.08 * sp.social) {
          const j = nearestFish(f, 0.9, (g) => joinable(f)(g) && (g.x - f.x) * cosH + (g.y - f.y) * sinH > 0);
          if (j >= 0) startFollow(f, j);
        }
        break;
      case MODES.FOLLOW: {
        const g = S.fish[f.leader];
        if (!g || g.mode === MODES.REST || g.mode === MODES.KISS && rand() < dt * 0.2) { chooseMode(f); break; }
        trailAt(g, f.slot[0] + g.sp.len * 0.5, tp);
        // the slot sits off to one side of the leader's trail
        const ahead = trailAt(g, f.slot[0] + g.sp.len * 0.5 - 0.05, [0, 0, 0]);
        let tx = ahead[0] - tp[0], ty = ahead[1] - tp[1]; const tl = Math.hypot(tx, ty) || 1; tx /= tl; ty /= tl;
        const sx = tp[0] - ty * f.slot[1], sy = tp[1] + tx * f.slot[1];
        const dx = sx - f.x, dy = sy - f.y, dist = Math.hypot(dx, dy);
        const along = dx * cosH + dy * sinH;
        // steer for the slot while it is ahead; if it has overshot, ease off and keep the leader's line rather than loop back
        const lat = -dx * sinH + dy * cosH;
        const k = along > 0 ? smooth(0.05, 0.6, dist) : 0;
        gx = dx / (dist || 1) * k + (Math.cos(g.h) - sinH * lat * 1.5) * (1 - k);
        gy = dy / (dist || 1) * k + (Math.sin(g.h) + cosH * lat * 1.5) * (1 - k);
        vd = clamp(g.v + along * 0.5, 0.04, sp.pace * 1.6);
        depthGoal = g.d + 0.02 * Math.sign(f.slot[1]);
        if (dist > 2.5) chooseMode(f);   // lost them
        break;
      }
      case MODES.KISS:
        // rise and nose the surface a few times, then sink back
        vd = 0.035; depthGoal = 0.012; turnR = 0.35;
        if (f.d < 0.03) {
          f.ringT -= dt;
          if (f.ringT <= 0) {
            ring(f.x + cosH * 0.02, f.y + sinH * 0.02, 0.7 + rand() * 0.3); f.ringT = 0.9 + rand() * 1.4;
            f.gulp = 1;
          }
        }
        break;
      case MODES.REST:
        vd = 0.022; depthGoal = bedAt(f.x, f.y) - 0.08; wGoal = 0.6;
        break;
      case MODES.DART:
        vd = 0.42 + sp.bold * 0.15; turnR = 0.6;
        if (f.modeT < 0.05) { f.beating = false; setMode(f, MODES.WANDER, 6 + rand() * 10); }
        break;
      case MODES.CHASE: {
        const g = S.fish[f.target];
        const dx = g.x - f.x, dy = g.y - f.y, d = Math.hypot(dx, dy) || 1;
        gx = dx / d; gy = dy / d; vd = 0.32; depthGoal = g.d; turnR = 0.4;
        if (d < 0.35 && g.mode !== MODES.DART) {   // tag: the other one bolts
          g.hw = Math.atan2(-dy, -dx) + (rand() - 0.5); setMode(g, MODES.DART, 0.8 + rand() * 0.5);
          setMode(f, MODES.WANDER, 5 + rand() * 6);
        }
        break;
      }
      case MODES.FEED: {
        // swim for the nearest floating pellet nobody else is about to take
        let best = null, bd = 1e9;
        for (const p of S.pellets) {
          if (!p.alive || p.t0 > S.t) continue;
          const d = Math.hypot(p.x - f.x, p.y - f.y) + (p.claim >= 0 && p.claim !== f.i ? 0.5 : 0);
          if (d < bd) { bd = d; best = p; }
        }
        if (!best) { f.modeT = Math.min(f.modeT, 3); depthGoal = 0.05; vd = 0.06; break; }
        best.claim = f.i;
        const mx = f.x + cosH * L * 0.02, my = f.y + sinH * L * 0.02;
        const dx = best.x - mx, dy = best.y - my, d = Math.hypot(dx, dy) || 1;
        gx = dx / d; gy = dy / d; turnR = 0.3;
        vd = clamp(0.05 + d * 0.35, 0.05, 0.3 + sp.bold * 0.1);
        depthGoal = d < 0.7 ? 0.01 : 0.08;
        if (d < 0.06 + L * 0.06 && f.d < 0.06) {     // gulp
          best.alive = false; ring(best.x, best.y, 1.0); f.gulp = 1; f.modeT = Math.max(f.modeT, 4);
        }
        break;
      }
      case MODES.VISIT: {
        // nose up to where the viewer stands, mouth open, hoping
        if (!visitor) { chooseMode(f); break; }
        const [tx, ty] = f.visitAt;
        const dx = tx - f.x, dy = ty - f.y, d = Math.hypot(dx, dy) || 1;
        const k = smooth(0.1, 0.5, d);
        const fx = visitor[0] - f.x, fy = visitor[1] - f.y, fl = Math.hypot(fx, fy) || 1;
        gx = dx / d * k + fx / fl * (1 - k); gy = dy / d * k + fy / fl * (1 - k);
        vd = 0.03 + clamp(d * 0.3, 0, 0.15); depthGoal = d < 0.6 ? 0.02 : 0.1; turnR = 0.3;
        if (!f.arrived && d < 0.45) { f.arrived = true; f.modeT = 10 + rand() * 15; }   // the wait starts once there
        if (d < 0.4 && f.d < 0.04) {
          f.ringT -= dt;
          if (f.ringT <= 0) { ring(f.x + cosH * L * 0.02, f.y + sinH * L * 0.02, 0.6); f.ringT = 1.2 + rand() * 1.8; f.gulp = 1; }
        }
        break;
      }
    }

    // ---- personal space: slide past each other, or over and under when the depths differ
    let sx = 0, sy = 0, dz = 0;
    for (const g of S.fish) {
      if (g === f) continue;
      const lead = L * 0.3;
      const hx = f.x + cosH * lead, hy = f.y + sinH * lead;
      for (const s of [0, 0.45]) {
        const ox = g.x - Math.cos(g.h) * g.sp.len * s, oy = g.y - Math.sin(g.h) * g.sp.len * s;
        const dx = hx - ox, dy = hy - oy, vz = (f.d - g.d) * 2.5;
        const d = Math.hypot(dx, dy, vz);
        const r = (L + g.sp.len) * (f.leader === g.i || g.leader === f.i ? 0.32 : 0.45) + 0.06;
        if (d < r && d > 1e-4) {
          const k = (1 - d / r) ** 2;
          sx += dx / d * k; sy += dy / d * k;
          dz += (f.d >= g.d ? 1 : -1) * k * 0.08;          // the one below dips a little lower
        }
      }
    }
    // eased, so a crowd doesn't make it twitch
    const ke = Math.min(1, dt / 0.5);
    f.sx += (sx - f.sx) * ke; f.sy += (sy - f.sy) * ke; f.dz += (dz - f.dz) * ke;

    // ---- the bearing it would like, then the nearest bearing to it that has open water ahead
    let dx = gx * wGoal + f.sx * 2.2, dy = gy * wGoal + f.sy * 2.2;
    if (Math.hypot(dx, dy) < 1e-4) { dx = cosH; dy = sinH; }
    const want0 = Math.atan2(dy, dx);
    const reach = 0.3 + f.v * 3.0 + turnR * 0.55, need = 0.3 + L * 0.2;
    const room = (a) => {
      let m = 9;
      for (const s of [0.3, 0.65, 1.0]) m = Math.min(m, clear(f.x + Math.cos(a) * reach * s, f.y + Math.sin(a) * reach * s));
      return m;
    };
    let want = want0, urg = clamp((need - room(f.h)) / need, 0, 1);
    if (room(want0) < need) {
      // search outward from the wish; commit to one side so it never dithers left-right
      if (f.sideHold <= 0 || f.side === 0) {
        let l = 99, r = 99;
        for (let k = 1; k <= 20 && (l === 99 || r === 99); k++) {
          if (l === 99 && room(want0 + k * 0.16) >= need) l = k;
          if (r === 99 && room(want0 - k * 0.16) >= need) r = k;
        }
        f.side = l === r ? (room(want0 + 0.8) >= room(want0 - 0.8) ? 1 : -1) : l < r ? 1 : -1;
      }
      let best = want0, bestRoom = -9;
      for (let k = 1; k <= 20; k++) {
        const a = want0 + f.side * k * 0.16, rm = room(a);
        if (rm > bestRoom) { bestRoom = rm; best = a; }
        if (rm >= need) { best = a; break; }
      }
      want = best;
      f.sideHold = 1.2;
      urg = Math.max(urg, 0.3);
      // its wandering intention follows along, so it doesn't fight its way back to the wall
      if (f.mode === MODES.WANDER || f.mode === MODES.REST || f.mode === MODES.KISS) { f.hw = want; f.wr *= 0.5; }
    } else {
      f.sideHold -= dt;
    }
    // already too close (a tight turn carried it in): head for open water along the slope of the field
    const here = clear(f.x, f.y);
    if (here < need * 0.75) {
      const k = clamp((need * 0.75 - here) / (need * 0.3), 0, 1);
      grad(f.x, f.y, g2);
      want = Math.atan2(Math.sin(want) * (1 - k) + g2[1] * k * 1.5, Math.cos(want) * (1 - k) + g2[0] * k * 1.5);
      urg = Math.max(urg, k);
    }
    dz = f.dz;

    // ---- turn toward it with an easing turn rate
    const err = wrap(want - f.h);
    const R = Math.max(0.12 + L * 0.4, turnR * (1 - 0.6 * Math.min(1, urg)));   // tighter only when it must, never a coil
    const wMax = Math.min(1.8, 0.1 + f.v / R);   // a little pectoral pivot even when slow
    const wTarget = clamp(err * 1.4, -wMax, wMax);
    const alpha = 1.0 + 2.5 * Math.min(1, urg) + (f.mode === MODES.DART || f.mode === MODES.FEED ? 2 : 0);
    f.w += clamp(wTarget - f.w, -alpha * dt, alpha * dt);
    f.h = wrap(f.h + f.w * dt);

    // ---- speed: a few tail beats, then a long glide (and slower through tight turns)
    if (f.mode !== MODES.FOLLOW || f.leader < 0) vd *= 1 - 0.15 * Math.min(followers(f.i), 2);   // leaders keep it easy
    vd *= 1 - 0.35 * clamp(Math.abs(err) - 0.6, 0, 1);
    if (urg > 0) vd = Math.max(vd, 0.05 + 0.04 * Math.min(1, urg));   // keep way on to steer clear
    // Koi swim in bursts: two to four unhurried strokes push them up to speed, then the tail goes still and
    // they coast on their momentum until they have slowed well down. Hurrying koi (darting, chasing, food)
    // just stroke more, and quicker.
    const fast = f.mode === MODES.DART || f.mode === MODES.CHASE || f.mode === MODES.FEED;
    const tau = 2.6 + L * 2;                                   // how long a glide carries
    if (!f.beating && f.v < vd * (fast ? 0.9 : urg > 0 ? 0.75 : f.mode === MODES.FOLLOW ? Math.max(f.lo, 0.6) : f.lo)) {   // at the rim it strokes to keep steerage
      f.beating = true;
      f.beats = fast ? 4 + Math.floor(rand() * 3) : 1 + Math.floor(rand() * 3);
      f.hz = fast ? 2.0 + rand() * 0.6 : 0.9 + rand() * 0.3;
      const goal = Math.min(vd * (fast ? 1.15 : f.mode === MODES.FOLLOW || urg > 0 ? 1.1 : 1.3 + rand() * 0.2), fast ? 0.6 : sp.pace * 2);
      f.acc = Math.max(0, goal - f.v) * f.hz / f.beats * 1.1 + goal / tau;   // reach it within the strokes
      f.beatPh = 0;
    }
    let thrust = 0;
    if (f.beating) {
      const before = Math.floor(f.beatPh / TAU);
      f.beatPh += TAU * f.hz * dt;
      thrust = f.acc * 2 * Math.sin(f.beatPh / 2) ** 2;     // each stroke is a push, not a steady shove
      if (Math.floor(f.beatPh / TAU) > before && --f.beats <= 0) { f.beating = false; f.lo = 0.4 + rand() * 0.15; }
    }
    const drag = f.v / tau + Math.abs(f.w) * f.v * 0.25;
    f.v = Math.max(0.018, f.v + (thrust - drag) * dt);

    // tail: big slow strokes while beating; gliding, it straightens and barely stirs
    const ampGoal = f.beating ? (fast ? 0.85 : 0.6 + 0.25 * Math.min(1, f.acc / 0.08)) : 0.03 + Math.min(0.12, Math.abs(f.w) * 0.08);
    f.amp += (ampGoal - f.amp) * Math.min(1, dt / (f.beating ? 0.2 : 0.5));
    f.phase += TAU * (f.beating ? f.hz : 0.12) * dt;
    // pectorals: scull when slow, fold back when fast
    f.scull += (clamp(1 - f.v / 0.22, 0, 1) - f.scull) * Math.min(1, dt / 0.6);
    f.flutter += TAU * (0.6 + 0.8 * f.scull) * dt;
    f.roll += (clamp(-f.w * f.v * 3.0, -0.35, 0.35) - f.roll) * Math.min(1, dt / 0.4);   // lean into the turn
    f.gulp = Math.max(0, f.gulp - dt * 1.6);

    // ---- move
    f.x += Math.cos(f.h) * f.v * dt;
    f.y += Math.sin(f.h) * f.v * dt;
    const floorD = bedAt(f.x, f.y) - 0.04 - L * 0.07;
    const dGoal = clamp(depthGoal + dz, 0.0, Math.max(0.01, floorD));
    const vzMax = 0.006 + 0.3 * f.v;                            // rise and sink at a shallow angle, never straight up
    const vz = clamp((dGoal - f.d) * 0.8, -vzMax, vzMax);
    f.d = clamp(f.d + vz * dt, 0.0, Math.max(0.0, floorD + 0.02));
    // the tilt that goes with it (depth gained per metre swum), eased so the fish rocks gently into it
    f.pitch += (clamp(vz / Math.max(f.v, 0.04), -0.35, 0.35) - f.pitch) * Math.min(1, dt / 0.8);
    // the rim is a wall even if steering was late
    const c = clear(f.x, f.y);
    if (c < 0.12) { grad(f.x, f.y, g2); f.x += g2[0] * (0.12 - c); f.y += g2[1] * (0.12 - c); }

    f.travelled += f.v * dt;
    while (f.travelled >= TRAIL_DS) { f.travelled -= TRAIL_DS; trailPush(f); }
  }

  // bodies are solid: nudge heads out of each other's way after the move
  function resolveOverlaps() {
    const F = S.fish;
    for (let a = 0; a < F.length; a++) for (let b = a + 1; b < F.length; b++) {
      const f = F[a], g = F[b];
      const dx = f.x - g.x, dy = f.y - g.y, vz = (f.d - g.d) * 2.5, d = Math.hypot(dx, dy, vz);
      const r = (f.sp.len + g.sp.len) * 0.17;
      if (d < r && d > 1e-5) {
        const push = (r - d) * 0.5 / d;
        f.x += dx * push; f.y += dy * push; g.x -= dx * push; g.y -= dy * push;
      }
    }
  }

  function socialEvents(dt) {
    // now and then one koi playfully chases another
    S.chaseT -= dt;
    if (S.chaseT <= 0) {
      S.chaseT = 50 + S.rand() * 90;
      const F = S.fish.filter((f) => f.mode === MODES.WANDER || f.mode === MODES.FOLLOW);
      if (F.length > 1) {
        const f = F[Math.floor(S.rand() * F.length)];
        const j = nearestFish(f, 2.0, (g) => g.mode !== MODES.REST);
        if (j >= 0) { setMode(f, MODES.CHASE, 5); f.target = j; }
      }
    }
    // a viewer at the edge draws the curious ones
    if (visitor) {
      const vc = clear(visitor[0], visitor[1]);
      if (vc > -1.6) {
        for (const f of S.fish) {
          if (f.mode !== MODES.WANDER && f.mode !== MODES.FOLLOW) continue;
          if (S.rand() < dt * 0.025 * f.sp.bold) {
            // the open water nearest the viewer, give or take a little jostling room
            const [bx, by] = nearestOpen(visitor[0] + (S.rand() - 0.5) * 0.6, visitor[1] + (S.rand() - 0.5) * 0.6, 0.32 + f.sp.len * 0.2);
            f.visitAt = [bx, by]; f.arrived = false;
            setMode(f, MODES.VISIT, 40);
          }
        }
      }
    }
  }

  function stepPellets(dt) {
    while (S.feedIdx < feeds.length && feeds[S.feedIdx].t <= S.t) {
      const e = feeds[S.feedIdx++];
      const n = 10 + Math.floor(S.rand() * 5);
      for (let k = 0; k < n; k++) {
        const a = S.rand() * TAU, r = Math.sqrt(S.rand()) * 0.35;
        let x = e.x + Math.cos(a) * r, y = e.y + Math.sin(a) * r;
        for (let q = 0; q < 30 && clear(x, y) < 0.12; q++) { grad(x, y, g2); x += g2[0] * 0.04; y += g2[1] * 0.04; }
        const t0 = S.t + S.rand() * 0.35;
        S.pellets.push({ x, y, t0, alive: true, claim: -1, seed: S.rand() });
        if (k < 6) S.rings.push({ x, y, t: t0, s: 0.45 });
      }
      // word spreads through the pond: the nearest and boldest come first
      for (const f of S.fish) {
        const d = Math.hypot(f.x - e.x, f.y - e.y);
        f.feedIn = 0.3 + d * (1.0 - 0.5 * f.sp.bold) + S.rand() * 1.5;
      }
    }
    for (const f of S.fish) {
      if (f.feedIn === undefined || f.feedIn < 0) continue;
      f.feedIn -= dt;
      if (f.feedIn < 0) setMode(f, MODES.FEED, 20 + S.rand() * 10);
    }
    for (const p of S.pellets) {
      if (!p.alive) continue;
      p.x += 0.012 * dt; p.y -= 0.006 * dt;                 // floating off with the breeze
      if (clear(p.x, p.y) < 0.1) { p.x -= 0.012 * dt; p.y += 0.006 * dt; }
      p.claim = -1;
      if (S.t - p.t0 > 120) p.alive = false;                 // soaked and sunk
    }
    if (S.pellets.length > 80) S.pellets = S.pellets.filter((p) => p.alive);
  }

  function snapshot(f) {
    f.prev = { pitch: f.pitch, x: f.x, y: f.y, d: f.d, phase: f.phase, amp: f.amp, flutter: f.flutter, scull: f.scull, roll: f.roll, gulp: f.gulp };
  }

  function step() {
    for (const f of S.fish) snapshot(f);
    stepPellets(DT);
    socialEvents(DT);
    for (const f of S.fish) stepFish(f, DT);
    resolveOverlaps();
    S.t += DT;
  }

  function advance(t) {
    if (!S || t < S.t - DT - 1e-6) reset();
    // a long jump (a scripted view far ahead) still walks every step, so it lands where a live pond would
    while (S.t < t) step();
    S.alpha = 1 - (S.t - t) / DT;
  }

  // ---------------------------------------------------------------- what the renderer reads
  const tmp3 = [0, 0, 0];
  // out.spine: SPINE_N points (x, y, depth) from nose to tail tip over len * reach
  function pose(i, out, reach = 1) {
    const f = S.fish[i], p = f.prev, a = clamp(S.alpha ?? 1, 0, 1);
    const lerp = (u, v) => u + (v - u) * a;
    const hx = lerp(p.x, f.x), hy = lerp(p.y, f.y), hd = lerp(p.d, f.d);
    out.pitch = Math.min(lerp(p.pitch, f.pitch), (hd - 0.015) / (f.sp.len * reach));   // the tail stays under water
    // the head runs ahead of the trail by however far it has come since the last remembered point
    const L = f.sp.len * reach;
    const sp = out.spine;
    sp[0] = hx; sp[1] = hy; sp[2] = hd;
    // the interpolated head sits a little behind the simulated one, so measure the trail from there
    const off = Math.hypot(f.x - hx, f.y - hy);
    for (let k = 1; k < SPINE_N; k++) {
      trailAt(f, k / (SPINE_N - 1) * L + off, tmp3);
      sp[k * 3] = tmp3[0]; sp[k * 3 + 1] = tmp3[1];
      // koi don't bend up and down: the body stays straight in depth and the whole fish tilts, nose down to
      // sink and nose up to rise, so depth runs along the body as a straight line at the current pitch
      sp[k * 3 + 2] = hd - out.pitch * k / (SPINE_N - 1) * L;
    }
    out.phase = lerp(p.phase, f.phase); out.amp = lerp(p.amp, f.amp);
    out.flutter = lerp(p.flutter, f.flutter); out.scull = lerp(p.scull, f.scull);
    out.roll = lerp(p.roll, f.roll); out.gulp = lerp(p.gulp, f.gulp);
    out.mode = f.mode; out.leader = f.leader;
    return out;
  }

  reset();
  return {
    advance, pose,
    get t() { return S.t; },
    get fish() { return S.fish; },
    get pellets() { return S.pellets; },
    get rings() { return S.rings; },
    feed(x, y) { feeds.push({ t: S.t + DT, x, y }); },
    set visitor(v) { visitor = v; },
    get visitor() { return visitor; },
    clear, bedAt,
  };
}
