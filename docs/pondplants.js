// Plants at the pond's edge: a lotus stand, two rafts of water lily pads with a flower each, and a few clumps of
// iris and rush along the rim. Kept to the margins: PLANT_SPOTS is also what the koi steer around (pond.js).
//
//   createPondPlants({ THREE, scene, material, groundAt });   // material(opts, params) -> patched Lambert
//
// Positions are Blender x/y (three.js x, -z); WY is the pond surface. Everything is static and merged into a few
// meshes, coloured per vertex, so it costs a handful of draw calls.
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const GROUND = -0.55, WY = GROUND - 0.07;
const POND = [[0.0, 7.6], [1.8, 6.6], [3.8, 6.9], [5.0, 8.1], [4.6, 9.6], [2.6, 10.3], [0.8, 9.9],
  [-0.6, 10.6], [-2.6, 10.2], [-3.4, 9.0], [-2.0, 8.0]];

// x, y, radius (m), kind. The koi keep this far clear of each spot.
export const PLANT_SPOTS = [
  { x: 2.65, y: 9.4, r: 0.5, kind: 'lotus' },     // back of the pond, toward the red maple
  { x: 1.2, y: 7.45, r: 0.42, kind: 'lily' },     // front, near the stepping stones
  { x: -1.85, y: 9.75, r: 0.36, kind: 'lily' },   // back left
  { x: -1.3, y: 10.35, r: 0.5, kind: 'grass' },   // grass clumps settle wherever the rim is free of stones nearby
  { x: -3.05, y: 9.4, r: 0.45, kind: 'grass' },
  { x: 2.9, y: 6.85, r: 0.5, kind: 'grass' },
  { x: 4.85, y: 8.9, r: 0.45, kind: 'grass' },
  { x: 1.7, y: 10.1, r: 0.5, kind: 'grass' },
];

function inPoly(x, y) {
  let inside = false;
  for (let i = 0, j = POND.length - 1; i < POND.length; j = i++) {
    const [xi, yi] = POND[i], [xj, yj] = POND[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi) inside = !inside;
  }
  return inside;
}
function polyDist(x, y) {
  let best = 1e9;
  for (let i = 0; i < POND.length; i++) {
    const [ax, ay] = POND[i], [bx, by] = POND[(i + 1) % POND.length];
    const dx = bx - ax, dy = by - ay;
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)));
    best = Math.min(best, Math.hypot(x - ax - t * dx, y - ay - t * dy));
  }
  return best;
}

export function createPondPlants({ THREE, scene, material, groundAt }) {
  // rocks stand up out of the heightmap: open water sits at WY, the muddy bank a few cm above
  const clearOfRocks = (x, y, r, top) => [[0, 0], [r, 0], [-r, 0], [0, r], [0, -r]].every(([dx, dy]) => groundAt(x + dx, y + dy) < top);
  const openWater = (x, y, r) => inPoly(x, y) && clearOfRocks(x, y, r, WY + 0.02);
  let seed = 11;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const between = (a, b) => a + (b - a) * rnd();
  const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
  const parts = { leaf: [], petal: [], stem: [] };   // single-sided-ish solids, thin double-sided sheets, stalks

  // paint every vertex of a geometry with fn(position) -> rgb, then place it
  function add(list, geo, colour, m) {
    const p = geo.attributes.position, c = new Float32Array(p.count * 3), v = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i); c.set(typeof colour === 'function' ? colour(v) : colour, i * 3); }
    geo.setAttribute('color', new THREE.BufferAttribute(c, 3));
    if (geo.index) geo = geo.toNonIndexed();
    for (const k of Object.keys(geo.attributes)) if (!['position', 'normal', 'color'].includes(k)) geo.deleteAttribute(k);
    geo.applyMatrix4(m);
    list.push(geo);
  }
  const at = (x, y, h, yaw = 0, tiltX = 0, tiltZ = 0) => new THREE.Matrix4().compose(
    new THREE.Vector3(x, WY + h, -y), new THREE.Quaternion().setFromEuler(new THREE.Euler(tiltX, yaw, tiltZ, 'YXZ')), new THREE.Vector3(1, 1, 1));
  function scatter(spot, n, ok, gap = 0.12) {
    const out = [];
    for (let k = 0; k < 400 && out.length < n; k++) {
      const a = rnd() * Math.PI * 2, d = Math.sqrt(rnd()) * spot.r, x = spot.x + Math.cos(a) * d, y = spot.y + Math.sin(a) * d;
      if (ok(x, y) && out.every(([px, py]) => Math.hypot(px - x, py - y) > gap)) out.push([x, y]);
    }
    return out;
  }

  // ---------------------------------------------------------------- water lilies: notched pads, one flower per raft
  const PAD = [0.05, 0.10, 0.035], PAD_OLD = [0.17, 0.12, 0.03], PAD_EDGE = [0.10, 0.04, 0.03];
  for (const spot of PLANT_SPOTS.filter((s) => s.kind === 'lily')) {
    const pts = scatter(spot, 9, (x, y) => openWater(x, y, 0.1) && polyDist(x, y) > 0.06);
    pts.forEach(([x, y], i) => {
      const r = between(0.07, 0.13), notch = 0.28;
      const g = new THREE.CircleGeometry(r, 22, notch / 2, Math.PI * 2 - notch).rotateX(-Math.PI / 2);
      const pos = g.attributes.position;
      for (let k = 0; k < pos.count; k++) {   // the rim curls up a little
        const d = Math.hypot(pos.getX(k), pos.getZ(k)) / r;
        pos.setY(k, 0.012 * r / 0.1 * Math.pow(d, 4));
      }
      g.computeVertexNormals();
      const age = rnd() < 0.3 ? between(0.4, 0.9) : between(0, 0.25);   // autumn: a few pads yellowing
      add(parts.leaf, g, (v) => mix(mix(PAD, PAD_OLD, age), PAD_EDGE, Math.pow(Math.hypot(v.x, v.z) / r, 6) * 0.6),
        at(x, y, 0.004 + 0.001 * i, rnd() * Math.PI * 2));
    });
    // the flower: two rings of pointed petals cupped upward round a yellow heart
    const [fx, fy] = pts[0] || [spot.x, spot.y];
    const pink = spot.y > 9, tip = pink ? [0.62, 0.22, 0.30] : [0.80, 0.78, 0.70], base = pink ? [0.75, 0.48, 0.52] : [0.86, 0.84, 0.78];
    for (const [ring, n, len, lift] of [[0, 10, 0.055, 0.55], [1, 8, 0.045, 0.95]]) {
      for (let k = 0; k < n; k++) {
        const g = new THREE.CircleGeometry(1, 10).scale(0.016, len, 1).translate(0, len, 0);
        const a = (k + ring * 0.5) / n * Math.PI * 2;
        add(parts.petal, g, (v) => mix(base, tip, Math.min(1, v.y / (2 * len))),
          at(fx, fy, 0.02, a).multiply(new THREE.Matrix4().makeRotationX(-Math.PI / 2 + lift)));
      }
    }
    add(parts.leaf, new THREE.SphereGeometry(0.014, 10, 6).scale(1, 0.6, 1), [0.75, 0.55, 0.05], at(fx, fy, 0.03));
  }

  // ---------------------------------------------------------------- lotus: raised bowl leaves on tall stalks
  const LOTUS = [0.06, 0.13, 0.04], LOTUS_OLD = [0.22, 0.17, 0.04], LOTUS_DRY = [0.16, 0.10, 0.04];
  for (const spot of PLANT_SPOTS.filter((s) => s.kind === 'lotus')) {
    const pts = scatter(spot, 10, (x, y) => openWater(x, y, 0.24), 0.2);   // whole leaf over water, clear of the stones
    pts.forEach(([x, y], i) => {
      const r = between(0.15, 0.22), h = i < 2 ? between(0.02, 0.05) : between(0.18, 0.6);   // a couple float, most stand up
      const g = new THREE.CircleGeometry(r, 28, 0, Math.PI * 2).rotateX(-Math.PI / 2);
      const pos = g.attributes.position;
      const wob = rnd() * 6;
      for (let k = 0; k < pos.count; k++) {   // shallow funnel with a wavy rim
        const px = pos.getX(k), pz = pos.getZ(k), d = Math.hypot(px, pz) / r, th = Math.atan2(pz, px);
        pos.setY(k, r * (0.22 * Math.pow(d, 1.6) + 0.06 * Math.sin(5 * th + wob) * Math.pow(d, 3)));
      }
      g.computeVertexNormals();
      const age = i === 3 ? 0.9 : rnd() < 0.35 ? between(0.3, 0.7) : between(0, 0.2);
      const leafCol = age > 0.8 ? LOTUS_DRY : mix(LOTUS, LOTUS_OLD, age);
      add(parts.leaf, g, (v) => mix(leafCol, [leafCol[0] * 1.3, leafCol[1] * 1.25, leafCol[2]], Math.hypot(v.x, v.z) / r),
        at(x, y, h, rnd() * Math.PI * 2, between(-0.35, 0.35), between(-0.35, 0.35)));
      if (h > 0.1) add(parts.stem, new THREE.CylinderGeometry(0.005, 0.006, h + 0.25, 5).translate(0, -(h + 0.25) / 2, 0),
        [0.10, 0.11, 0.04], at(x, y, h));
    });
    // one late flower, still in bud, and one seed head gone brown
    const extra = scatter(spot, 2, (x, y) => openWater(x, y, 0.08) && pts.every(([px, py]) => Math.hypot(px - x, py - y) > 0.12), 0.15);
    const [[bx, by], [sx, sy]] = extra.length === 2 ? extra : [[spot.x, spot.y], [spot.x + 0.1, spot.y - 0.1]];
    const bh = 0.62;
    add(parts.stem, new THREE.CylinderGeometry(0.005, 0.006, bh + 0.25, 5).translate(0, -(bh + 0.25) / 2, 0), [0.10, 0.11, 0.04], at(bx, by, bh));
    const bud = new THREE.LatheGeometry([0, 0.6, 0.95, 1, 0.8, 0.45, 0].map((w, k) => new THREE.Vector2(w * 0.035, k * 0.016)), 12);
    add(parts.petal, bud, (v) => mix([0.55, 0.32, 0.30], [0.70, 0.20, 0.30], v.y / 0.096), at(bx, by, bh));
    const sh = 0.48;
    add(parts.stem, new THREE.CylinderGeometry(0.005, 0.006, sh + 0.25, 5).translate(0, -(sh + 0.25) / 2, 0), [0.12, 0.10, 0.04], at(sx, sy, sh));
    add(parts.leaf, new THREE.CylinderGeometry(0.045, 0.022, 0.05, 14).translate(0, 0.025, 0), [0.16, 0.11, 0.05], at(sx, sy, sh, 0, 0.3));
  }

  // ---------------------------------------------------------------- iris and rush clumps along the rim
  function blade(len, w, bend, cols) {
    const N = 6, pos = [], col = [], idx = [];
    for (let k = 0; k <= N; k++) {
      const s = k / N, x = bend * len * s * s, y = len * s * (1 - 0.12 * s * s), hw = w * Math.pow(1 - s, 0.8) * 0.5 + 0.0005;
      pos.push(x, y, -hw, x, y, hw);
      const c = mix(cols[0], cols[1], Math.pow(s, 1.5));
      col.push(...c, ...c);
      if (k) idx.push(2 * k - 2, 2 * k - 1, 2 * k, 2 * k - 1, 2 * k + 1, 2 * k);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return { g, c: col };
  }
  for (const spot of PLANT_SPOTS.filter((s) => s.kind === 'grass')) {
    // only in the shallows or on the muddy bank between the rim stones, never on a rock
    const pts = scatter(spot, 3, (x, y) => polyDist(x, y) < 0.2 && clearOfRocks(x, y, 0.14, GROUND + 0.05), 0.18);
    for (const [x, y] of pts) {
      const wet = inPoly(x, y), base = wet ? -0.02 : groundAt(x, y) - WY + 0.004;
      if (!wet) {   // a patch of dark mud round the clump's foot
        const r = between(0.16, 0.24), g = new THREE.CircleGeometry(r, 16).rotateX(-Math.PI / 2), pos = g.attributes.position;
        for (let k = 1; k < pos.count; k++) { const f = 0.8 + 0.25 * Math.sin(k * 2.3 + x * 7); pos.setX(k, pos.getX(k) * f); pos.setZ(k, pos.getZ(k) * f); }
        add(parts.leaf, g, (v) => mix([0.05, 0.035, 0.02], [0.09, 0.07, 0.035], Math.hypot(v.x, v.z) / r), at(x, y, base, rnd() * 6.28));
      }
      const iris = rnd() < 0.5;
      const n = iris ? 9 : 22;
      for (let k = 0; k < n; k++) {
        const len = iris ? between(0.35, 0.6) : between(0.45, 0.85);
        const tipCol = rnd() < 0.3 ? [0.28, 0.20, 0.06] : [0.10, 0.15, 0.03];
        const { g, c } = blade(len, iris ? 0.04 : 0.016, between(0.1, 0.35), [[0.04, 0.09, 0.02], tipCol]);
        let vi = 0;
        add(parts.petal, g, () => c.slice((vi++) * 3, vi * 3), at(x + between(-0.04, 0.04), y + between(-0.04, 0.04),
          base, rnd() * Math.PI * 2, 0, between(-0.12, 0.12)));
      }
    }
  }

  // ---------------------------------------------------------------- merge per material
  const mats = {
    leaf: material({ c1: [1, 1, 1], mask: 1 }, { vertexColors: true, side: THREE.DoubleSide }),
    petal: material({ c1: [1, 1, 1], mask: 1, k: 'petal' }, { vertexColors: true, side: THREE.DoubleSide }),
    stem: material({ c1: [1, 1, 1], mask: 1, k: 'stem' }, { vertexColors: true }),
  };
  for (const [k, list] of Object.entries(parts)) {
    if (!list.length) continue;
    const mesh = new THREE.Mesh(mergeGeometries(list), mats[k]);
    mesh.castShadow = true; mesh.receiveShadow = true;
    scene.add(mesh);
  }
}
