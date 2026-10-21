// Koi pond for the garden viewer: its own water, pond bed and swimming koi, kept out of the nihonga paint pass.
//
//   const pond = createPond({ THREE, scene, renderer, camera, sun, timeU });
//   gltf traverse:  if (pond.adopt(o, name)) return;     // takes over 'water', 'pond_bed', 'koi_*'
//                   ... then, once the mesh has its material:  pond.submerge(m, name === 'moss');
//   render():       pond.render();                       // right after camera.updateMatrixWorld()
//   resize():       pond.setSize(W0(), H0());
//
// Every pond pixel leaves the scene pass with mask POND_MASK in alpha (the painter's 0.667 after packing),
// so the paint pass can hand it through untouched, like the leaves. Motion is driven by timeU only, so the
// reduced-motion fallback (timeU frozen) gives a still pond, and garden.view({ t }) is reproducible.

import { createKoiSim, SPINE_N } from './koi.js';

import { PLANT_SPOTS } from './pondplants.js';

export const POND_MASK = 4.0;
const NRING = 12;   // ripple rings alive at once

// Blender outline of the pond (x, y) and its surface height, from build_garden.py
const POND = [[0.0, 7.6], [1.8, 6.6], [3.8, 6.9], [5.0, 8.1], [4.6, 9.6], [2.6, 10.3], [0.8, 9.9],
  [-0.6, 10.6], [-2.6, 10.2], [-3.4, 9.0], [-2.0, 8.0]];
const GROUND = -0.55, WY = GROUND - 0.07;

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
const bedDepth = (dd) => Math.min(0.45, dd * 0.8) + 0.05 - 0.07;   // terrain_h inside the pond, below the surface
const hash = (n) => { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };

// ------------------------------------------------------------------ shared GLSL
const POND_GLSL = /* glsl */`
uniform float uTime;
uniform vec2 uPoly[${POND.length}];
uniform vec3 uDeep;
float pHash(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
float pNoise(vec3 x) {
  vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(pHash(i), pHash(i + vec3(1,0,0)), f.x), mix(pHash(i + vec3(0,1,0)), pHash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(pHash(i + vec3(0,0,1)), pHash(i + vec3(1,0,1)), f.x), mix(pHash(i + vec3(0,1,1)), pHash(i + vec3(1,1,1)), f.x), f.y), f.z);
}
// same swell as the garden's windGust, so the water ruffles when the maples sway
float pondGust(vec3 wp) {
  return 0.6 + 0.4 * (0.5 + 0.5 * sin(uTime * 0.13 + wp.x * 0.04)) * (0.7 + 0.3 * sin(uTime * 0.29 + 1.7 + wp.z * 0.05));
}
// distance (m) from a Blender (x, y) point to the pond's rim
float rimDist(vec2 b) {
  float best = 1e9;
  for (int i = 0; i < ${POND.length}; i++) {
    vec2 a = uPoly[i], c = uPoly[(i + 1) % ${POND.length}], d = c - a;
    float t = clamp(dot(b - a, d) / dot(d, d), 0.0, 1.0);
    best = min(best, length(b - a - t * d));
  }
  return best;
}
// light that travelled down to a submerged point and back up to the eye: red goes first, then green
vec3 underwater(vec3 c, float depth, float cosView) {
  float cosT = sqrt(1.0 - (1.0 - cosView * cosView) / 1.77);   // refracted angle under the surface
  float L = max(depth, 0.0) * (1.0 + 1.0 / max(cosT, 0.3));
  vec3 T = exp(-vec3(1.7, 0.75, 1.9) * L);
  return c * T + uDeep * (1.0 - T);
}
`;

// height field of the surface (three.js x/z, metres) and its slope, with pixel footprint fw for filtering
const SURFACE_GLSL = /* glsl */`
uniform vec4 uRings[${NRING}];   // x, z, birth time, strength
float ripples(vec2 p, float fw, out float ruffle) {
  vec2 wind = normalize(vec2(0.9, -0.45));
  float g = pondGust(vec3(p.x, 0.0, p.y));
  // cat's paws: patches of breeze drift across, leaving glassy water between them
  float paw = pNoise(vec3(p * 0.45 - wind * uTime * 0.22, uTime * 0.04));
  paw = smoothstep(0.42, 0.78, paw) * (0.35 + 0.65 * g);
  ruffle = paw;
  float h = 0.0;
  // slow swell, always there
  h += 0.0016 * sin(dot(p, vec2(0.83, -0.55)) * 5.1 + uTime * 0.9);
  h += 0.0011 * sin(dot(p, vec2(0.31, 0.95)) * 7.3 - uTime * 1.13);
  // wind ripples: short crests running downwind, dropped once they are finer than a pixel
  float a = 0.0018, f = 9.0;
  vec2 q = p;
  for (int i = 0; i < 4; i++) {
    float k = clamp(1.4 - f * fw * 1.2, 0.0, 1.0);
    float n = pNoise(vec3(q * f - wind * uTime * (1.4 + f * 0.04), uTime * 0.5 + float(i) * 3.7));
    h += a * k * (n - 0.5) * 2.0 * (0.25 + 0.75 * paw);
    q = mat2(0.8, -0.6, 0.6, 0.8) * q; a *= 0.6; f *= 1.9;
  }
  // rings from koi kissing the surface and drips
  for (int i = 0; i < ${NRING}; i++) {
    vec4 r = uRings[i];
    float age = uTime - r.z;
    if (r.w <= 0.0 || age < 0.0 || age > 6.0) continue;
    float d = length(p - r.xy), front = 0.06 + age * 0.32;
    float x = d - front;
    float env = exp(-x * x / (0.004 + age * 0.012)) * exp(-age * 0.75) * r.w;
    float k = clamp(1.4 - 55.0 * fw * 0.8, 0.0, 1.0);
    h += 0.0028 * env * sin(x * 55.0) * k / (1.0 + d * 4.0);
  }
  return h;
}
vec3 surfaceNormal(vec2 p, float fw, out float ruffle) {
  float e = max(0.004, fw * 0.5);
  float r0, r1, r2;
  float h0 = ripples(p, fw, r0), hx = ripples(p + vec2(e, 0.0), fw, r1), hz = ripples(p + vec2(0.0, e), fw, r2);
  ruffle = r0;
  return normalize(vec3(-(hx - h0) / e, 1.0, -(hz - h0) / e));
}
`;

export function createPond({ THREE, scene, renderer, camera, sun, timeU }) {
  const B = (x, y, z) => new THREE.Vector3(x, z, -y);
  const polyU = { value: POND.map(([x, y]) => new THREE.Vector2(x, y)) };
  const deepU = { value: new THREE.Color(0.012, 0.03, 0.022) };    // colour the water scatters back, linear
  const ringsU = { value: Array.from({ length: NRING }, () => new THREE.Vector4(0, 0, -99, 0)) };
  const sunDirU = { value: new THREE.Vector3() };
  const sunEU = { value: sun.color.clone().multiplyScalar(sun.intensity) };
  const shared = (sh) => Object.assign(sh.uniforms, { uTime: timeU, uPoly: polyU, uDeep: deepU });

  // ---------------------------------------------------------------- planar reflection
  const reflRT = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 2 });
  reflRT.texture.generateMipmaps = true;                     // mip bias blurs it where the breeze ruffles the water
  reflRT.texture.minFilter = THREE.LinearMipmapLinearFilter;
  const mirrorCam = new THREE.PerspectiveCamera();
  const texMat = new THREE.Matrix4();
  const tmp = { n: new THREE.Vector3(0, 1, 0), o: B(0, 8.5, WY), v: new THREE.Vector3(), t: new THREE.Vector3(),
    look: new THREE.Vector3(), rot: new THREE.Matrix4(), plane: new THREE.Plane(), clip: new THREE.Vector4(), q: new THREE.Vector4() };
  const frustum = new THREE.Frustum(), projView = new THREE.Matrix4();
  const pondBox = new THREE.Box3().setFromPoints(POND.flatMap(([x, y]) => [B(x, y, WY - 0.1), B(x, y, WY + 0.05)]));
  let reflScale = 0.5;

  // three.js' Reflector, reduced to the one horizontal plane: mirror the camera through the water and
  // clip everything below the surface with an oblique near plane
  function renderReflection(hidden) {
    const { n, o, v, t, look, rot, plane, clip, q } = tmp;
    const camPos = v.setFromMatrixPosition(camera.matrixWorld);
    if (camPos.y < WY + 0.01) return false;
    const view = t.subVectors(o, camPos).reflect(n).negate().add(o);
    rot.extractRotation(camera.matrixWorld);
    look.set(0, 0, -1).applyMatrix4(rot).add(camPos);
    const target = look.sub(o).reflect(n).add(o);            // the look-at point, mirrored through the water
    mirrorCam.position.copy(view);
    mirrorCam.up.set(0, 1, 0).applyMatrix4(rot).reflect(n);
    mirrorCam.lookAt(target);
    mirrorCam.far = camera.far;
    mirrorCam.updateMatrixWorld();
    mirrorCam.projectionMatrix.copy(camera.projectionMatrix);
    texMat.set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1)
      .multiply(mirrorCam.projectionMatrix).multiply(mirrorCam.matrixWorldInverse);
    plane.setFromNormalAndCoplanarPoint(n, o).applyMatrix4(mirrorCam.matrixWorldInverse);
    clip.set(plane.normal.x, plane.normal.y, plane.normal.z, plane.constant);
    const e = mirrorCam.projectionMatrix.elements;
    q.set((Math.sign(clip.x) + e[8]) / e[0], (Math.sign(clip.y) + e[9]) / e[5], -1, (1 + e[10]) / e[14]);
    clip.multiplyScalar(2 / clip.dot(q));
    e[2] = clip.x; e[6] = clip.y; e[10] = clip.z + 1 - 0.003; e[14] = clip.w;

    for (const m of hidden) m.visible = false;
    const shadowAuto = renderer.shadowMap.autoUpdate, rt = renderer.getRenderTarget();
    renderer.shadowMap.autoUpdate = false;                    // the main pass refreshes the shadows
    renderer.setRenderTarget(reflRT);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(scene, mirrorCam);
    renderer.setRenderTarget(rt);
    renderer.shadowMap.autoUpdate = shadowAuto;
    for (const m of hidden) m.visible = true;
    return true;
  }

  // ---------------------------------------------------------------- water surface
  function waterMaterial(planar) {
    const uniforms = {
      tRefl: { value: reflRT.texture }, uTexMat: { value: texMat }, uReflOn: { value: 0 },
      uRings: ringsU, uSunDir: sunDirU, uSunE: sunEU,
      uBasin: { value: planar ? 0 : 1 },
    };
    shared({ uniforms });
    return new THREE.ShaderMaterial({
      uniforms, transparent: true, depthWrite: false,
      blending: THREE.CustomBlending,
      blendSrc: THREE.SrcAlphaFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,   // colour only; the mask pass sets alpha
      vertexShader: /* glsl */`
        uniform mat4 uTexMat; varying vec3 vW; varying vec4 vR;
        void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vR = uTexMat * w; gl_Position = projectionMatrix * viewMatrix * w; }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tRefl; uniform float uReflOn, uBasin; uniform vec3 uSunDir, uSunE;
        varying vec3 vW; varying vec4 vR;
        ${POND_GLSL}
        ${SURFACE_GLSL}
        vec3 toLin(vec3 c){ return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
        void main(){
          vec2 p = vW.xz;
          float fw = length(fwidth(p));
          float ruffle;
          vec3 n = surfaceNormal(p, fw, ruffle);
          vec3 V = normalize(cameraPosition - vW);
          float dist = length(cameraPosition - vW);
          // grazing views would see the ripples' backs; lean the normal toward up as it gets finer than a pixel
          n = normalize(mix(n, vec3(0.0, 1.0, 0.0), clamp(fw * 6.0, 0.0, 0.7)));
          float cosV = clamp(dot(n, V), 0.0, 1.0);
          float F = 0.05 + 0.95 * pow(1.0 - cosV, 5.0);   // a touch above water's 0.02 so the gold always shows

          // reflection: the mirrored garden, nudged by the ripples, blurred where the breeze ruffles it
          vec3 refl;
          if (uReflOn > 0.5) {
            vec2 uv = vR.xy / vR.w + n.xz * vec2(0.9, -0.9) * (0.35 / (1.0 + dist * 0.6));
            uv = clamp(uv, vec2(0.001), vec2(0.999));
            vec4 s = texture2D(tRefl, uv, ruffle * 2.5 + 0.5);
            // the sky writes display-space gold with alpha 0; reflected, its sheets melt into one soft glow
            vec3 sky = toLin(texture2D(tRefl, uv, 6.0).rgb);
            refl = mix(sky, s.rgb, clamp(s.a * 4.0, 0.0, 1.0));
          } else {
            refl = mix(uDeep * 4.0, toLin(vec3(0.84, 0.70, 0.36)), smoothstep(-0.05, 0.4, reflect(-V, n).y));
          }

          // what the water itself gives back: dark green-black in the deep, clear over the shallows
          vec2 bp = vec2(vW.x, -vW.z);
          float depth = uBasin > 0.5 ? 0.12 : min(0.45, rimDist(bp) * 0.8) - 0.02;
          float murk = clamp(0.06 + depth * 0.35, 0.0, 0.25);   // the bed and koi tint themselves by depth
          vec3 body = uDeep * 1.3;

          // sun glints on the crests, like flakes of gold
          vec3 H = normalize(uSunDir + V);
          float spec = pow(max(dot(n, H), 0.0), 1400.0) * 1.5 * (0.3 + 0.7 * ruffle);
          vec3 glint = uSunE * spec * F;

          float a = clamp(F + (1.0 - F) * murk, 0.0, 1.0);
          vec3 col = (refl * F + body * (1.0 - F) * murk + glint) / max(a, 1e-3);
          // the rim: a thin wet line where water meets moss and stone
          float rim = uBasin > 0.5 ? 1.0 : smoothstep(0.0, 0.06, rimDist(bp));
          gl_FragColor = vec4(col, a * mix(0.6, 1.0, rim));
        }`,
    });
  }
  // second pass over the same surface: leave the colour alone, stamp the painter's pond mask
  const maskMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: true,
    blending: THREE.CustomBlending,
    blendSrc: THREE.ZeroFactor, blendDst: THREE.OneFactor,
    blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.ZeroFactor,
    vertexShader: `void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, ${POND_MASK.toFixed(1)}); }`,
  });

  const shape = new THREE.Shape(POND.map(([x, y]) => new THREE.Vector2(x, y)));
  const pondGeo = new THREE.ShapeGeometry(shape, 1);
  pondGeo.rotateX(-Math.PI / 2);                     // Blender (x, y) -> three (x, -y) on the ground plane
  pondGeo.translate(0, WY, 0);
  const pondWater = new THREE.Mesh(pondGeo, waterMaterial(true));
  const pondMask = new THREE.Mesh(pondGeo, maskMat);
  pondWater.renderOrder = 2; pondMask.renderOrder = 3;
  scene.add(pondWater, pondMask);

  // ---------------------------------------------------------------- what lies under the water
  // The pond's floor is the moss terrain dipping down, plus the rocks and trunks standing in it. submerge()
  // adds to any garden material: below the surface it takes on the pond-bed colour (floor only), catches
  // caustics in the shallows and is seen through the water's depth tint.
  const SUB_GLSL = POND_GLSL + `
    float caustic(vec2 p){
      vec2 w = normalize(vec2(0.9, -0.45)) * uTime * 0.05;
      float a = pNoise(vec3(p * 3.1 + w, uTime * 0.35)), b = pNoise(vec3(p * 3.9 - w * 1.3 + 5.0, uTime * 0.29 + 2.0));
      return pow(clamp(1.0 - abs(a - b) * 4.0, 0.0, 1.0), 5.0);
    }`;
  function submerge(m, floor) {
    const prev = m.onBeforeCompile;
    m.onBeforeCompile = (sh, r) => {
      if (prev) prev(sh, r);
      shared(sh);
      const hasW = sh.vertexShader.includes('varying vec3 vWPos');
      if (!hasW) sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;')
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\n' + (hasW ? '' : 'varying vec3 vWPos;\n') + SUB_GLSL)
        .replace('#include <emissivemap_fragment>', `
          float subDepth = ${WY.toFixed(3)} - vWPos.y;
          ${floor ? `if (subDepth > 0.0) {   // silt and fallen-leaf mulch on the floor
            vec2 bp = vec2(vWPos.x, -vWPos.z);
            float mott = pNoise(vec3(bp * 4.0, 1.0)) * 0.6 + pNoise(vec3(bp * 13.0, 2.0)) * 0.4;
            diffuseColor.rgb = mix(diffuseColor.rgb, mix(vec3(0.05, 0.05, 0.03), vec3(0.16, 0.12, 0.06), smoothstep(0.3, 0.7, mott)),
                                   smoothstep(0.0, 0.03, subDepth));
          }` : ''}
          #include <emissivemap_fragment>`)
        .replace('#include <opaque_fragment>', `
          if (subDepth > 0.0) {
            float fw = length(fwidth(vWPos.xz));
            float shallow = exp(-subDepth * 3.0) * clamp(1.4 - fw * 8.0, 0.0, 1.0);
            outgoingLight += reflectedLight.directDiffuse * caustic(vWPos.xz) * 1.6 * shallow;
            outgoingLight = underwater(outgoingLight, subDepth, normalize(cameraPosition - vWPos).y);
          }
          #include <opaque_fragment>`);
    };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + '|sub' + (floor ? 'F' : '');
    return m;
  }

  // ---------------------------------------------------------------- koi
  // Bodies are built once, nose at x = 0 running back to the tail tip at x = -1 (-reach for the long-finned
  // one), and bent onto each fish's spine in the vertex shader: the spine is the trail its head swam (koi.js),
  // so the whole body follows every curve, with a travelling wave of tail beats on top.
  //   aFin: 0 body, 1 dorsal, 2 pectoral, 3 tail, 4 pelvic, 5 anal; aSide: fin side, or cos(angle) round the body
  //   aR: 0 at a fin's root to 1 at its edge; aRay: position across the fin's rays
  const PROF_T = [0, 0.012, 0.04, 0.09, 0.16, 0.24, 0.33, 0.45, 0.58, 0.68, 0.76, 0.8];
  const PROF_W = [0.006, 0.03, 0.058, 0.086, 0.106, 0.118, 0.12, 0.108, 0.082, 0.056, 0.034, 0.026];   // half width
  const PROF_U = [0.005, 0.026, 0.05, 0.075, 0.095, 0.108, 0.114, 0.104, 0.08, 0.055, 0.036, 0.03];    // back
  const PROF_D = [0.005, 0.024, 0.046, 0.066, 0.08, 0.088, 0.09, 0.082, 0.062, 0.044, 0.03, 0.026];    // belly
  function prof(arr, t) {
    t = Math.min(0.8, Math.max(0, t));
    let i = 0; while (i < PROF_T.length - 2 && t > PROF_T[i + 1]) i++;
    const u = (t - PROF_T[i]) / (PROF_T[i + 1] - PROF_T[i]), s = u * u * (3 - 2 * u);
    return arr[i] + (arr[i + 1] - arr[i]) * s;
  }
  const noseDrop = (t) => -0.014 * Math.max(0, 1 - t / 0.3) ** 2;     // the mouth sits a touch low
  const EYE = [-0.07, noseDrop(0.07) + prof(PROF_U, 0.07) * 0.28, prof(PROF_W, 0.07) * 0.93];

  function koiGeometry(long) {
    const pos = [], fin = [], side = [], rr = [], ray = [], idx = [];
    const v = (x, y, z, f, s, r, q) => { pos.push(x, y, z); fin.push(f); side.push(s); rr.push(r); ray.push(q); return pos.length / 3 - 1; };
    // body: rings round the spine, packed toward the rounded nose
    const NS = 46, NR = 26, ring0 = [];
    const nose = v(0.002, noseDrop(0), 0, 0, 0.2, 0, 0);
    for (let i = 0; i <= NS; i++) {
      const k = i / NS, t = 0.004 + 0.796 * (0.45 * k + 0.55 * (1 - Math.cos(Math.PI / 2 * k)));
      const w = prof(PROF_W, t), hu = prof(PROF_U, t), hd = prof(PROF_D, t), yc = noseDrop(t);
      ring0.push(pos.length / 3);
      for (let j = 0; j < NR; j++) {
        const a = 2 * Math.PI * j / NR, ca = Math.cos(a), sa = Math.sin(a);
        const sp = (c) => Math.sign(c) * Math.abs(c) ** 0.88;         // a little fuller than an ellipse
        const ridge = 0.006 * Math.max(0, ca) ** 10 * Math.sin(Math.PI * Math.min(1, t / 0.8));
        v(-t, yc + sp(ca) * (ca > 0 ? hu : hd) + ridge, sp(sa) * w, 0, ca, j / NR, 0);
      }
    }
    for (let j = 0; j < NR; j++) idx.push(nose, ring0[0] + (j + 1) % NR, ring0[0] + j);
    for (let i = 0; i < NS; i++) for (let j = 0; j < NR; j++) {
      const a = ring0[i] + j, b = ring0[i] + (j + 1) % NR, c = ring0[i + 1] + j, d = ring0[i + 1] + (j + 1) % NR;
      idx.push(a, b, c, b, d, c);
    }
    const tail = ring0[NS];
    const cap = v(-0.8, noseDrop(0.8), 0, 0, 0, 0, 0);
    for (let j = 0; j < NR; j++) idx.push(cap, tail + j, tail + (j + 1) % NR);
    const bodyCount = idx.length;

    // a fan of rays from a root point out to a rim; map(along, out) places a point of the fin
    function fan(f, s, rim, root, map, rings = 4) {
      const base = pos.length / 3;
      for (let k = 0; k < rim.length; k++) for (let q = 0; q <= rings; q++) {
        const r = q / rings, [ru, rv] = root, [eu, ev] = rim[k];
        const [x, y, z] = map(ru + (eu - ru) * r, rv + (ev - rv) * r, r);
        v(x, y, z, f, s, r, k / (rim.length - 1));
      }
      for (let k = 0; k + 1 < rim.length; k++) for (let q = 0; q < rings; q++) {
        const a = base + k * (rings + 1) + q, b = a + rings + 1;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
    const fl = long ? 1.75 : 1;           // the butterfly koi trails long, soft fins
    // pectorals: rounded paddles low on the flanks
    const pecRim = [[0, 0], [-0.015, 0.07], [-0.04, 0.14], [-0.085, 0.185], [-0.14, 0.19], [-0.18, 0.15], [-0.175, 0.08], [-0.12, 0.03], [-0.07, 0]]
      .map(([a, b]) => [a * fl, b * (long ? 1.45 : 1)]);
    for (const s of [1, -1]) {
      const t0 = 0.175, yb = noseDrop(t0) - prof(PROF_D, t0) * 0.55, zb = prof(PROF_W, t0) * 0.78;
      fan(2, s, pecRim, [-0.035, 0], (a, b) => [-t0 + a, yb - b * 0.28, s * (zb + b)]);
    }
    // pelvics: smaller, under the belly
    const pelRim = [[0, 0], [-0.02, 0.05], [-0.06, 0.09], [-0.1, 0.085], [-0.11, 0.04], [-0.07, 0]].map(([a, b]) => [a * fl, b * (long ? 1.3 : 1)]);
    for (const s of [1, -1]) {
      const t0 = 0.43, yb = noseDrop(t0) - prof(PROF_D, t0) * 0.85, zb = prof(PROF_W, t0) * 0.4;
      fan(4, s, pelRim, [-0.03, 0], (a, b) => [-t0 + a, yb - b * 0.55, s * (zb + b * 0.8)]);
    }
    // dorsal: a sail along the back, highest at the front
    const dRim = [];
    for (let k = 0; k <= 8; k++) {
      const u = k / 8, t = 0.34 + u * 0.33;
      dRim.push([t, (0.075 - 0.04 * u) * (long ? 1.6 : 1), u]);
    }
    {
      const base = pos.length / 3, rings = 3;
      for (let k = 0; k < dRim.length; k++) for (let q = 0; q <= rings; q++) {
        const r = q / rings, [t, h] = dRim[k];
        const yb = noseDrop(t) + prof(PROF_U, t) - 0.004;
        v(-t - r * h * (long ? 0.7 : 0.45), yb + r * h, 0, 1, 0, r, k / (dRim.length - 1));
      }
      for (let k = 0; k + 1 < dRim.length; k++) for (let q = 0; q < rings; q++) {
        const a = base + k * (rings + 1) + q, b = a + rings + 1;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
    // anal fin
    fan(5, 0, [[0, 0], [-0.03, 0.045], [-0.075, 0.05], [-0.08, 0.02], [-0.06, 0]].map(([a, b]) => [a * fl, b * fl]), [-0.02, 0],
      (a, b) => [-0.66 + a, noseDrop(0.66) - prof(PROF_D, 0.66) * 0.95 - b, 0]);
    // tail: a broad, softly forked fan from the narrow root
    const tailRim = [], TL = long ? 0.5 : 0.22;
    for (let k = 0; k <= 14; k++) {
      const th = -0.78 + 1.56 * k / 14, at = Math.abs(th);
      const len = TL * (0.6 + 0.4 * Math.min(1, at / 0.55)) * (1 - 0.18 * Math.max(0, (at - 0.6) / 0.18));
      tailRim.push([-Math.cos(th) * len, Math.sin(th) * len * 1.05]);
    }
    fan(3, 0, tailRim, [0.02, 0], (a, b) => [-0.78 + a, noseDrop(0.8) + b, 0], 5);

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('aFin', new THREE.Float32BufferAttribute(fin, 1));
    g.setAttribute('aSide', new THREE.Float32BufferAttribute(side, 1));
    g.setAttribute('aR', new THREE.Float32BufferAttribute(rr, 1));
    g.setAttribute('aRay', new THREE.Float32BufferAttribute(ray, 1));
    g.setIndex(idx);
    g.computeVertexNormals();
    g.addGroup(0, bodyCount, 0);                   // the body is opaque
    g.addGroup(bodyCount, idx.length - bodyCount, 1);   // fins are veils, blended over it
    g.userData.reach = long ? 0.8 + TL : 1.0;
    return g;
  }
  const koiGeo = koiGeometry(false), koiGeoLong = koiGeometry(true);

  // the pond's koi: who they are and how they like to swim
  //   kind: 0 kohaku, 1 sanke, 2 showa, 3 ogon, 4 tancho, 5 chagoi, 6 shiro utsuri, 7 asagi
  const KOI = [
    { kind: 0, len: 0.62, social: 0.7, bold: 0.6, pace: 0.13 },
    { kind: 1, len: 0.55, social: 0.8, bold: 0.4, pace: 0.12 },
    { kind: 2, len: 0.58, social: 0.3, bold: 0.5, pace: 0.14 },
    { kind: 3, len: 0.5, social: 0.6, bold: 0.8, pace: 0.15 },
    { kind: 4, len: 0.45, social: 0.9, bold: 0.3, pace: 0.13 },
    { kind: 5, len: 0.7, social: 0.5, bold: 1.0, pace: 0.11 },      // chagoi: the big friendly one, first to the food
    { kind: 6, len: 0.52, social: 0.2, bold: 0.3, pace: 0.12 },     // a loner
    { kind: 7, len: 0.48, social: 0.7, bold: 0.5, pace: 0.14 },
    { kind: 3, len: 0.55, social: 0.4, bold: 0.7, pace: 0.12, long: true, platinum: true },   // butterfly
    { kind: 0, len: 0.38, social: 0.85, bold: 0.6, pace: 0.16 },
  ];
  const KIND_COL = {   // base, patch, fin
    0: [[0.86, 0.84, 0.79], [0.78, 0.09, 0.03], [0.88, 0.86, 0.82]],
    1: [[0.86, 0.84, 0.79], [0.80, 0.12, 0.03], [0.88, 0.86, 0.82]],
    2: [[0.025, 0.022, 0.022], [0.76, 0.11, 0.03], [0.85, 0.82, 0.76]],
    3: [[0.95, 0.66, 0.10], [1.0, 0.82, 0.35], [0.98, 0.72, 0.2]],
    4: [[0.88, 0.86, 0.82], [0.82, 0.08, 0.03], [0.9, 0.88, 0.84]],
    5: [[0.40, 0.25, 0.12], [0.22, 0.13, 0.06], [0.42, 0.28, 0.15]],
    6: [[0.025, 0.022, 0.022], [0.88, 0.86, 0.82], [0.85, 0.83, 0.79]],
    7: [[0.34, 0.42, 0.52], [0.86, 0.32, 0.08], [0.82, 0.5, 0.3]],
  };

  const KOI_VERT = /* glsl */`
    attribute float aFin, aSide, aR, aRay;
    uniform vec3 uSp[${SPINE_N}];
    uniform float uLen, uReach, uPhase, uAmp, uRoll, uScull, uFlutter;
    varying vec3 vRest; varying vec3 vWP; varying float vFin, vSide, vR, vRay;
    // the spine: a Catmull-Rom curve through the head's trail, u in body lengths from the nose
    vec3 koiSpine(float u, out vec3 F) {
      float f = clamp(u / uReach, 0.0, 1.0) * ${(SPINE_N - 1).toFixed(1)};
      int i = int(min(floor(f), ${(SPINE_N - 2).toFixed(1)}));
      float t = f - float(i);
      vec3 p0 = uSp[max(i - 1, 0)], p1 = uSp[i], p2 = uSp[i + 1], p3 = uSp[min(i + 2, ${SPINE_N - 1})];
      if (i == 0) p0 = 2.0 * p1 - p2;
      if (i == ${SPINE_N - 2}) p3 = 2.0 * p2 - p1;
      vec3 b = p2 - p0, c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3, d = -p0 + 3.0 * p1 - 3.0 * p2 + p3;
      vec3 der = 0.5 * (b + 2.0 * c * t + 3.0 * d * t * t);
      F = -normalize(der + vec3(1e-6, 0.0, 0.0));
      return 0.5 * (2.0 * p1 + b * t + c * t * t + d * t * t * t);
    }
    // sideways sway of the travelling wave, in body lengths, and its slope
    float koiWave(float u, out float slope) {
      float e = 0.1 + 0.9 * pow(clamp(u / 0.85, 0.0, 1.3), 2.0);
      float k = 5.4, a = uAmp * 0.07;
      slope = a * e * -k * cos(uPhase - u * k);
      return a * e * sin(uPhase - u * k);
    }`;

  function koiMaterials(spec, seed, geo) {
    const body = new THREE.MeshLambertMaterial({ side: THREE.DoubleSide });
    const fins = new THREE.MeshLambertMaterial({ side: THREE.DoubleSide, transparent: true, depthWrite: false });
    const col = KIND_COL[spec.kind];
    const U = {
      uSp: { value: Array.from({ length: SPINE_N }, () => new THREE.Vector3()) },
      uLen: { value: spec.len }, uReach: { value: geo.userData.reach },
      uPhase: { value: 0 }, uAmp: { value: 0 }, uRoll: { value: 0 }, uScull: { value: 0 }, uFlutter: { value: 0 },
      uBase: { value: new THREE.Color(...(spec.platinum ? [0.9, 0.9, 0.87] : col[0])) },
      uSpot: { value: new THREE.Color(...(spec.platinum ? [1, 1, 0.96] : col[1])) },
      uFinC: { value: new THREE.Color(...(spec.platinum ? [0.92, 0.92, 0.9] : col[2])) },
      uKind: { value: spec.kind }, uSeed: { value: seed },
      uMetal: { value: spec.kind === 3 ? 1 : 0 }, uEye: { value: new THREE.Vector3(...EYE) },
    };
    const compile = (sh) => {
      shared(sh);
      Object.assign(sh.uniforms, U);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\n' + KOI_VERT)
        .replace('#include <beginnormal_vertex>', `
          vec3 p = position;
          vRest = position; vFin = aFin; vSide = aSide; vR = aR; vRay = aRay;
          // fins: pectorals scull when slow and fold back when cruising; the tail and dorsal trail and ripple
          if (aFin > 1.5 && aFin < 2.5 || aFin > 3.5 && aFin < 4.5) {
            float s = sign(p.z), lever = aR;
            float tuck = (1.0 - uScull) * (aFin < 2.5 ? 0.55 : 0.3);
            float flap = sin(uFlutter + s * 0.5 + aR * 0.8) * (0.15 + 0.6 * uScull);
            p.x -= tuck * lever * 0.09;
            p.z = s * (abs(p.z) - lever * tuck * 0.06);
            p.y += flap * lever * 0.07;
          }
          if (aFin > 2.5 && aFin < 3.5) {   // the soft tail fan lies over toward flat and flexes with each beat
            float th = 1.05 + 0.3 * sin(uPhase - 1.6);
            p.z = p.y * sin(th); p.y = p.y * cos(th);
          }
          float u = clamp(-p.x, 0.0, uReach);
          float slope, lat = koiWave(u, slope);
          if (aFin > 2.5 && aFin < 3.5) lat += 0.025 * aR * aR * sin(uPhase - 1.3 - aR * 2.2) * (0.35 + uAmp);
          if (aFin > 0.5 && aFin < 1.5) lat += 0.012 * aR * sin(uPhase * 0.7 - u * 9.0 + 1.0);
          if (aFin > 4.5) lat += 0.01 * aR * sin(uPhase - u * 6.0 - 1.0);
          vec3 F, P = koiSpine(u, F);
          vec3 S = normalize(vec3(-F.z, 0.0, F.x)), Uv = cross(S, F);
          float cr = cos(uRoll), sr = sin(uRoll);
          vec3 S2 = S * cr + Uv * sr, U2 = Uv * cr - S * sr;
          vec3 koiPos = P + (S2 * (p.z + lat) + U2 * p.y) * uLen;
          vec3 objectNormal = normalize(F * normal.x + U2 * normal.y + S2 * normal.z - F * normal.z * slope);`)
        .replace('#include <begin_vertex>', 'vec3 transformed = koiPos;')
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWP = transformed;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>
          uniform vec3 uBase, uSpot, uFinC, uEye; uniform float uKind, uSeed, uMetal;
          varying vec3 vRest; varying vec3 vWP; varying float vFin, vSide, vR, vRay;
          ${POND_GLSL}
          float koiNoise(vec3 p) { return pNoise(p) * 0.65 + pNoise(p * 2.3 + 7.1) * 0.35; }`)
        .replace('vec4 diffuseColor = vec4( diffuse, opacity );', `
          vec3 c = uBase; float alpha = 1.0;
          float t = -vRest.x;
          vec3 lp = vRest * vec3(1.0, 1.7, 1.7) + uSeed * 3.17;
          if (vFin < 0.5) {
            float top = smoothstep(-0.35, 0.25, vSide);              // patterns sit on the back and upper flanks
            float n = koiNoise(lp * 7.0), n2 = koiNoise(lp * 15.0 + 4.0);
            float notNose = smoothstep(0.015, 0.05, t);
            if (uKind < 0.5) {                                        // kohaku: bold red steps on white
              c = mix(uBase, uSpot, smoothstep(0.47, 0.5, n + 0.12 * top - 0.08) * top * notNose);
            } else if (uKind < 1.5) {                                 // sanke: kohaku with small ink spots on the back
              c = mix(uBase, uSpot, smoothstep(0.48, 0.51, n + 0.12 * top - 0.08) * top * notNose);
              c = mix(c, vec3(0.02), smoothstep(0.66, 0.69, n2) * smoothstep(0.2, 0.7, vSide) * step(0.2, t));
            } else if (uKind < 2.5) {                                 // showa: ink wrapping round, red and white over it
              c = mix(uBase, vec3(0.86, 0.84, 0.79), smoothstep(0.55, 0.58, n2) * top);
              c = mix(c, uSpot, smoothstep(0.52, 0.55, n) * top * notNose);
            } else if (uKind < 3.5) {                                 // ogon: one metallic colour
              c = mix(uBase, uSpot, 0.25 + 0.25 * n);
            } else if (uKind < 4.5) {                                 // tancho: a single red sun on the head
              float r = length(vec2((t - 0.1) * 1.1, vRest.z));
              c = mix(uBase, uSpot, (1.0 - smoothstep(0.042, 0.048, r)) * step(0.0, vSide));
            } else if (uKind < 5.5) {                                 // chagoi: tea brown
              c = uBase * (0.9 + 0.2 * n);
            } else if (uKind < 6.5) {                                 // shiro utsuri: white over ink
              c = mix(uBase, uSpot, smoothstep(0.47, 0.5, n + 0.1 * top));
            } else {                                                  // asagi: blue net above, red below
              c = mix(uSpot, uBase, smoothstep(-0.25, 0.15, vSide) * smoothstep(0.03, 0.09, t));
              c = mix(c, vec3(0.8, 0.82, 0.8), (1.0 - smoothstep(0.03, 0.1, t)) * step(0.0, vSide));
            }
            // pale belly (the inky varieties wrap round)
            float belly = 1.0 - smoothstep(-0.75, -0.3, vSide);
            if (uKind < 1.5 || uKind > 2.5 && uKind < 4.5) c = mix(c, vec3(0.9, 0.88, 0.84) * (uKind > 2.5 && uKind < 3.5 ? vec3(1.0, 0.85, 0.6) : vec3(1.0)), belly * 0.7);
            // scales: a fine net, strongest on the bronze and blue koi
            vec2 sc = vec2(t * 34.0 + vR * 30.0, t * 34.0 - vR * 30.0);   // diamond-shaped scale net
            vec2 e = abs(fract(sc) - 0.5);
            float net = smoothstep(0.38, 0.5, max(e.x, e.y)) * smoothstep(0.1, 0.18, t) * smoothstep(0.78, 0.66, t);
            float netK = uKind > 4.5 && uKind < 5.5 || uKind > 6.5 ? 0.22 : uKind > 2.5 && uKind < 3.5 ? 0.12 : 0.05;
            c *= 1.0 - net * netK;
            // eyes: dark with a gold ring; a small mouth
            float de = length(vec3(vRest.x - uEye.x, vRest.y - uEye.y, abs(vRest.z) - uEye.z));
            c = mix(c, vec3(0.75, 0.55, 0.2), 1.0 - smoothstep(0.014, 0.017, de));
            c = mix(c, vec3(0.01), 1.0 - smoothstep(0.009, 0.011, de));
            c = mix(c, vec3(0.35, 0.12, 0.1), (1.0 - smoothstep(0.004, 0.012, t)) * step(vRest.y, -0.012));
          } else {
            // fins: veils of the base colour with stiffer rays, thinning to the edge
            float rays = pow(abs(sin(vRay * 3.14159 * (vFin > 2.5 && vFin < 3.5 ? 12.0 : 7.0))), 6.0);
            c = uFinC * (0.92 + 0.12 * rays);
            if ((uKind > 1.5 && uKind < 2.5 || uKind > 5.5 && uKind < 6.5) && vFin > 1.5 && vFin < 2.5)
              c = mix(c, vec3(0.02), 1.0 - smoothstep(0.25, 0.45, vR));       // motoguro: ink at the pectoral root
            if (uKind > 0.5 && uKind < 1.5 && vFin > 1.5 && vFin < 3.5)
              c = mix(c, vec3(0.03), step(0.82, fract(vRay * 3.0 + uSeed * 0.37)) * step(0.15, vR));   // sanke fin stripes
            if (uKind > 6.5 && vFin > 1.5 && vFin < 2.5) c = mix(uSpot, c, smoothstep(0.1, 0.6, vR));
            if (vFin > 0.5 && vFin < 1.5 && uKind < 2.5) c = mix(c, uBase, 0.0);
            alpha = mix(0.85, 0.3, vR) + rays * 0.15;
            alpha *= 1.0 - smoothstep(0.85, 1.0, vR) * 0.6;
          }
          vec4 diffuseColor = vec4(c, alpha);`)
        .replace('#include <opaque_fragment>', `
          #if NUM_DIR_LIGHTS > 0
          {   // the wet sheen on the back, strong on the metallic koi
            vec3 Vv = normalize(vViewPosition);
            vec3 Hh = normalize(directionalLights[0].direction + Vv);
            float sp = pow(max(dot(normal, Hh), 0.0), mix(14.0, 36.0, uMetal));
            outgoingLight += directionalLights[0].color * sp * (0.025 + 0.3 * uMetal) * (vFin < 0.5 ? 1.0 : 0.3) * mix(vec3(1.0), diffuseColor.rgb, uMetal);
          }
          #endif
          {
            float depth = ${WY.toFixed(3)} - vWP.y;
            vec3 V = normalize(cameraPosition - vWP);
            outgoingLight = underwater(outgoingLight, depth, V.y);
          }
          #include <opaque_fragment>`);
    };
    for (const m of [body, fins]) { m.userData = U; m.onBeforeCompile = compile; m.customProgramCacheKey = () => 'koi2'; }
    return [body, fins];
  }

  // open water around a point: the rim, the rocks set along it, and the plants at the edge
  const RIM_ROCKS = POND.flatMap(([x, y], k) => { const [nx, ny] = POND[(k + 1) % POND.length]; return [[x, y], [(x + nx) / 2, (y + ny) / 2]]; });
  function clearance(x, y) {
    const d = polyDist(x, y);
    if (!inPoly(x, y)) return -d;
    let c = d;
    for (const p of PLANT_SPOTS) c = Math.min(c, Math.hypot(x - p.x, y - p.y) - p.r);
    for (const [rx, ry] of RIM_ROCKS) c = Math.min(c, Math.hypot(x - rx, y - ry) - 0.38);
    return c;
  }
  const sim = createKoiSim({
    clearance, floorDepth: (x, y) => (inPoly(x, y) ? bedDepth(polyDist(x, y)) : 0),
    bounds: [-3.6, 6.4, 5.2, 10.8], fish: KOI,
  });
  const koi = KOI.map((spec, i) => {
    const geo = spec.long ? koiGeoLong : koiGeo;
    const mats = koiMaterials(spec, i + 1, geo), mat = mats[0];
    const mesh = new THREE.Mesh(geo, mats);
    mesh.renderOrder = 1;                // fins blend before the water surface (2) goes over them
    mesh.castShadow = false; mesh.receiveShadow = true;
    mesh.frustumCulled = false;          // placed in the shader; the pond's own box decides visibility
    scene.add(mesh);
    return { mesh, mat, spec, pose: { spine: new Float32Array(SPINE_N * 3) } };
  });
  function placeKoi(t) {
    sim.advance(t);
    koi.forEach((k, i) => {
      const p = sim.pose(i, k.pose, k.mesh.geometry.userData.reach), u = k.mat.userData;
      for (let j = 0; j < SPINE_N; j++) u.uSp.value[j].set(p.spine[j * 3], WY - p.spine[j * 3 + 2], -p.spine[j * 3 + 1]);
      u.uPhase.value = p.phase; u.uAmp.value = p.amp; u.uRoll.value = p.roll;
      u.uScull.value = p.scull; u.uFlutter.value = p.flutter;
    });
  }

  // food floating on the water, until a koi takes it
  const PELLETS = 96;
  const pelletMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.011, 8, 6),
    new THREE.MeshLambertMaterial({ color: new THREE.Color(0.32, 0.2, 0.09) }), PELLETS);
  pelletMesh.frustumCulled = false; pelletMesh.count = 0;
  scene.add(pelletMesh);
  const pm = new THREE.Matrix4();
  function placePellets(t) {
    let n = 0;
    for (const p of sim.pellets) {
      if (!p.alive || p.t0 > t || n >= PELLETS) continue;
      const age = t - p.t0, bob = 0.002 * Math.sin(t * 2.1 + p.seed * 20) - 0.006 * Math.exp(-age * 6);
      pm.makeTranslation(p.x, WY + 0.003 + bob, -p.y);
      pelletMesh.setMatrixAt(n++, pm);
    }
    pelletMesh.count = n;
    pelletMesh.instanceMatrix.needsUpdate = true;
  }

  // ---------------------------------------------------------------- ripple rings
  // drips from the maple fire on fixed periods; the shishi-odoshi's spill when it tips; the rest of the slots
  // carry the newest rings the koi made (nosing the surface, gulping food) and food landing on the water
  const RING_SLOTS = [
    { period: 4.1, src: 'drop' }, { period: 5.9, src: 'drop' }, { period: 6.7, src: 'drop' },
    { period: 30, src: 'pour' },
  ];
  function dropPoint(n) {
    for (let k = 0; k < 50; k++) {
      // mostly under the big maple by the pond
      const x = 3.6 + (hash(n * 7 + k) - 0.5) * 4.5, y = 8.5 + (hash(n * 13 + k * 5) - 0.5) * 3.5;
      if (inPoly(x, y) && polyDist(x, y) > 0.2) return [x, y];
    }
    return [1.5, 8.6];
  }
  function placeRings(t) {
    RING_SLOTS.forEach((r, i) => {
      const off = hash(i + 77) * r.period, n = Math.floor((t + off) / r.period), t0 = n * r.period - off;
      const v = ringsU.value[i];
      if (r.src === 'pour') { if (pour) v.set(pour.x, -pour.y, pour.last(t), 1.0); else v.set(0, 0, t0, 0); return; }   // the shishi-odoshi's spill
      const [x, y] = dropPoint(n * 8 + i); v.set(x, -y, t0, 0.5 + 0.5 * hash(n + i));
    });
    const rings = sim.rings;
    let j = rings.length - 1;
    while (j >= 0 && rings[j].t > t) j--;                    // food still in the air
    for (let i = RING_SLOTS.length; i < NRING; i++, j--) {
      const r = rings[j];
      if (r && t - r.t < 6) ringsU.value[i].set(r.x, -r.y, r.t, r.s); else ringsU.value[i].set(0, 0, -99, 0);
    }
  }

  // toss a handful of food where the viewer is looking, or failing that onto the nearest water
  const ray = new THREE.Ray(), hit = new THREE.Vector3(), waterPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -WY);
  function feed() {
    camera.getWorldPosition(ray.origin);
    camera.getWorldDirection(ray.direction);
    let x, y;
    if (ray.intersectPlane(waterPlane, hit) && clearance(hit.x, -hit.z) > 0.1 && hit.distanceTo(ray.origin) < 14) { x = hit.x; y = -hit.z; }
    else {
      x = ray.origin.x; y = -ray.origin.z;
      for (let k = 0; k < 400 && clearance(x, y) < 0.5; k++) {   // walk uphill on open water
        const e = 0.05, gx = clearance(x + e, y) - clearance(x - e, y), gy = clearance(x, y + e) - clearance(x, y - e), l = Math.hypot(gx, gy) || 1;
        x += gx / l * 0.05; y += gy / l * 0.05;
      }
    }
    sim.feed(x, y);
    return [x, y];
  }

  // ---------------------------------------------------------------- hooks for the viewer
  const replaced = [];
  let pour = null;   // { x, y (Blender), last(t): when the latest spill landed }, set by the viewer
  function adopt(o, name) {
    if (name === 'water' || name === 'koi_orange' || name === 'koi_white') {
      o.visible = false;                         // replaced by the pond's own water and koi
      replaced.push(o);
      return true;
    }
    if (name === 'pond_bed') {             // pokes through the moss floor it sits under; the moss is the bed
      o.visible = false;
      replaced.push(o);
      return true;
    }
    return false;
  }

  const hideForReflection = [pondWater, pondMask, pelletMesh, ...koi.map((k) => k.mesh)];
  const camAt = new THREE.Vector3();
  function render() {
    const t = timeU.value;
    camera.getWorldPosition(camAt);
    sim.visitor = [camAt.x, -camAt.z];     // curious koi come over to someone standing at the edge
    placeKoi(t);
    placePellets(t);
    placeRings(t);
    sunDirU.value.copy(sun.position).sub(sun.target.position).normalize();
    projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(projView);
    const on = frustum.intersectsBox(pondBox) && renderReflection(hideForReflection);
    pondWater.material.uniforms.uReflOn.value = on ? 1 : 0;
  }
  function setSize(w, h) { reflRT.setSize(Math.max(1, Math.floor(w * reflScale)), Math.max(1, Math.floor(h * reflScale))); }
  setSize(renderer.domElement.width, renderer.domElement.height);

  return {
    adopt, submerge, render, setSize, feed, mask: POND_MASK, koi: sim,
    set pour(p) { pour = p; },
    set reflectionScale(s) { reflScale = s; setSize(renderer.domElement.width, renderer.domElement.height); },
  };
}
