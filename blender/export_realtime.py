"""Export garden.blend for the real-time three.js viewer.

python3 export_realtime.py [--blend ../garden.blend] [--out web]

Writes into --out:
  garden.glb      static geometry, one merged mesh per material (foliage and gravel removed)
  leaves.bin      foliage as instanced leaf cards (thinned + enlarged so coverage stays the same)
  leaves.json     group table for leaves.bin
  materials.json  procedural colour recipe per material, read back by the viewer's shaders
  height.bin      walkable-surface heightmap so the WASD camera follows floors, gravel and moss
Blender is Z-up; glTF/three.js is Y-up, so (x, y, z) -> (x, z, -y) everywhere below.
"""
import argparse
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
ap = argparse.ArgumentParser()
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--blend", default=os.path.join(here, "..", "garden.blend"))
ap.add_argument("--out", default=os.path.join(here, "web"))
args = ap.parse_args(argv)
os.makedirs(args.out, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=args.blend)
GROUND = -0.55
rng = np.random.default_rng(5)


def to_yup(p):
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


# ---------------------------------------------------------------- organic gravel outline
# Signed distance (m, Blender x/y) to the edge of the raked gravel: negative on gravel, positive on moss.
# The viewer's shader carries the same function (GRAVEL_SDF in index.html) for the cut and the raking: keep in step.
def _cap(x, y, ax, ay, bx, by, r):
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(x - ax - t * dx, y - ay - t * dy) - r


def _smin(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


def gravel_sdf(x, y):
    d = _cap(x, y, -6.6, 3.4, 6.4, 3.2, 1.85)
    d = _smin(d, _cap(x, y, -0.4, 5.0, 1.3, 5.75, 0.7), 1.2)       # gravel reaching up toward the pond
    d = _smin(d, _cap(x, y, -5.4, 4.8, -6.3, 8.3, 0.7), 1.2)       # a stream running back past the lantern
    d = max(d, -_cap(x, y, 3.9, 6.6, 2.9, 4.5, 0.42))              # moss peninsula from the right
    d += 0.10 * math.sin(x * 1.7 + y * 0.6) + 0.07 * math.sin(x * 0.7 - y * 2.3 + 1.3)
    return max(d, 1.56 - y)                                        # straight along the drip edge


# ---------------------------------------------------------------- material recipes

def recipe(m):
    """Read the build script's procedural node setup back into plain numbers."""
    nt = m.node_tree
    r = {"name": m.name}
    ramp = next((n for n in nt.nodes if n.type == "VALTORGB"), None)
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if ramp and not any(n.type == "ATTRIBUTE" for n in nt.nodes):
        els = ramp.color_ramp.elements
        r["c1"] = list(els[0].color[:3])
        r["c2"] = list(els[-1].color[:3])
        r["ramp"] = [els[0].position, els[-1].position]
        tex = next((n for n in nt.nodes if n.type in ("TEX_NOISE", "TEX_WAVE", "TEX_VORONOI")), None)
        r["tex"] = {"TEX_NOISE": "noise", "TEX_WAVE": "wave", "TEX_VORONOI": "voronoi"}[tex.type]
        r["scale"] = tex.inputs["Scale"].default_value
        if tex.type == "TEX_WAVE":
            r["wave"] = tex.bands_direction
        mp = next((n for n in nt.nodes if n.type == "MAPPING"), None)
        r["stretch"] = list(mp.inputs["Scale"].default_value) if mp else [1, 1, 1]
    elif bsdf:
        r["c1"] = r["c2"] = list(bsdf.inputs["Base Color"].default_value[:3])
    else:  # shoji paper (diffuse + translucent)
        d = next(n for n in nt.nodes if n.type in ("BSDF_DIFFUSE",))
        r["c1"] = r["c2"] = list(d.inputs["Color"].default_value[:3])
    if bsdf:
        r["rough"] = bsdf.inputs["Roughness"].default_value
    return r


def ramp_colors(m):
    """Colour stops of a leaf material (colour picked per leaf from the 'leafcol' attribute)."""
    ramp = next(n for n in m.node_tree.nodes if n.type == "VALTORGB")
    pos = np.array([e.position for e in ramp.color_ramp.elements])
    col = np.array([e.color[:3] for e in ramp.color_ramp.elements])
    return pos, col


# ---------------------------------------------------------------- foliage -> instanced cards

LEAF_N = {"maple": 14, "ginkgo": 8, "ellipse": 8, "needle": 4}
FAR = ("cedar", "ginkgo", "outer_side", "side_tree")


def keep_fraction(name, mat):
    if name.startswith("fallen"):
        return 0.5   # half the leaves on the ground; the viewer adds leaves that fall and drift
    if name.startswith("maple"):
        return 0.5
    if name.startswith("pine"):
        return 0.6
    if name.startswith("azalea"):
        return 0.35
    return 0.3  # background trees behind/around the wall


# ---------------------------------------------------------------- the big red maple: out of the water, onto the bank
# build_garden.py roots it at (3.8, 8.6), inside the pond. Move it, crown and all, to the pond's right edge.
HERO_FROM, HERO_TO = (3.8, 8.6), (5.45, 8.75)
_mg = bpy.data.objects["moss_ground"]
_bm = bmesh.new()
_bm.from_mesh(_mg.data)
_bm.transform(_mg.matrix_world)
_tbvh = BVHTree.FromBMesh(_bm)
_z = lambda x, y: _tbvh.ray_cast(Vector((x, y, 5.0)), Vector((0, 0, -1)))[0].z
hero_shift = Vector((HERO_TO[0] - HERO_FROM[0], HERO_TO[1] - HERO_FROM[1], _z(*HERO_TO) - _z(*HERO_FROM)))
for ob in bpy.data.objects:
    if ob.name.startswith("maple_hero"):
        ob.matrix_world = Matrix.Translation(hero_shift) @ ob.matrix_world
print("hero maple moved by", tuple(round(v, 2) for v in hero_shift))

leaf_groups = {}
core_rgb = {}  # mean leaf colour per foliage blob, used to tint its dark inner core
for ob in list(bpy.data.objects):
    if ob.type != "MESH" or "leafcol" not in ob.data.attributes:
        continue
    me = ob.data
    mat = me.materials[0]
    V, F = len(me.vertices), len(me.polygons)
    n = F // (V - F)
    if n == 14:
        kind = "claude" if mat.name == "maple_orange" else "maple"   # easter egg: the orange maples grow Claude Code critters
    elif n == 4:
        kind = "needle"
    elif mat.name == "ginkgo":
        kind = "ginkgo"
    else:
        kind = "ellipse"
    co = np.empty(V * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mw = np.array(ob.matrix_world)
    co = co @ mw[:3, :3].T + mw[:3, 3]
    L = V // (n + 1)
    co = co.reshape(L, n + 1, 3)
    val = np.empty(F, np.float32)
    me.attributes["leafcol"].data.foreach_get("value", val)
    val = val.reshape(L, n)[:, 0]
    c = co[:, 0]
    d = co[:, 1:] - c[:, None]
    nrm = np.cross(d, np.roll(d, -1, 1)).sum(1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    inplane = d - (d * nrm[:, None]).sum(-1, keepdims=True) * nrm[:, None]
    size = np.linalg.norm(inplane, axis=-1).max(1)  # half-extent of the leaf outline
    tan = inplane[:, 0] / (np.linalg.norm(inplane[:, 0], axis=1, keepdims=True) + 1e-9)
    pos, cols = ramp_colors(mat)
    rgb = np.stack([np.interp(val, pos, cols[:, k]) for k in range(3)], -1)
    core_rgb[ob.name] = rgb.mean(0)
    keep = keep_fraction(ob.name, mat)
    sel = rng.random(L) < keep
    grow = 1.0 if ob.name.startswith("fallen") else 1.0 / math.sqrt(keep) * (1.12 if keep < 1 else 1.0)  # a little extra: alpha-tested cards lose coverage
    key = kind + ("_far" if ob.name.startswith(FAR) else "")
    g = leaf_groups.setdefault(key, {"pos": [], "nrm": [], "tan": [], "size": [], "rgb": []})
    g["pos"].append(c[sel]); g["nrm"].append(nrm[sel]); g["tan"].append(tan[sel])
    g["size"].append(size[sel] * grow); g["rgb"].append(rgb[sel])
    bpy.data.objects.remove(ob)

def tighten(pos, cell, min_nb, pull):
    """Drop leaves with few neighbours and pull the rest toward their local cluster, so crowns keep a clean edge."""
    vox = np.floor(pos / cell).astype(np.int64)
    keys, inv, cnt = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    sums = np.zeros((len(keys), 3))
    np.add.at(sums, inv, pos)
    lut = {tuple(k): i for i, k in enumerate(keys)}
    nb_cnt = np.zeros(len(keys))
    nb_sum = np.zeros((len(keys), 3))
    for d in np.stack(np.meshgrid(*[[-1, 0, 1]] * 3), -1).reshape(-1, 3):
        idx = np.array([lut.get(tuple(k + d), -1) for k in keys])
        ok = idx >= 0
        nb_cnt[ok] += cnt[idx[ok]]
        nb_sum[ok] += sums[idx[ok]]
    keep = nb_cnt[inv] >= min_nb
    centre = nb_sum[inv] / nb_cnt[inv, None]
    return keep, pos + (centre - pos) * pull


table, blobs, off = [], [], 0
for key, g in sorted(leaf_groups.items()):
    pos_b = np.concatenate(g["pos"])
    far = key.endswith("_far")
    keep, tight = tighten(pos_b, 0.6 if far else 0.25, 12 if far else 16, 0.25 if far else 0.3)
    # fallen leaves lie on the ground in a loose scatter: leave them as they are
    ground = pos_b[:, 2] < GROUND + 0.12
    keep |= ground
    # the old gravel rectangle is part moss now: lift leaves lying there onto whichever surface they're on
    for i in np.nonzero(ground & (pos_b[:, 1] > 1.5) & (pos_b[:, 1] < 5.45) & (np.abs(pos_b[:, 0]) < 16))[0]:
        pos_b[i, 2] = GROUND + (0.03 if gravel_sdf(pos_b[i, 0], pos_b[i, 1]) < 0 else 0.034)
    pos_b = np.where(ground[:, None], pos_b, tight)
    pos = to_yup(pos_b[keep]).astype(np.float32)
    nrm = to_yup(np.concatenate(g["nrm"])[keep])
    tan = to_yup(np.concatenate(g["tan"])[keep])
    size = np.concatenate(g["size"])[keep]
    rgb = np.concatenate(g["rgb"])[keep]
    cnt = len(pos)
    # 24-byte record: f32 xyz | i8 normal xyz | i8 tangent xyz | u16 size (0.1 mm) | u8 rgb (linear) | pad
    rec = np.zeros(cnt, dtype=[("p", "<f4", 3), ("n", "i1", 3), ("t", "i1", 3), ("s", "<u2"), ("c", "u1", 3),
                               ("pad", "u1")])
    rec["p"] = pos
    rec["n"] = np.clip(np.round(nrm * 127), -127, 127)
    rec["t"] = np.clip(np.round(tan * 127), -127, 127)
    rec["s"] = np.clip(np.round(size * 10000), 1, 65535)
    rec["c"] = np.clip(np.round(np.power(np.clip(rgb, 0, 1), 1 / 2.2) * 255), 0, 255)  # gamma-coded for precision
    blobs.append(rec.tobytes())
    table.append({"key": key, "kind": key.split("_")[0], "far": key.endswith("_far"), "count": cnt, "offset": off})
    off += rec.nbytes
    print(f"leaves {key}: {cnt}")
with open(os.path.join(args.out, "leaves.bin"), "wb") as f:
    f.write(b"".join(blobs))
with open(os.path.join(args.out, "leaves.json"), "w") as f:
    json.dump({"stride": 24, "groups": table}, f, indent=1)

# ---------------------------------------------------------------- gravel: organic bed, raking done in the shader
# The gravel is a coarse sheet over the outline's bounding box, cut to shape in the shader. The moss in front of
# the engawa is one sheet too, a few millimetres higher and cut out where the gravel is. Further back the gravel
# follows the moss terrain just under it.
gr = bpy.data.objects.get("raked_gravel")
gmat = gr.data.materials[0]
moss_mat = bpy.data.objects["moss_ground"].data.materials[0]
for name in ("raked_gravel", "moss_side_l", "moss_side_r"):
    bpy.data.objects.remove(bpy.data.objects[name])
mg = bpy.data.objects["moss_ground"]
mg_bm = bmesh.new()
mg_bm.from_mesh(mg.data)
mg_bm.transform(mg.matrix_world)
moss_bvh = BVHTree.FromBMesh(mg_bm)


def sheet(name, x0, y0, x1, y1, step, fn, mat):
    bm = bmesh.new()
    nx, ny = int(round((x1 - x0) / step)) + 1, int(round((y1 - y0) / step)) + 1
    vs = [[bm.verts.new((x0 + (x1 - x0) * i / (nx - 1), y0 + (y1 - y0) * j / (ny - 1), 0.0)) for i in range(nx)]
          for j in range(ny)]
    for row in vs:
        for v in row:
            v.co.z = fn(v.co.x, v.co.y)
    for j in range(ny - 1):
        for i in range(nx - 1):
            bm.faces.new((vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat)
    bpy.context.scene.collection.objects.link(bpy.data.objects.new(name, me))


def gravel_z(x, y):
    hit = moss_bvh.ray_cast(Vector((x, y, 5.0)), Vector((0, 0, -1)))[0]
    return GROUND + 0.015 if hit is None else min(GROUND + 0.015, hit.z - 0.004) if y < 5.6 else hit.z - 0.004


sheet("raked_gravel", -8.75, 1.5, 8.5, 9.25, 0.25, gravel_z, gmat)
# only a few millimetres above the gravel: a taller lip makes a depth step the painter outlines, and at a glancing
# angle that outline breaks into flickering dashes
sheet("moss_front", -16.0, 1.5, 16.0, 5.36, 0.12, lambda x, y: GROUND + 0.019, moss_mat)

# the bamboo spout and stone basin move from the gravel to the pond as the shishi-odoshi (built by the viewer)
for ob in list(bpy.data.objects):
    if ob.type != "MESH" or not ob.name.startswith(("Cylinder", "Sphere", "tsukubai")):
        continue
    c = ob.matrix_world @ (sum((Vector(b) for b in ob.bound_box), Vector()) / 8)
    if math.hypot(c.x - 4.4, c.y - 2.4) < 0.9:
        bpy.data.objects.remove(ob)
# and the pond-edge rock where it pours into the water
if bpy.data.objects.get("pond_rock.020"):
    bpy.data.objects.remove(bpy.data.objects["pond_rock.020"])

# ---------------------------------------------------------------- eave fix
# build_garden.py rotates the eave boards and rafters about the world origin (and the wrong way), which lifts
# their outer ends and pulls them 25 cm short of the fascia: sky shows through. Re-pivot about their own centres.
EAVE, SLOPE_HI, SLOPE_LO = 2.6, 2.45, 2.18
ang = math.atan2(SLOPE_HI - SLOPE_LO, EAVE)
for name, cz in (("eave_boards", (SLOPE_HI + SLOPE_LO) / 2 + 0.06), ("rafters", (SLOPE_HI + SLOPE_LO) / 2)):
    ob = bpy.data.objects.get(name)
    if ob:
        c = Vector((0.0, EAVE / 2, cz))
        rot = Matrix.Rotation(-ang, 3, "X")
        ob.rotation_euler = (-ang, 0.0, 0.0)
        ob.location = c - rot @ c
# close the slot between the fascia's tile edge and the roof's underside, seen from the garden
tile_mat = bpy.data.materials.get("roof_tile")
bm = bmesh.new()
r = bmesh.ops.create_cube(bm, size=1.0)
bmesh.ops.scale(bm, vec=(14.6, 0.3, 0.62), verts=r["verts"])
bmesh.ops.translate(bm, vec=(0.0, 2.6, 2.6), verts=r["verts"])
me = bpy.data.meshes.new("eave_fill")
bm.to_mesh(me)
me.materials.append(tile_mat)
bpy.context.scene.collection.objects.link(bpy.data.objects.new("eave_fill", me))

# ---------------------------------------------------------------- bamboo fences: one flat panel each, slats drawn in the shader
# (hundreds of 4 cm slats alias into moire at a distance; a procedural stripe can fade to its average instead)
WALL_Y = 15.5
fence_src = [ob for ob in bpy.data.objects if ob.name.startswith("bamboo_fence")]
if fence_src:
    bmat = fence_src[0].data.materials[0]
    fmat = bpy.data.materials.new("bamboo_fence")
    fmat.use_nodes = True
    for ob in fence_src:
        side = 1 if ob.location.x > 0 else -1
        bpy.data.objects.remove(ob)
        bm = bmesh.new()
        r = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(0.035, WALL_Y - 0.6, 1.9), verts=r["verts"])
        bmesh.ops.translate(bm, vec=(side * 15.5, (WALL_Y + 0.6) / 2, GROUND + 0.85), verts=r["verts"])
        me = bpy.data.meshes.new("bamboo_fence_panel")
        bm.to_mesh(me)
        me.materials.append(fmat)
        bpy.context.scene.collection.objects.link(bpy.data.objects.new("bamboo_fence_panel", me))
        bm = bmesh.new()
        for z in (0.4, 1.0, 1.6):  # the three binding rails, on the garden side
            r = bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, vec=(0.03, WALL_Y - 0.6, 0.04), verts=r["verts"])
            bmesh.ops.translate(bm, vec=(side * 15.5 - side * 0.035, (WALL_Y + 0.6) / 2, GROUND + z), verts=r["verts"])
        me = bpy.data.meshes.new("bamboo_fence_rails")
        bm.to_mesh(me)
        me.materials.append(bmat)
        bpy.context.scene.collection.objects.link(bpy.data.objects.new("bamboo_fence_rails", me))

# ---------------------------------------------------------------- merge static meshes per material
for ob in list(bpy.data.objects):
    if ob.type != "MESH":
        bpy.data.objects.remove(ob)
dg = bpy.context.evaluated_depsgraph_get()
merged = {}
for ob in bpy.data.objects:
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    mat = ob.data.materials[0] if ob.data.materials else None
    mname = mat.name if mat else "default"
    if mname.endswith("_core"):
        mname = "foliage_core"
    mw = np.array(ob.matrix_world)
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    nt = len(me.loop_triangles)
    tri = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    ln = np.empty(nt * 9, np.float32)
    me.loop_triangles.foreach_get("split_normals", ln)
    ln = ln.reshape(-1, 3, 3)
    nm = np.linalg.inv(mw[:3, :3]).T
    ln = ln @ nm.T
    ln /= np.linalg.norm(ln, axis=-1, keepdims=True) + 1e-9
    m = merged.setdefault(mname, {"p": [], "n": [], "c": [], "mat": mat})
    m["p"].append(co[tri].reshape(-1, 3))
    m["n"].append(ln.reshape(-1, 3))
    tint = core_rgb.get(ob.name[:-5], np.full(3, 0.02)) * 0.7 if mname == "foliage_core" else np.ones(3)
    m["c"].append(np.tile(tint, (nt * 3, 1)))
    ev.to_mesh_clear()

walk_names = {"engawa_floor", "tatami", "tatami_border", "moss", "gravel", "stone", "stone_mossy", "soil", "water",
              "zabuton", "wood_dark"}
walk_v, walk_f = [], []
recipes = {}
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)
for mname, m in merged.items():
    p = np.concatenate(m["p"]).astype(np.float64)
    nr = np.concatenate(m["n"])
    cc = np.concatenate(m["c"])
    # weld identical (position, normal) corners so the glb isn't 3 verts per triangle
    key = np.round(np.concatenate([p * 2000, nr * 500], 1)).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    first = np.zeros(len(uniq), np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    up, un = p[first], nr[first]
    faces = inv.reshape(-1, 3)
    me = bpy.data.meshes.new(mname)
    me.from_pydata(up.tolist(), [], faces.tolist())
    me.normals_split_custom_set_from_vertices(un.tolist())
    if mname == "foliage_core":
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        ca.data.foreach_set("color", np.concatenate([cc[first], np.ones((len(first), 1))], 1).ravel())
        me.color_attributes.active_color = ca
    mat = m["mat"]
    if mname == "foliage_core":
        mat = bpy.data.materials.new("foliage_core")
        recipes[mname] = {"name": mname, "c1": [1, 1, 1], "c2": [1, 1, 1], "vertexColor": True}
    elif mname == "bamboo_fence":
        recipes[mname] = dict(recipes.get("bamboo") or recipe(bpy.data.materials["bamboo"]), name=mname, tex="slats")
    elif mat:
        recipes[mname] = recipe(mat)
    # plain stand-in material: the viewer swaps in its own shader by name
    if mat:
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (*recipes[mname]["c1"], 1)
        nt.links.new(bsdf.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs[0])
        me.materials.append(mat)
    ob = bpy.data.objects.new(mname, me)
    bpy.context.scene.collection.objects.link(ob)
    if mname in walk_names:
        walk_v.append(up)
        walk_f.append(faces + sum(len(v) for v in walk_v[:-1]))
    print(f"mesh {mname}: {len(faces)} tris")

with open(os.path.join(args.out, "materials.json"), "w") as f:
    json.dump(recipes, f, indent=1)

bpy.ops.export_scene.gltf(filepath=os.path.join(args.out, "garden.glb"), export_format="GLB", export_yup=True,
                          export_texcoords=False, export_normals=True, export_cameras=False, export_lights=False,
                          export_materials="EXPORT", export_vertex_color="ACTIVE")

# ---------------------------------------------------------------- heightmap of walkable surfaces
wv = np.concatenate(walk_v)
wf = np.concatenate(walk_f)
bvh = BVHTree.FromPolygons([Vector(v) for v in wv], wf.tolist())
HX0, HX1, HY0, HY1, STEP = -16.0, 16.0, -6.0, 17.0, 0.1
nx, ny = int(round((HX1 - HX0) / STEP)) + 1, int(round((HY1 - HY0) / STEP)) + 1
hm = np.full((ny, nx), GROUND, np.float32)
for j in range(ny):
    for i in range(nx):
        hit = bvh.ray_cast(Vector((HX0 + i * STEP, HY0 + j * STEP, 1.2)), Vector((0, 0, -1)))
        if hit[0] is not None:
            hm[j, i] = hit[0].z
with open(os.path.join(args.out, "height.bin"), "wb") as f:
    f.write(np.round((hm + 2.0) * 1000).astype("<u2").tobytes())
with open(os.path.join(args.out, "height.json"), "w") as f:
    json.dump({"x0": HX0, "y0": HY0, "step": STEP, "nx": nx, "ny": ny, "note": "blender x/y, value=z+2 in mm"}, f)
print("done")
