"""Japanese fall garden viewed from an engawa (veranda), rendered as a 360 equirectangular panorama.

Run headless:  python3 build_garden.py --res 4096 --samples 256 --out render/garden_pano.png [--blend garden.blend]

Coordinates (meters): engawa floor top at z=0, viewer at origin facing +Y (garden),
tatami room behind (-Y), shoji panels slid open to the left/right of the viewer.
"""
import argparse
import math
import random
import sys

import bpy  # must come first: it makes bmesh/mathutils importable
import bmesh
from mathutils import Euler, Matrix, Vector, noise

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=1024)
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--out", default="garden_pano.png")
ap.add_argument("--blend", default="")
ap.add_argument("--persp", action="store_true", help="debug: 90 deg perspective render toward the garden")
ap.add_argument("--exr", action="store_true", help="write multilayer EXR with depth/normal/object-index passes")
args = ap.parse_args(argv)

random.seed(7)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
col = scene.collection

GROUND = -0.55  # garden ground level below the engawa floor

# ---------------------------------------------------------------- helpers

def link(obj):
    col.objects.link(obj)
    return obj


def mesh_obj(name, bm, mat=None, smooth=False):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    if mat:
        ob.data.materials.append(mat)
    return link(ob)


def box(name, center, size, mat, rot=(0, 0, 0)):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    ob = mesh_obj(name, bm, mat)
    ob.location = center
    ob.rotation_euler = rot
    return ob


def add_box(bm, center, size, rot_z=0.0):
    """Append an axis-aligned (optionally z-rotated) box into an existing bmesh."""
    res = bmesh.ops.create_cube(bm, size=1.0)
    vs = res["verts"]
    bmesh.ops.scale(bm, vec=Vector(size), verts=vs)
    if rot_z:
        bmesh.ops.rotate(bm, verts=vs, cent=(0, 0, 0), matrix=Matrix.Rotation(rot_z, 3, "Z"))
    bmesh.ops.translate(bm, vec=Vector(center), verts=vs)


def nodes_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return m, nt, out


def principled(name, color, rough=0.5, spec=0.5, **kw):
    m, nt, out = nodes_mat(name)
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Specular IOR Level"].default_value = spec
    for k, v in kw.items():
        p.inputs[k].default_value = v
    nt.links.new(p.outputs[0], out.inputs[0])
    return m


def textured(name, c1, c2, rough=0.6, scale=5.0, detail=6.0, bump=0.3, coord="Object",
             wave=None, voronoi=False, ramp_pos=(0.35, 0.65), extra=None, stretch=(1, 1, 1)):
    """Principled material whose colour mixes c1/c2 by noise (or wave) and gets matching bump."""
    m, nt, out = nodes_mat(name)
    L = nt.links.new
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    L(tc.outputs[coord], mp.inputs[0])
    mp.inputs["Scale"].default_value = stretch
    if wave:
        tx = nt.nodes.new("ShaderNodeTexWave")
        tx.wave_type = "BANDS"
        tx.bands_direction = wave
        tx.inputs["Scale"].default_value = scale
        tx.inputs["Distortion"].default_value = 4.0
        tx.inputs["Detail"].default_value = detail
        fac = tx.outputs["Fac"]
    elif voronoi:
        tx = nt.nodes.new("ShaderNodeTexVoronoi")
        tx.inputs["Scale"].default_value = scale
        fac = tx.outputs["Distance"]
    else:
        tx = nt.nodes.new("ShaderNodeTexNoise")
        tx.inputs["Scale"].default_value = scale
        tx.inputs["Detail"].default_value = detail
        fac = tx.outputs["Fac"]
    L(mp.outputs[0], tx.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = ramp_pos[0]
    ramp.color_ramp.elements[0].color = (*c1, 1)
    ramp.color_ramp.elements[1].position = ramp_pos[1]
    ramp.color_ramp.elements[1].color = (*c2, 1)
    L(fac, ramp.inputs[0])
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    p.inputs["Roughness"].default_value = rough
    L(ramp.outputs[0], p.inputs["Base Color"])
    if bump:
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        L(fac, b.inputs["Height"])
        L(b.outputs[0], p.inputs["Normal"])
    if extra:
        extra(nt, p)
    L(p.outputs[0], out.inputs[0])
    return m


def leaf_material(name, colors):
    """Thin leaf: diffuse+translucent, colour picked per face from the 'leafcol' attribute."""
    m, nt, out = nodes_mat(name)
    L = nt.links.new
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "leafcol"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "LINEAR"
    for i, c in enumerate(colors):
        pos = i / (len(colors) - 1)
        if i < 2:
            e = cr.elements[i]
            e.position = pos
        else:
            e = cr.elements.new(pos)
        e.color = (*c, 1)
    L(at.outputs["Fac"], ramp.inputs[0])
    d = nt.nodes.new("ShaderNodeBsdfPrincipled")
    d.inputs["Roughness"].default_value = 0.55
    d.inputs["Specular IOR Level"].default_value = 0.35
    L(ramp.outputs[0], d.inputs["Base Color"])
    t = nt.nodes.new("ShaderNodeBsdfTranslucent")
    L(ramp.outputs[0], t.inputs["Color"])
    mx = nt.nodes.new("ShaderNodeMixShader")
    mx.inputs[0].default_value = 0.42
    L(d.outputs[0], mx.inputs[1])
    L(t.outputs[0], mx.inputs[2])
    L(mx.outputs[0], out.inputs[0])
    return m


def rock(name, loc, scale, mat, seed, flat=0.6):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=1.0)
    off = Vector((seed * 3.1, seed * 1.7, seed * 2.3))
    for v in bm.verts:
        n = v.co.normalized()
        d = 1.0 + 0.35 * noise.noise(n * 1.3 + off) + 0.1 * noise.noise(n * 4 + off) + 0.03 * noise.noise(n * 12 + off)
        v.co = n * d
        if v.co.z < -0.2:
            v.co.z = -0.2 - (v.co.z + 0.2) * 0.3
    ob = mesh_obj(name, bm, mat, smooth=True)
    ob.scale = (scale[0], scale[1], scale[2] * flat)
    ob.location = loc
    ob.rotation_euler = (0, 0, seed)
    return ob


# ---------------------------------------------------------------- materials

M = {}
M["wood"] = textured("wood", (0.17, 0.095, 0.05), (0.30, 0.18, 0.09), rough=0.45, scale=4.0,
                     detail=8, bump=0.05, stretch=(1, 1, 12))
M["wood_dark"] = textured("wood_dark", (0.06, 0.035, 0.022), (0.12, 0.07, 0.04), rough=0.5,
                          scale=4.0, detail=8, bump=0.05, stretch=(1, 1, 12))
M["floor"] = textured("engawa_floor", (0.14, 0.075, 0.038), (0.27, 0.15, 0.075), rough=0.3,
                      scale=3.0, detail=10, bump=0.03, stretch=(0.25, 14, 1))
M["cedar"] = textured("cedar_ceiling", (0.30, 0.18, 0.10), (0.46, 0.30, 0.16), rough=0.6,
                      scale=3.0, detail=10, bump=0.02, stretch=(16, 0.3, 1))
M["plaster"] = textured("plaster", (0.55, 0.43, 0.28), (0.64, 0.52, 0.36), rough=0.9, scale=8,
                        bump=0.15)
M["plaster_white"] = textured("plaster_white", (0.78, 0.75, 0.68), (0.86, 0.84, 0.78), rough=0.9,
                              scale=8, bump=0.1)
M["tatami"] = textured("tatami", (0.42, 0.40, 0.20), (0.56, 0.52, 0.28), rough=0.7, scale=90,
                       wave="X", detail=1, bump=0.12, ramp_pos=(0.2, 0.8))
M["heri"] = principled("tatami_border", (0.02, 0.025, 0.03), rough=0.6)
M["tile"] = textured("roof_tile", (0.08, 0.085, 0.09), (0.14, 0.145, 0.15), rough=0.35, scale=6,
                     bump=0.1)
M["stone"] = textured("stone", (0.11, 0.105, 0.10), (0.27, 0.26, 0.24), rough=0.8, scale=6.5,
                      detail=12, bump=1.0)
M["stone_moss"] = textured("stone_mossy", (0.07, 0.10, 0.03), (0.24, 0.235, 0.21), rough=0.85,
                           scale=4.0, detail=12, bump=1.0, ramp_pos=(0.42, 0.58))
M["lantern"] = textured("lantern_stone", (0.24, 0.24, 0.22), (0.42, 0.41, 0.38), rough=0.85,
                        scale=7, detail=12, bump=0.8)
M["moss"] = textured("moss", (0.06, 0.11, 0.025), (0.18, 0.24, 0.05), rough=0.95, scale=3.5,
                     detail=12, bump=0.8)
M["gravel"] = textured("gravel", (0.33, 0.315, 0.28), (0.50, 0.48, 0.43), rough=0.9, scale=180,
                       voronoi=True, bump=0.5)
M["soil"] = textured("soil", (0.10, 0.07, 0.045), (0.18, 0.13, 0.08), rough=0.95, scale=6,
                     bump=0.4)
M["water"] = principled("water", (0.75, 0.85, 0.8), rough=0.03, spec=0.5,
                        **{"Transmission Weight": 1.0, "IOR": 1.33})
M["pond_bed"] = textured("pond_bed", (0.02, 0.03, 0.02), (0.07, 0.07, 0.04), rough=0.9, scale=4)
M["koi_o"] = principled("koi_orange", (0.9, 0.25, 0.03), rough=0.3)
M["koi_w"] = principled("koi_white", (0.85, 0.82, 0.78), rough=0.3)
M["bark"] = textured("bark", (0.06, 0.05, 0.04), (0.16, 0.14, 0.11), rough=0.9, scale=12,
                     detail=8, bump=0.6)
M["bark_red"] = textured("bark_pine", (0.12, 0.06, 0.04), (0.25, 0.14, 0.08), rough=0.9, scale=14,
                         voronoi=True, bump=0.8)
M["paper"] = None  # built below
M["zabuton"] = textured("zabuton", (0.02, 0.03, 0.08), (0.03, 0.045, 0.11), rough=0.9, scale=30,
                        bump=0.05)
M["ceramic"] = principled("tea_ceramic", (0.22, 0.30, 0.24), rough=0.15, **{"Coat Weight": 0.6})
M["ceramic_dark"] = principled("teapot", (0.05, 0.04, 0.035), rough=0.35)
M["tea"] = principled("tea", (0.35, 0.4, 0.08), rough=0.05)
M["tray"] = principled("lacquer_tray", (0.15, 0.015, 0.01), rough=0.2, **{"Coat Weight": 1.0})
M["bamboo"] = textured("bamboo", (0.45, 0.38, 0.18), (0.55, 0.48, 0.26), rough=0.4, scale=40,
                       wave="Z", bump=0.05)
M["rope"] = principled("rope", (0.05, 0.03, 0.02), rough=0.9)
M["scroll"] = textured("scroll", (0.80, 0.77, 0.66), (0.86, 0.83, 0.74), rough=0.8, scale=20)
M["ink"] = principled("ink", (0.015, 0.015, 0.02), rough=0.7)
M["fusuma"] = textured("fusuma", (0.72, 0.66, 0.50), (0.80, 0.74, 0.58), rough=0.85, scale=15,
                       bump=0.05)
M["hill"] = textured("far_hills", (0.20, 0.17, 0.10), (0.33, 0.20, 0.10), rough=1.0, scale=3,
                     bump=0.0)

# shoji paper: translucent so the morning light glows through
m, nt, out = nodes_mat("shoji_paper")
d = nt.nodes.new("ShaderNodeBsdfDiffuse")
d.inputs["Color"].default_value = (0.85, 0.83, 0.77, 1)
t = nt.nodes.new("ShaderNodeBsdfTranslucent")
t.inputs["Color"].default_value = (0.9, 0.85, 0.72, 1)
mx = nt.nodes.new("ShaderNodeMixShader")
mx.inputs[0].default_value = 0.45
nt.links.new(d.outputs[0], mx.inputs[1])
nt.links.new(t.outputs[0], mx.inputs[2])
nt.links.new(mx.outputs[0], out.inputs[0])
M["paper"] = m

MAPLE_RED = leaf_material("maple_red", [(0.35, 0.012, 0.008), (0.62, 0.03, 0.01), (0.80, 0.10, 0.012),
                                        (0.85, 0.28, 0.02)])
MAPLE_ORANGE = leaf_material("maple_orange", [(0.55, 0.06, 0.01), (0.80, 0.22, 0.02), (0.90, 0.42, 0.04),
                                              (0.75, 0.55, 0.08)])
GINKGO = leaf_material("ginkgo", [(0.70, 0.45, 0.02), (0.88, 0.62, 0.05), (0.92, 0.74, 0.12),
                                  (0.55, 0.55, 0.10)])
PINE = leaf_material("pine_needles", [(0.015, 0.04, 0.012), (0.03, 0.07, 0.02), (0.05, 0.09, 0.025)])
EVERGREEN = leaf_material("evergreen", [(0.012, 0.03, 0.01), (0.03, 0.06, 0.018), (0.06, 0.08, 0.02)])
SHRUB = leaf_material("azalea", [(0.02, 0.05, 0.015), (0.04, 0.08, 0.02), (0.25, 0.05, 0.02)])
FALLEN = leaf_material("fallen", [(0.25, 0.03, 0.01), (0.55, 0.08, 0.015), (0.70, 0.25, 0.03),
                                  (0.30, 0.15, 0.05)])

# ---------------------------------------------------------------- architecture
ROOM_W = 2.7      # half width of the tatami room / shoji opening
ROOM_D = 4.6      # room depth (behind viewer)
ENG_D = 1.25      # engawa depth
EAVE = 2.6        # eave overhang from the facade
H_KAMOI = 1.80    # lintel height
H_CEIL = 2.45

# engawa floor boards (run along X)
bm = bmesh.new()
bw = 0.15
y = 0.0
i = 0
while y < ENG_D - 1e-3:
    w = min(bw, ENG_D - y)
    add_box(bm, (0, y + w / 2, -0.03), (14.0 + random.uniform(-0.002, 0.002), w - 0.004, 0.06))
    y += w
    i += 1
mesh_obj("engawa_boards", bm, M["floor"])
box("engawa_base", (0, ENG_D / 2, -0.3), (14.0, ENG_D, 0.5), M["wood_dark"])
box("engawa_edge", (0, ENG_D - 0.04, -0.06), (14.0, 0.09, 0.12), M["wood"])
for x in (-6.9, -4.6, -2.3, 0.0, 2.3, 4.6, 6.9):
    if abs(x) > 0.1:  # keep the view clear in front
        box("engawa_post_low", (x, ENG_D - 0.15, (GROUND) / 2 - 0.05), (0.12, 0.12, -GROUND), M["wood_dark"])
    # foundation stones under short posts
    rock("found_stone", (x, ENG_D - 0.15, GROUND + 0.02), (0.13, 0.13, 0.12), M["stone"], x + 3)

# eave posts (hashira) on the garden edge, framing the view
for x in (-ROOM_W - 0.05, ROOM_W + 0.05, -6.9, 6.9):
    box("eave_post", (x, ENG_D - 0.15, 1.15), (0.13, 0.13, 2.4), M["wood"])
# beam along posts
box("eave_beam", (0, ENG_D - 0.15, 2.32), (14.2, 0.14, 0.18), M["wood"])

# engawa ceiling: rafters (taruki) and boards, sloping from facade to eave
slope_start, slope_end = 2.45, 2.18
ang = math.atan2(slope_start - slope_end, EAVE)
bm = bmesh.new()
add_box(bm, (0, EAVE / 2, (slope_start + slope_end) / 2 + 0.06), (14.4, EAVE / math.cos(ang), 0.025))
me_board = mesh_obj("eave_boards", bm, M["cedar"])
me_board.rotation_euler = (ang, 0, 0)
me_board.location = (0, 0, 0)
bm = bmesh.new()
x = -7.1
while x <= 7.15:
    add_box(bm, (x, EAVE / 2, (slope_start + slope_end) / 2), (0.045, EAVE / math.cos(ang) + 0.05, 0.06))
    x += 0.36
raf = mesh_obj("rafters", bm, M["wood"])
raf.rotation_euler = (ang, 0, 0)
box("fascia", (0, EAVE + 0.02, slope_end + 0.02), (14.5, 0.04, 0.16), M["wood_dark"])
# roof tiles above (thick wedge) so no sky leaks through
roof = box("roof", (0, EAVE / 2 - 0.5, slope_start + 1.4), (15.0, EAVE + 1.2, 0.8), M["tile"])
roof.rotation_euler = (math.radians(20), 0, 0)
# tile row edge visible from below/garden
bm = bmesh.new()
x = -7.2
while x <= 7.25:
    add_box(bm, (x, EAVE + 0.06, slope_end + 0.16), (0.24, 0.12, 0.08))
    x += 0.27
mesh_obj("tile_edge", bm, M["tile"])

# facade beyond the shoji opening: plaster walls with wooden frame + closed shoji
for s in (-1, 1):
    x0 = s * (ROOM_W + 0.1)
    x1 = s * 7.2
    cx, w = (x0 + x1) / 2, abs(x1 - x0)
    box("facade_lower", (cx, -0.06, 0.12), (w, 0.12, 0.24), M["wood_dark"])
    box("facade_upper", (cx, -0.06, (H_KAMOI + H_CEIL) / 2 + 0.05), (w, 0.12, H_CEIL - H_KAMOI + 0.1),
        M["plaster_white"])
    box("facade_kamoi", (cx, -0.02, H_KAMOI + 0.03), (w, 0.1, 0.07), M["wood"])

# room structure
box("room_floor_base", (0, -ROOM_D / 2, -0.05), (2 * ROOM_W + 0.2, ROOM_D, 0.1), M["wood_dark"])
box("room_ceiling", (0, -ROOM_D / 2, H_CEIL + 0.02), (2 * ROOM_W + 0.4, ROOM_D + 0.2, 0.04), M["cedar"])
bm = bmesh.new()
y = -ROOM_D
while y < 0:
    add_box(bm, (0, y, H_CEIL - 0.015), (2 * ROOM_W, 0.035, 0.03))  # ceiling battens (saobuchi)
    y += 0.45
mesh_obj("ceiling_battens", bm, M["wood"])

# front sill + lintel (shikii / kamoi) with shoji grooves
box("shikii", (0, 0.0, 0.015), (2 * ROOM_W, 0.11, 0.03), M["wood"])
box("kamoi", (0, 0.0, H_KAMOI + 0.03), (2 * ROOM_W + 0.2, 0.11, 0.07), M["wood"])
box("ranma_wall", (0, -0.02, (H_KAMOI + H_CEIL) / 2 + 0.05), (2 * ROOM_W + 0.2, 0.08, H_CEIL - H_KAMOI - 0.06),
    M["plaster_white"])
# corner pillars of the room
for x in (-ROOM_W, ROOM_W):
    box("room_pillar", (x, 0.0, H_CEIL / 2), (0.12, 0.12, H_CEIL), M["wood"])
    box("room_pillar_b", (x, -ROOM_D, H_CEIL / 2), (0.12, 0.12, H_CEIL), M["wood"])

# side walls: fusuma sliding doors under a plaster band
for s in (-1, 1):
    x = s * (ROOM_W + 0.03)
    box("side_wall_upper", (x, -ROOM_D / 2, (H_KAMOI + H_CEIL) / 2 + 0.03), (0.06, ROOM_D, H_CEIL - H_KAMOI),
        M["plaster_white"])
    box("side_kamoi", (s * (ROOM_W - 0.02), -ROOM_D / 2, H_KAMOI + 0.03), (0.06, ROOM_D, 0.07), M["wood"])
    n = 4
    pw = ROOM_D / n
    for k in range(n):
        yc = -pw / 2 - k * pw
        box("fusuma", (x, yc, H_KAMOI / 2), (0.03, pw - 0.01, H_KAMOI), M["fusuma"])
        # lacquered frame
        bm = bmesh.new()
        add_box(bm, (0, -pw / 2 + 0.015, 0), (0.035, 0.03, H_KAMOI))
        add_box(bm, (0, pw / 2 - 0.015, 0), (0.035, 0.03, H_KAMOI))
        add_box(bm, (0, 0, H_KAMOI / 2 - 0.015), (0.035, pw, 0.03))
        add_box(bm, (0, 0, -H_KAMOI / 2 + 0.015), (0.035, pw, 0.03))
        fr = mesh_obj("fusuma_frame", bm, M["ceramic_dark"])
        fr.location = (x - s * 0.005, yc, H_KAMOI / 2)
        # round recessed pull
        h = box("fusuma_pull", (x - s * 0.02, yc + (pw / 2 - 0.12) * (1 if k % 2 else -1), 0.85),
                (0.01, 0.05, 0.08), M["ceramic_dark"])

# back wall with tokonoma alcove (left) and shelves (right)
box("back_wall_r", (ROOM_W / 2 + 0.3, -ROOM_D - 0.03, H_CEIL / 2), (ROOM_W - 0.6, 0.06, H_CEIL), M["plaster"])
TK_X0, TK_X1 = -ROOM_W + 0.05, 0.6
box("toko_back", ((TK_X0 + TK_X1) / 2, -ROOM_D - 0.75, H_CEIL / 2), (TK_X1 - TK_X0, 0.06, H_CEIL), M["plaster"])
box("toko_side_l", (TK_X0 - 0.02, -ROOM_D - 0.38, H_CEIL / 2), (0.05, 0.75, H_CEIL), M["plaster"])
box("toko_side_r", (TK_X1 + 0.02, -ROOM_D - 0.38, H_CEIL / 2), (0.05, 0.75, H_CEIL), M["plaster"])
box("toko_ceiling", ((TK_X0 + TK_X1) / 2, -ROOM_D - 0.38, H_CEIL - 0.1), (TK_X1 - TK_X0, 0.75, 0.03), M["cedar"])
box("toko_otoshi", ((TK_X0 + TK_X1) / 2, -ROOM_D, 2.0), (TK_X1 - TK_X0, 0.07, 0.08), M["wood"])
box("toko_otoshi_wall", ((TK_X0 + TK_X1) / 2, -ROOM_D - 0.01, (2.04 + H_CEIL) / 2), (TK_X1 - TK_X0, 0.05, H_CEIL - 2.04),
    M["plaster"])
box("toko_floor", ((TK_X0 + TK_X1) / 2, -ROOM_D - 0.38, 0.09), (TK_X1 - TK_X0, 0.75, 0.12), M["floor"])
box("tokobashira", (TK_X1, -ROOM_D, H_CEIL / 2), (0.11, 0.11, H_CEIL), M["bark_red"])
# hanging scroll
box("scroll", (-1.05, -ROOM_D - 0.71, 1.25), (0.5, 0.01, 1.35), M["scroll"])
box("scroll_rod_t", (-1.05, -ROOM_D - 0.70, 1.93), (0.56, 0.02, 0.02), M["wood_dark"])
box("scroll_rod_b", (-1.05, -ROOM_D - 0.70, 0.57), (0.6, 0.03, 0.03), M["wood_dark"])
for k, (yy, hh) in enumerate([(1.62, 0.22), (1.35, 0.18), (1.1, 0.24), (0.86, 0.15)]):
    box("calligraphy", (-1.05 + random.uniform(-0.03, 0.03), -ROOM_D - 0.70, yy),
        (0.06 + random.uniform(0, 0.05), 0.005, hh), M["ink"])
# ikebana vase with a maple twig
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.09, depth=0.28, location=(-0.3, -ROOM_D - 0.4, 0.29))
bpy.context.object.data.materials.append(M["ceramic"])
# shelves (chigaidana)
box("shelf1", (1.6, -ROOM_D + 0.18, 0.9), (1.4, 0.34, 0.03), M["wood"])
box("shelf2", (1.9, -ROOM_D + 0.18, 1.25), (0.9, 0.34, 0.03), M["wood"])
box("shelf_cab", (1.6, -ROOM_D + 0.18, 2.0), (1.4, 0.34, 0.3), M["fusuma"])

# tatami mats: 0.9 x 1.8 m, 8-mat layout filling the room
TH = 0.055
mats = []
for yy in (-0.9, -2.7):         # front row: four mats lengthwise across x? use classic layout
    pass
layout = [  # (cx, cy, rotated)
    (-1.8, -0.45, True), (0.0, -0.45, True), (1.8, -0.45, True),
    (-2.25, -1.8, False), (-1.35, -1.8, False), (-0.45, -1.8, False), (0.45, -1.8, False),
    (1.35, -1.8, False), (2.25, -1.8, False),
    (-1.8, -3.15, True), (0.0, -3.15, True), (1.8, -3.15, True),
    (-1.8, -4.05, True), (0.0, -4.05, True), (1.8, -4.05, True),
]
for cx, cy, rot in layout:
    if rot:   # long side along X, 1.8 x 0.9
        sx, sy = 1.8, 0.9
    else:
        sx, sy = 0.9, 1.8
    sy = min(sy, ROOM_D)
    t = box("tatami", (cx, cy, TH / 2), (sx - 0.006, sy - 0.006, TH), M["tatami"])
    if not rot:
        t.rotation_euler = (0, 0, 0)
    # black cloth borders along the long sides
    if rot:
        box("heri", (cx, cy + sy / 2 - 0.016, TH / 2 + 0.001), (sx - 0.006, 0.03, TH), M["heri"])
        box("heri", (cx, cy - sy / 2 + 0.016, TH / 2 + 0.001), (sx - 0.006, 0.03, TH), M["heri"])
    else:
        box("heri", (cx + sx / 2 - 0.016, cy, TH / 2 + 0.001), (0.03, sy - 0.006, TH), M["heri"])
        box("heri", (cx - sx / 2 + 0.016, cy, TH / 2 + 0.001), (0.03, sy - 0.006, TH), M["heri"])

# low table (chabudai) in the room
box("table_top", (0.4, -2.4, 0.33), (1.2, 0.75, 0.035), M["wood"])
for dx in (-0.5, 0.5):
    for dy in (-0.28, 0.28):
        box("table_leg", (0.4 + dx, -2.4 + dy, 0.165), (0.05, 0.05, 0.31), M["wood"])
box("zabuton_room", (0.4, -1.65, 0.09), (0.55, 0.6, 0.07), M["zabuton"])

# ---- shoji panels, slid open to both sides
def shoji(name, cx, cy, w=0.92, h=H_KAMOI - 0.03):
    bm = bmesh.new()
    fw = 0.03
    add_box(bm, (-w / 2 + fw / 2, 0, h / 2), (fw, 0.03, h))
    add_box(bm, (w / 2 - fw / 2, 0, h / 2), (fw, 0.03, h))
    add_box(bm, (0, 0, h - fw / 2), (w, 0.03, fw))
    add_box(bm, (0, 0, 0.2), (w, 0.03, 0.035))   # kick rail
    add_box(bm, (0, 0, 0.1), (w, 0.03, 0.2))     # koshi-ita (wood bottom)
    cols, rows = 4, 7
    for k in range(1, cols):
        x = -w / 2 + k * w / cols
        add_box(bm, (x, 0, (h + 0.2) / 2), (0.012, 0.022, h - 0.2))
    for k in range(1, rows):
        z = 0.2 + k * (h - 0.2) / rows
        add_box(bm, (0, 0, z), (w, 0.022, 0.012))
    fr = mesh_obj(name + "_frame", bm, M["wood"])
    fr.location = (cx, cy, 0.03)
    p = box(name + "_paper", (cx, cy - 0.006, 0.03 + (h + 0.2) / 2), (w - 0.04, 0.002, h - 0.2), M["paper"])

# open: left pair at x in [-2.7, -1.78], right pair mirrored, two tracks
for s in (-1, 1):
    shoji("shoji", s * (ROOM_W - 0.47), 0.018, )
    shoji("shoji", s * (ROOM_W - 0.47 - 0.25), -0.018)
# closed shoji further along the facade (left and right of the room)
for s in (-1, 1):
    for k in range(4):
        x = s * (ROOM_W + 0.6 + k * 0.92)
        if abs(x) < 7.1:
            shoji("shoji_far", x, 0.0)

# ---- viewer's spot: zabuton + tea tray
box("zabuton", (0, 0.4, 0.04), (0.52, 0.56, 0.08), M["zabuton"])
box("tray", (0.62, 0.55, 0.012), (0.36, 0.26, 0.024), M["tray"])
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.04, depth=0.07, location=(0.56, 0.6, 0.06))
bpy.context.object.data.materials.append(M["ceramic"])
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.034, depth=0.002, location=(0.56, 0.6, 0.085))
bpy.context.object.data.materials.append(M["tea"])
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.07, location=(0.70, 0.5, 0.08))
tp = bpy.context.object
tp.scale = (1, 1, 0.8)
tp.data.materials.append(M["ceramic_dark"])
for p in tp.data.polygons:
    p.use_smooth = True
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.01, depth=0.12, location=(0.62, 0.5, 0.1),
                                    rotation=(0, math.radians(-60), 0))
bpy.context.object.data.materials.append(M["ceramic_dark"])

# ---------------------------------------------------------------- garden terrain
POND = [(0.0, 7.6), (1.8, 6.6), (3.8, 6.9), (5.0, 8.1), (4.6, 9.6), (2.6, 10.3), (0.8, 9.9),
        (-0.6, 10.6), (-2.6, 10.2), (-3.4, 9.0), (-2.0, 8.0)]


def in_poly(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def poly_dist(x, y, poly):
    best = 1e9
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        t = max(0, min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
        px, py = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(x - px, y - py))
    return best


GRAVEL = (-7.5, 1.6, 7.5, 5.4)   # x0, y0, x1, y1 raked gravel band in front of the engawa
RIPPLE_ROCKS = [(-3.6, 3.4, 0.55), (3.2, 3.0, 0.45)]


def terrain_h(x, y):
    h = GROUND
    # gentle mounds toward the back
    h += 0.5 * max(0.0, noise.noise(Vector((x * 0.12, y * 0.12, 1.0)))) * min(1, max(0, (y - 6) / 4))
    h += 0.9 * math.exp(-((x + 7) ** 2 + (y - 12) ** 2) / 10) + 0.7 * math.exp(-((x - 8) ** 2 + (y - 11) ** 2) / 8)
    h += 0.04 * noise.noise(Vector((x * 0.8, y * 0.8, 3.0)))
    if in_poly(x, y, POND):
        dd = poly_dist(x, y, POND)
        h = GROUND - min(0.45, dd * 0.8) - 0.05
    return h


# moss/ground mesh (excluding the gravel band which is its own mesh)
def grid(name, x0, y0, x1, y1, step, fn, mat):
    nx, ny = int((x1 - x0) / step) + 1, int((y1 - y0) / step) + 1
    me = bpy.data.meshes.new(name)
    verts = [(x0 + i * step, y0 + j * step, fn(x0 + i * step, y0 + j * step)) for j in range(ny) for i in range(nx)]
    faces = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i)
             for j in range(ny - 1) for i in range(nx - 1)]
    me.from_pydata(verts, [], faces)
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(mat)
    return link(ob)


grid("moss_ground", -16, GRAVEL[3] - 0.05, 16, 17, 0.12, terrain_h, M["moss"])
# moss strips at the sides of the gravel band and under the engawa
grid("soil_under", -16, -1.0, 16, GRAVEL[1] + 0.02, 0.3, lambda x, y: GROUND - 0.01, M["soil"])
grid("moss_side_l", -16, GRAVEL[1], GRAVEL[0], GRAVEL[3], 0.12, lambda x, y: GROUND + 0.02
     + 0.03 * noise.noise(Vector((x, y, 0))), M["moss"])
grid("moss_side_r", GRAVEL[2], GRAVEL[1], 16, GRAVEL[3], 0.12, lambda x, y: GROUND + 0.02
     + 0.03 * noise.noise(Vector((x, y, 0))), M["moss"])


def gravel_h(x, y):
    h = GROUND + 0.015
    ripple = None
    for rx, ry, rr in RIPPLE_ROCKS:
        d = math.hypot(x - rx, y - ry)
        if d < rr + 0.7:
            ripple = d
    if ripple is not None:
        h += 0.012 * math.sin(ripple * 2 * math.pi / 0.09)
    else:
        h += 0.012 * math.sin((y + 0.04 * math.sin(x * 0.6)) * 2 * math.pi / 0.09)
    return h


grid("raked_gravel", GRAVEL[0], GRAVEL[1], GRAVEL[2], GRAVEL[3], 0.018, gravel_h, M["gravel"])
# stone edging along the gravel / under the eaves drip line
bm = bmesh.new()
x = GRAVEL[0]
while x < GRAVEL[2]:
    add_box(bm, (x + 0.1, GRAVEL[1] - 0.06, GROUND + 0.01), (0.19, 0.1, 0.07), random.uniform(-0.05, 0.05))
    x += 0.2
mesh_obj("drip_edge", bm, M["stone"])

# pond: water surface + bed is the terrain itself
bm = bmesh.new()
verts = [bm.verts.new((px, py, GROUND - 0.07)) for px, py in POND]
bm.faces.new(verts)
bmesh.ops.triangulate(bm, faces=bm.faces[:])
w = mesh_obj("pond_water", bm, M["water"])
w.scale = (1.0, 1.0, 1.0)
# pond bed material: separate dark mesh just under the moss terrain inside the pond
grid("pond_bed", -3.6, 6.4, 5.2, 10.8, 0.15,
     lambda x, y: (terrain_h(x, y) - 0.01) if in_poly(x, y, POND) else GROUND - 0.6, M["pond_bed"])

# koi
for k, (kx, ky, kr, mat) in enumerate([(1.6, 8.2, 0.4, "koi_o"), (2.3, 8.7, 2.1, "koi_w"), (0.4, 8.9, -1.0, "koi_o")]):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.1, location=(kx, ky, GROUND - 0.2))
    o = bpy.context.object
    o.scale = (2.4, 0.7, 0.55)
    o.rotation_euler = (0, 0, kr)
    o.data.materials.append(M[mat])

# rocks around the pond edge
for k, (px, py) in enumerate(POND):
    nx_, ny_ = POND[(k + 1) % len(POND)]
    for t in (0.0, 0.5):
        x = px + (nx_ - px) * t
        y = py + (ny_ - py) * t
        s = random.uniform(0.25, 0.5)
        rock("pond_rock", (x, y, GROUND - 0.05), (s, s * random.uniform(0.7, 1.1), s * 0.8),
             M["stone_moss"], k * 2 + t * 7)

# big arrangement stones
for k, (x, y, s, h) in enumerate([(-3.6, 3.4, 0.5, 0.9), (3.2, 3.0, 0.42, 0.7), (-5.2, 8.0, 0.8, 1.1),
                                 (6.0, 7.6, 0.7, 0.9), (2.0, 11.4, 0.9, 1.0), (-1.2, 11.2, 0.6, 0.8),
                                 (5.8, 10.8, 0.5, 0.7)]):
    rock("garden_rock", (x, y, GROUND - 0.05), (s, s * 0.8, s * h), M["stone_moss"] if y > 5 else M["stone"],
         k * 1.9 + 0.4, flat=0.9)

# shoe stone and stepping stones (tobi-ishi) toward the pond
rock("kutsunugi", (0.0, ENG_D + 0.55, GROUND + 0.05), (0.42, 0.32, 0.18), M["stone"], 2.2, flat=1.0)
path = [(0.35, 2.3), (-0.2, 2.95), (0.3, 3.6), (-0.15, 4.3), (0.5, 5.0), (1.2, 5.6), (1.9, 6.15)]
for k, (x, y) in enumerate(path):
    s = random.uniform(0.24, 0.3)
    rock("tobi_ishi", (x, y, GROUND + 0.0), (s, s * 0.85, 0.12), M["stone"], k * 3.3, flat=1.0)

# stone lantern (kasuga-style) by the pond
LX, LY = -3.7, 6.6
base = terrain_h(LX, LY)


def lantern_part(kind, z, r, h, verts=6):
    if kind == "cyl":
        bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=h, location=(LX, LY, base + z + h / 2))
    else:
        bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r, radius2=r * 0.15, depth=h,
                                        location=(LX, LY, base + z + h / 2))
    o = bpy.context.object
    o.data.materials.append(M["lantern"])
    bev = o.modifiers.new("bev", "BEVEL")
    bev.width = 0.01
    bev.segments = 2
    return o


lantern_part("cyl", 0.0, 0.36, 0.14)
lantern_part("cyl", 0.14, 0.12, 0.75, 16)
lantern_part("cyl", 0.89, 0.30, 0.12)
fire = lantern_part("cyl", 1.01, 0.22, 0.32)
lantern_part("cone", 1.33, 0.48, 0.3)
lantern_part("cone", 1.6, 0.12, 0.16, 12)
# light box openings: dark inset squares
for a in range(0, 360, 120):
    r = math.radians(a)
    box("lantern_window", (LX + 0.19 * math.cos(r), LY + 0.19 * math.sin(r), base + 1.17), (0.06, 0.16, 0.16),
        M["ink"], rot=(0, 0, r))

# bamboo water spout (kakehi) and stone basin (tsukubai) at right near the engawa
TX, TY = 4.4, 2.6
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.32, depth=0.42, location=(TX, TY, GROUND + 0.18))
bas = bpy.context.object
bas.data.materials.append(M["lantern"])
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.21, depth=0.02, location=(TX, TY, GROUND + 0.36))
bpy.context.object.data.materials.append(M["water"])
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.035, depth=1.0, location=(TX + 0.55, TY + 0.2, GROUND + 0.6),
                                    rotation=(0, 0, 0))
bpy.context.object.data.materials.append(M["bamboo"])
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.025, depth=0.6, location=(TX + 0.3, TY + 0.1, GROUND + 0.98),
                                    rotation=(0, math.radians(80), math.radians(20)))
bpy.context.object.data.materials.append(M["bamboo"])
bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.07, location=(TX - 0.25, TY - 0.55, GROUND + 0.05))
bpy.context.object.data.materials.append(M["stone"])
rock("tsukubai_front", (TX - 0.1, TY - 0.6, GROUND), (0.28, 0.22, 0.1), M["stone"], 5.5, flat=1.0)
rock("tsukubai_side", (TX - 0.45, TY + 0.1, GROUND), (0.22, 0.2, 0.25), M["stone_moss"], 6.5)

# boundary: earthen wall (tsuiji-bei) with tile cap
WALL_Y = 15.5
box("tsuiji_wall", (0, WALL_Y, GROUND + 1.1), (32, 0.45, 2.3), M["plaster"])
cap = box("wall_cap", (0, WALL_Y, GROUND + 2.35), (32.4, 0.9, 0.12), M["tile"])
ridge = box("wall_ridge", (0, WALL_Y, GROUND + 2.48), (32.4, 0.25, 0.16), M["tile"])
for s in (-1, 1):  # side bamboo fences
    bm = bmesh.new()
    y = 0.6
    while y < WALL_Y:
        add_box(bm, (0, y, 0.85), (0.04, 0.035, 1.9))
        y += 0.045
    for z in (0.4, 1.0, 1.6):
        add_box(bm, (s * -0.05, (WALL_Y + 0.6) / 2, z), (0.03, WALL_Y - 0.6, 0.04))
    f = mesh_obj("bamboo_fence", bm, M["bamboo"])
    f.location = (s * 15.5, 0, GROUND)

# far hills behind the garden wall
grid("far_hills", -120, 40, 120, 160, 4.0,
     lambda x, y: -2 + 26 * max(0, noise.noise(Vector((x * 0.012, y * 0.012, 5)))) + 18 * math.exp(-((y - 120) / 60) ** 2)
     + 6 * noise.noise(Vector((x * 0.05, y * 0.05, 8))), M["hill"])
grid("outer_ground", -60, 15, 60, 42, 2.0, lambda x, y: GROUND + 0.2, M["moss"])

# ---------------------------------------------------------------- trees

def tube(bm, a, b, r0, r1, sides=7):
    axis = (b - a)
    L = axis.length
    if L < 1e-5:
        return
    z = axis.normalized()
    x = z.orthogonal().normalized()
    y = z.cross(x)
    ring0, ring1 = [], []
    for i in range(sides):
        t = 2 * math.pi * i / sides
        d = x * math.cos(t) + y * math.sin(t)
        ring0.append(bm.verts.new(a + d * r0))
        ring1.append(bm.verts.new(b + d * r1))
    for i in range(sides):
        j = (i + 1) % sides
        bm.faces.new((ring0[i], ring0[j], ring1[j], ring1[i]))


def leaf_shape(kind):
    """Return 2D outline (list of (u,v)) of a leaf, centred, size ~1."""
    pts = []
    if kind == "maple":  # 7-lobed palmate
        n = 14
        for i in range(n):
            t = 2 * math.pi * i / n + math.pi / 2
            r = 0.55 if i % 2 == 0 else 0.22
            pts.append((r * math.cos(t), r * math.sin(t) * 0.95))
    elif kind == "ginkgo":  # fan
        for i in range(7):
            t = math.radians(30 + 120 * i / 6)
            pts.append((0.55 * math.cos(t), 0.55 * math.sin(t)))
        pts.append((0.0, -0.05))
    elif kind == "needle":  # tuft
        pts = [(-0.5, -0.06), (0.5, -0.06), (0.5, 0.06), (-0.5, 0.06)]
    else:  # ellipse
        for i in range(8):
            t = 2 * math.pi * i / 8
            pts.append((0.3 * math.cos(t), 0.5 * math.sin(t)))
    return pts


class LeafBuilder:
    def __init__(self, name, kind, mat):
        self.name, self.kind, self.mat = name, kind, mat
        self.shape = leaf_shape(kind)
        self.verts, self.faces, self.vals = [], [], []

    def add(self, pos, size, normal=None, val=None):
        if normal is None:
            normal = Vector((random.gauss(0, 0.6), random.gauss(0, 0.6), random.uniform(0.4, 1.0))).normalized()
        x = normal.orthogonal().normalized()
        rot = random.uniform(0, 2 * math.pi)
        x = (Matrix.Rotation(rot, 3, normal) @ x)
        y = normal.cross(x)
        base = len(self.verts)
        c = Vector(pos)
        self.verts.append(c.to_tuple())
        for u, v in self.shape:
            p = c + x * (u * size) + y * (v * size) + normal * (-(u * u + v * v) * size * 0.25)
            self.verts.append(p.to_tuple())
        n = len(self.shape)
        for i in range(n):
            self.faces.append((base, base + 1 + i, base + 1 + (i + 1) % n))
        self.vals.append((val if val is not None else random.random(), n))

    def build(self):
        me = bpy.data.meshes.new(self.name)
        me.from_pydata(self.verts, [], self.faces)
        attr = me.attributes.new("leafcol", "FLOAT", "FACE")
        data = []
        for v, n in self.vals:
            data.extend([v] * n)
        attr.data.foreach_set("value", data)
        ob = bpy.data.objects.new(self.name, me)
        ob.data.materials.append(self.mat)
        return link(ob)


def maple(name, root, height, spread, leafmat, seed, n_leaves=26000, leaf_size=0.075, lean=(0, 0)):
    """Spreading, layered Japanese maple: several leaders, horizontal-ish branching, leaves at tips."""
    rnd = random.Random(seed)
    bm = bmesh.new()
    tips = []

    def grow(p, d, length, r, depth):
        steps = 3
        q = p
        for s in range(steps):
            d = (d + Vector((rnd.gauss(0, 0.15), rnd.gauss(0, 0.15), rnd.gauss(0.02, 0.1)))).normalized()
            nq = q + d * (length / steps)
            tube(bm, q, nq, r * (1 - s / steps * 0.3), r * (1 - (s + 1) / steps * 0.3), 6 if depth < 2 else 4)
            q = nq
        if depth >= 5 or r < 0.006:
            tips.append((q, d))
            return
        if depth >= 3:
            tips.append((q, d))
        nkids = 2 if depth < 1 else rnd.choice((2, 3, 3))
        for k in range(nkids):
            az = rnd.uniform(0, 2 * math.pi)
            side = Vector((math.cos(az), math.sin(az), 0))
            nd = (d * 0.55 + side * 0.65 + Vector((0, 0, rnd.uniform(-0.15, 0.25)))).normalized()
            grow(q, nd, length * rnd.uniform(0.62, 0.78), r * 0.62, depth + 1)

    base = Vector(root)
    trunk_len = height * 0.35
    for k in range(rnd.choice((2, 3))):
        az = rnd.uniform(0, 2 * math.pi)
        d = Vector((math.cos(az) * 0.35 + lean[0], math.sin(az) * 0.35 + lean[1], 1)).normalized()
        grow(base - Vector((0, 0, 0.1)), d, trunk_len, height * 0.022, 0)
    wood = mesh_obj(name + "_wood", bm, M["bark"], smooth=True)

    lb = LeafBuilder(name + "_leaves", "maple", leafmat)
    # leaves: clusters around tips, flattened into layered canopy pads
    per = max(1, n_leaves // max(1, len(tips)))
    for (q, d) in tips:
        cluster_r = spread * 0.075
        for i in range(per):
            off = Vector((rnd.gauss(0, cluster_r), rnd.gauss(0, cluster_r), rnd.gauss(0, cluster_r * 0.45)))
            # color varies with height + noise so the crown has red tops, orange interior
            v = min(1, max(0, 0.15 + 0.6 * rnd.random() + 0.25 * noise.noise(q * 0.7)))
            lb.add(q + off, leaf_size * rnd.uniform(0.75, 1.25), val=v)
    lb.build()
    # scale branch structure to spread
    return wood


def evergreen_blob(name, center, radii, mat, n=9000, size=0.12, seed=0, kind="ellipse"):
    rnd = random.Random(seed)
    lb = LeafBuilder(name, kind, mat)
    c = Vector(center)
    for i in range(n):
        # points near the ellipsoid surface
        v = Vector((rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 1))).normalized()
        rr = rnd.uniform(0.8, 1.0)
        bumpy = 1 + 0.18 * noise.noise(v * 2.5 + Vector((seed, 0, 0)))
        p = c + Vector((v.x * radii[0], v.y * radii[1], v.z * radii[2])) * rr * bumpy
        nrm = Vector((v.x / radii[0], v.y / radii[1], v.z / radii[2])).normalized()
        nrm = (nrm + Vector((rnd.gauss(0, 0.5), rnd.gauss(0, 0.5), rnd.gauss(0, 0.5)))).normalized()
        lb.add(p, size * rnd.uniform(0.7, 1.3), normal=nrm)
    lb.build()
    # inner dark core so gaps don't show sky
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1, location=center)
    o = bpy.context.object
    o.scale = (radii[0] * 0.8, radii[1] * 0.8, radii[2] * 0.8)
    o.data.materials.append(principled(name + "_core", (0.01, 0.02, 0.008), rough=1.0))


def pine(name, root, seed):
    """Cloud-pruned black pine (niwaki): leaning trunk with flat needle pads."""
    rnd = random.Random(seed)
    bm = bmesh.new()
    p = Vector(root)
    d = Vector((0.5, 0.2, 1)).normalized()
    pads = []
    r = 0.14
    for s in range(7):
        d = (d + Vector((rnd.gauss(0, 0.25), rnd.gauss(0, 0.2), 0.15))).normalized()
        q = p + d * 0.55
        tube(bm, p, q, r, r * 0.85, 8)
        r *= 0.85
        if s >= 2:
            az = rnd.uniform(0, 2 * math.pi)
            bd = Vector((math.cos(az), math.sin(az), 0.15)).normalized()
            e = q + bd * rnd.uniform(0.7, 1.3)
            tube(bm, q, e, r * 0.5, r * 0.3, 6)
            pads.append(e)
        p = q
    pads.append(p)
    mesh_obj(name + "_trunk", bm, M["bark_red"], smooth=True)
    for k, e in enumerate(pads):
        evergreen_blob(name + f"_pad{k}", (e.x, e.y, e.z + 0.1), (0.75, 0.6, 0.22), PINE, n=2200,
                       size=0.11, seed=seed * 10 + k, kind="needle")


# Hero maples
maple("maple_hero", (3.8, 8.6, terrain_h(3.8, 8.6)), 5.2, 5.0, MAPLE_RED, 11, n_leaves=34000, lean=(-0.15, -0.2))
maple("maple_left", (-6.2, 10.5, terrain_h(-6.2, 10.5)), 4.6, 4.5, MAPLE_ORANGE, 23, n_leaves=26000, lean=(0.2, 0))
maple("maple_near", (6.8, 3.6, GROUND), 3.6, 3.5, MAPLE_RED, 37, n_leaves=18000, lean=(-0.35, 0.05))
maple("maple_small", (-1.6, 12.6, terrain_h(-1.6, 12.6)), 3.2, 3.0, MAPLE_ORANGE, 41, n_leaves=12000)
pine("pine", (-4.6, 7.4, terrain_h(-4.6, 7.4)), 5)

# pruned azalea mounds (karikomi)
for k, (x, y, rx, rz) in enumerate([(-6.5, 5.6, 1.0, 0.55), (-5.3, 6.1, 0.7, 0.45), (6.2, 6.1, 1.1, 0.6),
                                    (4.7, 5.5, 0.6, 0.4), (-2.4, 12.4, 0.9, 0.5), (7.5, 9.6, 1.2, 0.7),
                                    (-8.4, 8.6, 1.3, 0.75), (1.0, 13.0, 1.0, 0.55), (-10, 4.5, 1.0, 0.6),
                                    (10, 5.0, 1.2, 0.65)]):
    evergreen_blob(f"azalea{k}", (x, y, terrain_h(x, y) + rz * 0.6), (rx, rx * 0.85, rz), SHRUB, n=int(5500 * rx),
                   size=0.07, seed=100 + k)

# background: evergreens and ginkgo behind the wall
for k in range(26):
    x = -26 + k * 2.0 + random.uniform(-0.8, 0.8)
    y = WALL_Y + random.uniform(2.5, 9)
    hgt = random.uniform(6, 11)
    if k % 5 == 2:
        evergreen_blob(f"ginkgo{k}", (x, y, hgt * 0.55), (2.0, 2.0, hgt * 0.45), GINKGO, n=9000, size=0.16,
                       seed=200 + k, kind="ginkgo")
    else:
        evergreen_blob(f"cedar{k}", (x, y, hgt * 0.5), (1.6, 1.6, hgt * 0.5), EVERGREEN, n=6000, size=0.2,
                       seed=300 + k)
    box("bg_trunk", (x, y, hgt * 0.15), (0.3, 0.3, hgt * 0.4), M["bark"])
for s_ in (-1, 1):
    for k in range(9):
        x = s_ * random.uniform(18, 23)
        y = -6 + k * 3.0 + random.uniform(-1, 1)
        hgt = random.uniform(6, 10)
        evergreen_blob(f"outer_side{s_}_{k}", (x, y, hgt * 0.5), (1.8, 1.8, hgt * 0.5),
                       GINKGO if k % 4 == 1 else EVERGREEN, n=5000, size=0.22, seed=500 + k + 20 * s_,
                       kind="ginkgo" if k % 4 == 1 else "ellipse")
# inside-wall background trees at the sides
for k, (x, y) in enumerate([(-12, 12), (-13.5, 7), (12.5, 12.5), (13, 6.5), (-11, 2), (11.5, 2.2)]):
    hgt = random.uniform(5, 7)
    mat = [MAPLE_ORANGE, EVERGREEN, GINKGO, EVERGREEN, MAPLE_RED, EVERGREEN][k]
    kind = "ginkgo" if mat is GINKGO else ("maple" if mat in (MAPLE_RED, MAPLE_ORANGE) else "ellipse")
    evergreen_blob(f"side_tree{k}", (x, y, hgt * 0.6), (2.0, 2.0, hgt * 0.4), mat, n=12000, size=0.13,
                   seed=400 + k, kind=kind)
    box("side_trunk", (x, y, hgt * 0.2), (0.25, 0.25, hgt * 0.5), M["bark"])

# fallen leaves on moss, gravel, stones and the engawa edge
lb = LeafBuilder("fallen_leaves", "maple", FALLEN)
for i in range(9000):
    # bias toward under the maples
    tx, ty, tr = random.choice([(3.8, 8.6, 4.0), (-6.2, 10.5, 3.5), (6.8, 3.6, 3.0), (0, 6, 8)])
    x = tx + random.gauss(0, tr * 0.6)
    y = ty + random.gauss(0, tr * 0.6)
    if y < ENG_D + 0.1 or abs(x) > 15 or y > WALL_Y - 0.3:
        continue
    if in_poly(x, y, POND):
        z = GROUND - 0.065  # floating on the pond
    elif GRAVEL[0] < x < GRAVEL[2] and GRAVEL[1] < y < GRAVEL[3]:
        if random.random() < 0.6:
            continue
        z = gravel_h(x, y) + 0.015
    else:
        z = terrain_h(x, y) + 0.02
    nrm = Vector((random.gauss(0, 0.15), random.gauss(0, 0.15), 1)).normalized()
    lb.add((x, y, z), random.uniform(0.06, 0.09), normal=nrm)
for i in range(14):  # a few leaves blown onto the engawa
    x = random.uniform(-3, 4)
    y = random.uniform(0.7, ENG_D - 0.05)
    if abs(x) < 0.5 and y < 0.9:
        continue
    lb.add((x, y, 0.004), random.uniform(0.06, 0.08), normal=Vector((0, 0, 1)))
lb.build()

# ---------------------------------------------------------------- light & world
world = bpy.data.worlds.new("world")
scene.world = world
world.use_nodes = True
wn = world.node_tree
wn.nodes.clear()
sky = wn.nodes.new("ShaderNodeTexSky")
sky.sky_type = "MULTIPLE_SCATTERING"
SUN_ELEV = math.radians(32)
SUN_ROT = math.radians(-40)  # front-left morning sun -> backlit maples (negative = toward -X)
sky.sun_elevation = SUN_ELEV
sky.sun_rotation = SUN_ROT
sky.air_density = 1.0
sky.aerosol_density = 1.3  # morning haze softens the light
sky.ozone_density = 1.0
sky.sun_disc = False  # the sun lamp provides direct light
sky.sun_size = math.radians(1.2)
sky.sun_intensity = 0.7
bg = wn.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 0.16
wo = wn.nodes.new("ShaderNodeOutputWorld")
wn.links.new(sky.outputs[0], bg.inputs[0])
wn.links.new(bg.outputs[0], wo.inputs[0])

# match a sun lamp to the sky's sun so shadows are crisp-but-soft
sun_data = bpy.data.lights.new("sun", "SUN")
sun_data.energy = 6.0
sun_data.angle = math.radians(4.0)
sun_data.color = (1.0, 0.86, 0.68)
sun = bpy.data.objects.new("sun", sun_data)
link(sun)
# sky texture convention: rotation is azimuth from +Y toward -X? derive direction vector explicitly
sd = Vector((math.sin(SUN_ROT) * -1 * math.cos(SUN_ELEV) * -1, math.cos(SUN_ROT) * math.cos(SUN_ELEV), math.sin(SUN_ELEV)))
sun.rotation_euler = sd.to_track_quat("Z", "Y").to_euler()

# portal at the shoji opening helps the interior gather skylight
pd = bpy.data.lights.new("portal", "AREA")
pd.shape = "RECTANGLE"
pd.size = 2 * ROOM_W
pd.size_y = H_KAMOI
pd.cycles.is_portal = True
portal = bpy.data.objects.new("portal", pd)
portal.location = (0, -0.1, H_KAMOI / 2)
portal.rotation_euler = (math.radians(90), 0, 0)  # face -Y (into room)
link(portal)

# warm low fill inside the room (paper lantern feel), very subtle
fd = bpy.data.lights.new("room_fill", "AREA")
fd.size = 3.0
fd.energy = 25
fd.color = (1.0, 0.8, 0.6)
fill = bpy.data.objects.new("room_fill", fd)
fill.location = (0, -2.6, H_CEIL - 0.05)
link(fill)
fill.visible_camera = False

# ---------------------------------------------------------------- camera & render
cam_data = bpy.data.cameras.new("eye")
cam = bpy.data.objects.new("eye", cam_data)
link(cam)
cam.location = (0, 0.42, 0.9)
cam.rotation_euler = (math.radians(90), 0, 0)  # level, looking +Y (toward the garden)
if args.persp:
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(100)
    scene.render.resolution_x = args.res
    scene.render.resolution_y = int(args.res * 0.6)
else:
    cam_data.type = "PANO"
    cam_data.panorama_type = "EQUIRECTANGULAR"
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res // 2
cam_data.clip_start = 0.05
cam_data.clip_end = 1000
scene.camera = cam

scene.render.engine = "CYCLES"
cy = scene.cycles
cy.device = "CPU"
cy.samples = args.samples
cy.use_adaptive_sampling = True
cy.adaptive_threshold = 0.02
cy.use_denoising = True
cy.max_bounces = 8
cy.diffuse_bounces = 4
cy.glossy_bounces = 3
cy.transmission_bounces = 6
cy.transparent_max_bounces = 8
cy.caustics_reflective = False
cy.caustics_refractive = False
cy.sample_clamp_indirect = 8.0
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Base Contrast"
scene.view_settings.exposure = 0.0
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = args.out
scene.render.threads_mode = "AUTO"

if args.exr:
    # pass_index 1 = maple leaves, 2 = other foliage (used by the painterly style scripts)
    for ob in bpy.data.objects:
        if ob.type == "MESH" and ob.data.materials and ob.data.materials[0]:
            mn = ob.data.materials[0].name
            if mn.startswith("maple") or mn == "fallen":
                ob.pass_index = 1
            elif mn in ("ginkgo", "pine_needles", "evergreen", "azalea") or mn.endswith("_core"):
                ob.pass_index = 2
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_normal = True
    vl.use_pass_object_index = True
    vl.use_pass_diffuse_color = True
    scene.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_depth = "32"
if args.blend:
    bpy.ops.wm.save_as_mainfile(filepath=args.blend)
print("objects:", len(bpy.data.objects), "rendering", scene.render.resolution_x, "x", scene.render.resolution_y)
bpy.ops.render.render(write_still=True)
if args.exr:  # also keep the colour-managed beauty image next to the EXR
    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    bpy.data.images["Render Result"].save_render(args.out.rsplit(".", 1)[0] + "_beauty.png")
print("done:", args.out)
