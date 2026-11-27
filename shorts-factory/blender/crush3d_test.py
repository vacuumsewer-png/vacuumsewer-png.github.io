"""3D pilot (26 Nov 2026): a hydraulic press crushes a glossy pre-fractured block on a glowing grid.

Run headless:  blender -b -P blender/crush3d_test.py -- OUTDIR [FRAMES] [PIECES] [SAMPLES]
Writes OUTDIR/f_0001.png ... and OUTDIR/timing.txt. Free tools only (Blender EEVEE).
The block is split into Voronoi cells with bmesh bisects (no add-on needed); every cell is an active
rigid body that starts asleep; the press is an animated (kinematic) rigid body.
"""
import math
import os
import random
import sys
import time

import bmesh
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[0] if argv else os.path.abspath("out/3dtest")
FRAMES = int(argv[1]) if len(argv) > 1 else 150
PIECES = int(argv[2]) if len(argv) > 2 else 48
SAMPLES = int(argv[3]) if len(argv) > 3 else 16
os.makedirs(OUT, exist_ok=True)
rnd = random.Random(7)

# ------------------------------------------------------------------ scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try:
        sc.render.engine = eng
        break
    except TypeError:
        pass
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1080, 1920, 100
sc.render.fps = 30
sc.frame_start, sc.frame_end = 1, FRAMES
try:
    sc.eevee.taa_render_samples = SAMPLES
except AttributeError:
    pass
for attr, val in (("use_raytracing", True), ("use_shadows", True)):
    try:
        setattr(sc.eevee, attr, val)
    except AttributeError:
        pass
sc.view_settings.view_transform = "AgX"
try:
    sc.view_settings.look = "AgX - Medium High Contrast"
except TypeError:
    pass
sc.render.image_settings.file_format = "PNG"

world = bpy.data.worlds.new("W")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.012, 0.01, 0.03, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0


def mat_metal(name, color, rough=0.22, emit=None, emit_strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = rough
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = emit_strength
    return m


def mat_floor():
    m = bpy.data.materials.new("Floor")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.02, 0.02, 0.035, 1)
    b.inputs["Roughness"].default_value = 0.18
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    lines = []
    for ax in ("X", "Y"):
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 1.25
        fr = nt.nodes.new("ShaderNodeMath"); fr.operation = "FRACT"
        lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = 0.035
        nt.links.new(sep.outputs[ax], mul.inputs[0]); nt.links.new(mul.outputs[0], fr.inputs[0])
        nt.links.new(fr.outputs[0], lt.inputs[0]); lines.append(lt)
    mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"
    nt.links.new(lines[0].outputs[0], mx.inputs[0]); nt.links.new(lines[1].outputs[0], mx.inputs[1])
    b.inputs["Emission Color"].default_value = (1.0, 0.12, 0.55, 1)
    nt.links.new(mx.outputs[0], b.inputs["Emission Strength"])
    return m


def rigid(obj, kind="ACTIVE", shape="CONVEX_HULL", mass=1.0):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add(type=kind)
    rb = obj.rigid_body
    rb.collision_shape = shape
    rb.mass = mass
    rb.friction = 0.6
    rb.restitution = 0.05
    rb.collision_margin = 0.002
    obj.select_set(False)
    return rb


bpy.ops.rigidbody.world_add()
rbw = sc.rigidbody_world
rbw.substeps_per_frame = 10
rbw.solver_iterations = 20
rbw.point_cache.frame_start, rbw.point_cache.frame_end = 1, FRAMES

# floor
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
floor = bpy.context.active_object
floor.data.materials.append(mat_floor())
rigid(floor, "PASSIVE", "BOX")

# ------------------------------------------------------------------ the block, fractured into Voronoi cells
S = 1.7                                                   # block edge
OUTER = mat_metal("Outer", (0.42, 0.18, 0.95), 0.2)       # violet metal
INNER = mat_metal("Inner", (0.55, 1.0, 0.12), 0.35, emit=(0.4, 1.0, 0.1), emit_strength=1.2)  # lime core
seeds = [Vector((rnd.uniform(-S / 2, S / 2), rnd.uniform(-S / 2, S / 2), rnd.uniform(0, S))) for _ in range(PIECES)]
pieces = []
for i, p in enumerate(seeds):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=S)
    bmesh.ops.translate(bm, vec=(0, 0, S / 2), verts=bm.verts)
    for f in bm.faces:
        f.material_index = 0
    for j, q in enumerate(seeds):
        if i == j:
            continue
        mid, n = (p + q) / 2, (q - p).normalized()
        res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=mid, plane_no=n,
                                     clear_outer=True)
        edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
        if edges:
            new = bmesh.ops.edgeloop_fill(bm, edges=edges)
            for f in new.get("faces", []):
                f.material_index = 1
        if not bm.verts:
            break
    if len(bm.verts) < 4:
        bm.free()
        continue
    me = bpy.data.meshes.new(f"cell{i}")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(f"cell{i}", me)
    sc.collection.objects.link(ob)
    ob.data.materials.append(OUTER)
    ob.data.materials.append(INNER)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
    ob.select_set(False)
    rb = rigid(ob, "ACTIVE", "CONVEX_HULL", mass=me.polygons and 0.5 + rnd.random() or 1.0)
    rb.use_deactivation = True
    rb.use_start_deactivated = True
    pieces.append(ob)
print("PIECES", len(pieces))

# ------------------------------------------------------------------ the press (kinematic)
bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=1.55, depth=0.6, location=(0, 0, 7))
press = bpy.context.active_object
press.data.materials.append(mat_metal("Steel", (0.75, 0.77, 0.82), 0.15))
rigid(press, "PASSIVE", "CYLINDER")
press.rigid_body.kinematic = True
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.35, depth=8, location=(0, 0, 11.3))
rod = bpy.context.active_object
rod.data.materials.append(press.data.materials[0])
rod.parent = press
rod.location = (0, 0, 4.3)
t_hit, t_down = int(FRAMES * 0.25), int(FRAMES * 0.78)
for f, z in ((1, 7.0), (t_hit, S + 0.3), (t_down, 0.32), (FRAMES, 0.32)):
    press.location.z = z
    press.keyframe_insert("location", frame=f)
for fc in press.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

# ------------------------------------------------------------------ lights and camera (slow push-in = escalation)
def area(name, loc, energy, color, size):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.color, ld.size = energy, color, size
    ob = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector((0, 0, 0.8)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return ob


area("Key", (4, -4, 7), 1400, (1.0, 0.95, 0.9), 4)
area("Rim", (-5, 4, 4), 900, (0.3, 0.6, 1.0), 3)
area("Fill", (0, -6, 2), 250, (1.0, 0.4, 0.8), 3)
cd = bpy.data.cameras.new("Cam")
cd.lens = 40
cam = bpy.data.objects.new("Cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
target = Vector((0, 0, 1.1))
for f, dist, h in ((1, 9.5, 3.2), (FRAMES, 7.0, 2.4)):
    cam.location = Vector((0.0, -dist, h)) + Vector((dist * 0.35, 0, 0))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)

# ------------------------------------------------------------------ simulate in order and render each frame
times = []
t0 = time.time()
for f in range(1, FRAMES + 1):
    sc.frame_set(f)
    sc.render.filepath = os.path.join(OUT, f"f_{f:04d}.png")
    ts = time.time()
    bpy.ops.render.render(write_still=True)
    times.append(time.time() - ts)
    if f in (1, 2, 10) or f % 25 == 0:
        print(f"FRAME {f}/{FRAMES} {times[-1]:.2f}s", flush=True)
with open(os.path.join(OUT, "timing.txt"), "w") as fh:
    fh.write(f"engine {sc.render.engine} samples {SAMPLES} pieces {len(pieces)} frames {FRAMES}\n")
    fh.write(f"total {time.time() - t0:.1f}s  mean {sum(times) / len(times):.2f}s/frame  max {max(times):.2f}s\n")
print("DONE", open(os.path.join(OUT, "timing.txt")).read())
