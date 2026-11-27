"""3D CRUSH Short (27 Nov 2026): a wrecking ball swings into a brick wall four times, each swing from higher up.

Run headless:  blender -b -P blender/wreckingwall3d.py -- OUTDIR [FRAMES] [SAMPLES] [RENDER]
Writes OUTDIR/f_0001.png ..., OUTDIR/sim.npz (ball and brick positions per frame), OUTDIR/sim.json
(hit frames, ball speed at each hit, bricks knocked out) and OUTDIR/timing.txt. RENDER=0 simulates only.

Physics: 172 bricks (running bond, half bricks at the row ends) are active rigid boxes that start asleep.
The ball is a kinematic body whose swing is a real pendulum (theta'' = -g/L sin theta) released from rest
at a bigger angle each time; inside the wall it is braked to a set depth, then a crane pulls it back.
The answer is measured: a brick counts as knocked out when its centre ends more than 0.30 m from where it
was laid.
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[0] if argv else os.path.abspath("out/3d_wreckingwall")
FRAMES = int(argv[1]) if len(argv) > 1 else 750
SAMPLES = int(argv[2]) if len(argv) > 2 else 16
RENDER = int(argv[3]) if len(argv) > 3 else 1
FPS = 30
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ the swing (pure python, before Blender)
G, PIVOT_Z, L, R = 9.81, 9.0, 7.6, 0.6
BRICK = (0.5, 0.25, 0.2)                       # length (x), depth (y), height (z), metres
ROWS, COLS = 15, 11
GAP = 0.004
WALL_FRONT = -BRICK[1] / 2
PIVOT_Y = WALL_FRONT - R - 0.02                # ball touches the wall at theta ~ 0
# (release angle deg, braking depth m, x of the hit, target hit time s)
HITS = [(22, 0.22, -1.35, 0.8), (38, 0.40, 1.30, 6.0), (56, 0.65, -0.25, 11.4), (80, 1.55, 0.35, 16.8)]


def quarter(a_deg, dt=1 / 3000):
    th, w, t = -math.radians(a_deg), 0.0, 0.0
    while th < 0:
        w += -(G / L) * math.sin(th) * dt
        th += w * dt
        t += dt
    return t


def schedule():
    """theta(t) and pivot x(t) sampled at every frame (and 10 substeps) from a real pendulum plus crane pulls."""
    dt = 1 / (FPS * 50)
    n = int(FRAMES * 50) + 1
    th_out, x_out = np.zeros(n), np.zeros(n)
    hit_t, hit_v = [], []
    # hit 1: released at rest before t=0 so the ball is already swinging on frame 0
    a, d, x, tgt = HITS[0]
    t = tgt - quarter(a)
    th, w, k, phase, idx = -math.radians(a), 0.0, 0.0, "free", 0
    xcur = x
    pull = None
    out_i = 0
    while out_i < n:
        if t >= 0 and out_i < n and t >= out_i * dt - 1e-12:
            th_out[out_i], x_out[out_i] = th, xcur
            out_i += 1
        if phase == "free":
            acc = -(G / L) * math.sin(th)
            if th > 0 and w > 0:
                if k == 0.0:                    # first contact: brake so the ball stops at depth d
                    k = w * L / HITS[idx][1]
                    hit_t.append(t)
                    hit_v.append(w * L)
                acc -= k * w
            w += acc * dt
            th += w * dt
            if th > 0 and w <= 0 and k > 0:       # stopped inside the wall: crane pulls it back
                if idx + 1 < len(HITS):
                    a2, d2, x2, tgt2 = HITS[idx + 1]
                    t_rel = tgt2 - quarter(a2)
                    pull = (t, t_rel, th, -math.radians(a2), xcur, x2)
                else:
                    pull = (t, t + 3.2, th, -math.radians(9), xcur, xcur)
                phase = "pull"
        elif phase == "pull":
            t0, t1, th0, th1, x0, x1 = pull
            s = min(1.0, (t - t0) / (t1 - t0))
            e = s * s * (3 - 2 * s)
            th = th0 + (th1 - th0) * e
            xcur = x0 + (x1 - x0) * e
            if s >= 1.0:
                if idx + 1 < len(HITS):
                    idx += 1
                    phase, w, k = "free", 0.0, 0.0
                else:
                    phase, w, k = "hold", 0.0, 0.0
        else:                                     # hold: small damped sway
            acc = -(G / L) * math.sin(th + math.radians(9)) - 0.6 * w
            w += acc * dt
            th += w * dt
        t += dt
    return th_out[::50], x_out[::50], th_out, x_out, hit_t, hit_v


TH, XS, TH_FINE, XS_FINE, HIT_T, HIT_V = schedule()
print("HITS s", [round(v, 2) for v in HIT_T], "speed km/h", [round(v * 3.6, 1) for v in HIT_V])


def ball_pos(th, x):
    return Vector((x, PIVOT_Y + L * math.sin(th), PIVOT_Z - L * math.cos(th)))


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
sc.render.fps = FPS
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
world.node_tree.nodes["Background"].inputs[0].default_value = (0.035, 0.05, 0.085, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0


def mat_metal(name, color, rough=0.3):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = rough
    return m


def mat_brick():
    m = bpy.data.materials.new("Brick")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.78
    info = nt.nodes.new("ShaderNodeObjectInfo")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.50, 0.12, 0.05, 1)
    ramp.color_ramp.elements[1].color = (0.80, 0.30, 0.12, 1)
    nt.links.new(info.outputs["Random"], ramp.inputs[0])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.35
    nt.links.new(ramp.outputs["Color"], mix.inputs[6])
    nt.links.new(noise.outputs["Fac"], mix.inputs[7])
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    return m


def mat_floor():
    m = bpy.data.materials.new("Floor")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.05, 0.055, 0.065, 1)
    b.inputs["Roughness"].default_value = 0.35
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    lines = []
    for ax in ("X", "Y"):
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 1.0
        fr = nt.nodes.new("ShaderNodeMath"); fr.operation = "FRACT"
        lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = 0.025
        nt.links.new(sep.outputs[ax], mul.inputs[0]); nt.links.new(mul.outputs[0], fr.inputs[0])
        nt.links.new(fr.outputs[0], lt.inputs[0]); lines.append(lt)
    mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"
    nt.links.new(lines[0].outputs[0], mx.inputs[0]); nt.links.new(lines[1].outputs[0], mx.inputs[1])
    sc_ = nt.nodes.new("ShaderNodeMath"); sc_.operation = "MULTIPLY"; sc_.inputs[1].default_value = 1.6
    nt.links.new(mx.outputs[0], sc_.inputs[0])
    b.inputs["Emission Color"].default_value = (0.1, 0.85, 1.0, 1)
    nt.links.new(sc_.outputs[0], b.inputs["Emission Strength"])
    return m


def rigid(obj, kind="ACTIVE", shape="BOX", mass=1.0):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add(type=kind)
    rb = obj.rigid_body
    rb.collision_shape = shape
    rb.mass = mass
    rb.friction = 0.7
    rb.restitution = 0.05
    rb.collision_margin = 0.002
    obj.select_set(False)
    return rb


bpy.ops.rigidbody.world_add()
rbw = sc.rigidbody_world
rbw.substeps_per_frame = 10
rbw.solver_iterations = 20
rbw.point_cache.frame_start, rbw.point_cache.frame_end = 1, FRAMES

bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
floor = bpy.context.active_object
floor.data.materials.append(mat_floor())
rigid(floor, "PASSIVE", "BOX")

# ------------------------------------------------------------------ the wall: running bond, half bricks at the ends
BM = mat_brick()
bricks, origin = [], []
W = COLS * BRICK[0] + (COLS - 1) * GAP
for r in range(ROWS):
    z = r * (BRICK[2] + GAP) + BRICK[2] / 2 + 0.001
    if r % 2 == 0:
        lens = [BRICK[0]] * COLS
    else:
        lens = [BRICK[0] / 2 - GAP / 2] + [BRICK[0]] * (COLS - 1) + [BRICK[0] / 2 - GAP / 2]
    x = -W / 2
    for ln in lens:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x + ln / 2, 0, z))
        ob = bpy.context.active_object
        ob.scale = (ln, BRICK[1], BRICK[2])
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        ob.data.materials.append(BM)
        bev = ob.modifiers.new("bev", "BEVEL")
        bev.width, bev.segments = 0.014, 2
        rb = rigid(ob, "ACTIVE", "BOX", mass=3.0 * ln / BRICK[0])
        rb.use_deactivation = True
        rb.use_start_deactivated = True
        bricks.append(ob)
        origin.append(tuple(ob.location))
        x += ln + GAP
origin = np.array(origin)
print("BRICKS", len(bricks))

# ------------------------------------------------------------------ the ball (kinematic) and its chain
STEEL = mat_metal("Steel", (0.2, 0.21, 0.23), 0.32)
bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=R, location=ball_pos(TH[0], XS[0]))
ball = bpy.context.active_object
bpy.ops.object.shade_smooth()
ball.data.materials.append(STEEL)
rigid(ball, "PASSIVE", "SPHERE")
ball.rigid_body.kinematic = True
bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.12, depth=0.25, location=(0, 0, 0))
lug = bpy.context.active_object
lug.data.materials.append(STEEL)

pivot = bpy.data.objects.new("Pivot", None)
sc.collection.objects.link(pivot)
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.045, depth=L - R, location=(0, 0, 0))
chain = bpy.context.active_object
chain.data.materials.append(mat_metal("Chain", (0.55, 0.56, 0.6), 0.25))
chain.parent = pivot
chain.location = (0, 0, -(L - R) / 2)
lug.parent = pivot
lug.location = (0, 0, -(L - R))

for f in range(1, FRAMES + 1):
    th, x = TH[f - 1], XS[f - 1]
    ball.location = ball_pos(th, x)
    ball.keyframe_insert("location", frame=f)
    pivot.location = (x, PIVOT_Y, PIVOT_Z)
    pivot.rotation_euler = (th, 0, 0)
    pivot.keyframe_insert("location", frame=f)
    pivot.keyframe_insert("rotation_euler", frame=f)
for ob in (ball, pivot):
    for fc in ob.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

# ------------------------------------------------------------------ lights and camera (slow push-in)
def area(name, loc, energy, color, size, target=(0, 0, 1.3)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.color, ld.size = energy, color, size
    ob = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return ob


area("Key", (6, -6, 8), 2600, (1.0, 0.9, 0.78), 5)
area("Rim", (-6, 5, 5), 1600, (0.25, 0.75, 1.0), 4)
area("Fill", (2, -8, 2), 500, (0.6, 0.8, 1.0), 4)
cd = bpy.data.cameras.new("Cam")
cd.lens = 35
cd.sensor_fit = "VERTICAL"
cd.sensor_height = 36
cd.shift_y = 0.1
cam = bpy.data.objects.new("Cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
target = Vector((0, 0.3, 1.45))
dirv = Vector((0.62, -0.75, 0.24)).normalized()
for f, dist in ((1, 12.5), (FRAMES, 9.4)):
    cam.location = target + dirv * dist
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
for fc in cam.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "BEZIER"

# ------------------------------------------------------------------ simulate in order, record, render
pos = np.zeros((FRAMES, len(bricks), 3), np.float32)
bpos = np.zeros((FRAMES, 3), np.float32)
times = []
t0 = time.time()
for f in range(1, FRAMES + 1):
    sc.frame_set(f)
    for i, ob in enumerate(bricks):
        pos[f - 1, i] = ob.matrix_world.translation
    bpos[f - 1] = ball.matrix_world.translation
    if RENDER:
        sc.render.filepath = os.path.join(OUT, f"f_{f:04d}.png")
        ts = time.time()
        bpy.ops.render.render(write_still=True)
        times.append(time.time() - ts)
    if f in (1, 2, 10) or f % 50 == 0:
        print(f"FRAME {f}/{FRAMES} {times[-1] if times else 0:.2f}s", flush=True)

disp = np.linalg.norm(pos - origin[None, :, :], axis=2)
out_by_frame = (disp > 0.30).sum(axis=1)
hit_frames = [int(round(t * FPS)) + 1 for t in HIT_T]
per_hit = []
for k, hf in enumerate(hit_frames):
    end = hit_frames[k + 1] - 1 if k + 1 < len(hit_frames) else FRAMES
    per_hit.append(int(out_by_frame[end - 1]))
np.savez_compressed(os.path.join(OUT, "sim.npz"), pos=pos, ball=bpos, origin=origin)
res = dict(bricks=len(bricks), hit_frames=hit_frames, hit_seconds=[round(t, 3) for t in HIT_T],
           hit_speed_kmh=[round(v * 3.6, 1) for v in HIT_V], release_angle_deg=[h[0] for h in HITS],
           knocked_out_after_hit=per_hit, knocked_out_by_frame=[int(v) for v in out_by_frame],
           answer=int(out_by_frame[-1]), rule="brick centre more than 0.30 m from where it was laid",
           pendulum_m=L, ball_radius_m=R, brick_m=BRICK)
with open(os.path.join(OUT, "sim.json"), "w") as fh:
    json.dump(res, fh)
with open(os.path.join(OUT, "timing.txt"), "w") as fh:
    fh.write(f"engine {sc.render.engine} samples {SAMPLES} bricks {len(bricks)} frames {FRAMES} render {RENDER}\n")
    fh.write(f"total {time.time() - t0:.1f}s")
    if times:
        fh.write(f"  mean {sum(times) / len(times):.2f}s/frame  max {max(times):.2f}s")
    fh.write("\n")
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "knocked_out_by_frame"}))
print("DONE", open(os.path.join(OUT, "timing.txt")).read())
