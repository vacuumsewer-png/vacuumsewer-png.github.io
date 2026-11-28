"""BRICK STACK (SCALE family, episode 3): how many toy bricks can you stack before the bottom one breaks?

Plastic 2x2 toy bricks are stacked on one brick in escalating batches; the counter climbs; the camera pulls back past
a person, a tree, a house, the Eiffel Tower (330 m), the Burj Khalifa (828 m) and the clouds. A close-up circle shows
the bottom brick: it whitens, squashes and cracks as the load grows, and it gives way only when the weight of the
bricks above it beats the load a real brick survived in a lab test.

Physics (constants stated here, the answer is computed from them, nothing is tuned to a target):
  * one 2x2 plastic toy brick weighs about 1.15 g and is 9.6 mm tall without studs (15.8 mm x 15.8 mm footprint);
  * in a 2012 Open University load test a 2x2 brick withstood about 4,240 N before it deformed;
  * the bottom brick carries the weight of every brick above it: n_above x 1.15 g x 9.81 m/s^2;
  * bricks above needed = ceil(4,240 N / (0.00115 kg x 9.81)) ~ 375,800, plus the bottom brick itself (ANSWER);
  * tower height = ANSWER x 9.6 mm ~ 3.6 km (honest: the camera and reference objects use this height).
The tower is drawn at least MINW px wide so it stays visible at 3.6 km (true width 15.8 mm); heights are honest.
The seed only changes cosmetics (crack shapes, shard spin, dust). Deterministic.
"""
import math

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import TAU, clamp, darken, ease_in, ease_out, glow, hex_rgb, lighten, mix, ramp
from engine.scale import H, W, ScaleScene
from scenes.balloonlift import draw_house, draw_person, draw_tree

# ---------------------------------------------------------------------- physics
M_BRICK = 0.00115           # kg, one 2x2 brick
F_FAIL = 4240.0             # N, load a 2x2 brick withstood before deforming (Open University test, 2012)
G = 9.81                    # m/s^2
BH = 0.0096                 # m, brick height without studs
BW = 0.0158                 # m, brick width (and depth)
STUD_H = 0.0017             # m
ABOVE = int(math.ceil(F_FAIL / (M_BRICK * G)))                # bricks the bottom one can carry
ANSWER = ABOVE + 1                                             # the tower, bottom brick included
TOWER_M = ANSWER * BH

# ---------------------------------------------------------------------- look / layout
OX, OY = 0.32, 0.25         # oblique projection of the depth axis (px per px of depth)
MINW = 14.0                 # px, minimum drawn width of the tower
COLS = [hex_rgb(c) for c in ("#E3262E", "#F7C318", "#1E6FD9")]
PERSON_X, TREE_X, HOUSE_X = -0.9, 4.2, -14.0
EIFFEL_X, BURJ_X = -150.0, -430.0
EIFFEL = ((0, 62.0), (57, 33.0), (115, 18.0), (200, 9.0), (276, 5.0), (300, 2.0), (324, 0.8), (330, 0.3))
BURJ = ((150, 32.0), (300, 26.0), (440, 20.0), (570, 14.0), (670, 9.0), (750, 5.0), (790, 2.6), (828, 0.7))
CLOUDS = ((-650, 1050, 520, 0), (620, 1420, 640, 0), (-260, 2050, 820, 1), (520, 2650, 700, 0),
          (-820, 3150, 640, 0), (300, 3500, 560, 1))
CXI, CYI, RI = 795.0, 1180.0, 150.0      # close-up circle of the bottom brick (screen px)
FW_I = 178.0                              # bottom brick front width inside the circle


def score_seed(seed):
    return dict(seed=seed, score=1.0, answer=ANSWER)


def hash1(a, b=0.0):
    v = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - math.floor(v)


def norm(x):
    return x / max(1e-9, np.abs(x).max())


def clack(rng, f, dur=0.08):
    """Hard plastic on plastic: a short bright noise click plus two damped partials."""
    t = np.arange(int(dur * au.SR)) / au.SR
    body = 0.8 * norm(au.noise(rng, dur, 1800, 8000)) * np.exp(-t / 0.005)
    body += 0.7 * np.sin(TAU * f * t) * np.exp(-t / 0.020)
    body += 0.35 * np.sin(TAU * f * 2.76 * t) * np.exp(-t / 0.010)
    return norm(body * np.minimum(1.0, t / 0.0008))


# ---------------------------------------------------------------------- brick drawing (screen px, y down)
def slab(ctx, x0, yb, fw, h, col, lw=0.0):
    """Front face and right side face of a block fw wide and h tall whose front-bottom-left corner is (x0, yb)."""
    dx, dy = OX * fw, OY * fw
    g = cairo.LinearGradient(x0, 0, x0 + fw, 0)
    g.add_color_stop_rgb(0, *lighten(col, 0.10))
    g.add_color_stop_rgb(0.14, *lighten(col, 0.30))
    g.add_color_stop_rgb(0.30, *col)
    g.add_color_stop_rgb(1, *darken(col, 0.16))
    ctx.rectangle(x0, yb - h, fw, h)
    ctx.set_source(g)
    ctx.fill()
    ctx.move_to(x0 + fw, yb)
    ctx.line_to(x0 + fw + dx, yb - dy)
    ctx.line_to(x0 + fw + dx, yb - h - dy)
    ctx.line_to(x0 + fw, yb - h)
    ctx.close_path()
    g = cairo.LinearGradient(0, yb - h - dy, 0, yb)
    g.add_color_stop_rgb(0, *darken(col, 0.30))
    g.add_color_stop_rgb(1, *darken(col, 0.45))
    ctx.set_source(g)
    ctx.fill()
    if lw > 0:
        ctx.move_to(x0, yb)
        ctx.line_to(x0 + fw, yb)
        ctx.line_to(x0 + fw + dx, yb - dy)
        ctx.line_to(x0 + fw + dx, yb - h - dy)
        ctx.line_to(x0 + dx, yb - h - dy)
        ctx.line_to(x0, yb - h)
        ctx.close_path()
        ctx.move_to(x0 + fw, yb)
        ctx.line_to(x0 + fw, yb - h)
        ctx.line_to(x0, yb - h)
        ctx.move_to(x0 + fw, yb - h)
        ctx.line_to(x0 + fw + dx, yb - h - dy)
        ctx.set_source_rgb(*darken(col, 0.62))
        ctx.set_line_width(lw)
        ctx.stroke()


def top_face(ctx, x0, yt, fw, col, lw=0.0):
    """Top face (with 2x2 studs when big enough) of a block whose front-top-left corner is (x0, yt)."""
    dx, dy = OX * fw, OY * fw
    ctx.move_to(x0, yt)
    ctx.line_to(x0 + fw, yt)
    ctx.line_to(x0 + fw + dx, yt - dy)
    ctx.line_to(x0 + dx, yt - dy)
    ctx.close_path()
    ctx.set_source_rgb(*lighten(col, 0.18))
    ctx.fill_preserve()
    if lw > 0:
        ctx.set_source_rgb(*darken(col, 0.62))
        ctx.set_line_width(lw)
        ctx.stroke()
    else:
        ctx.new_path()
    r = 0.152 * fw
    if r < 1.2:
        return
    ry = 0.42 * r
    hs = 0.108 * fw
    for v in (0.75, 0.25):
        for u in (0.25, 0.75):
            cx, cy = x0 + u * fw + v * dx, yt - v * dy
            # cylinder wall
            ctx.save()
            ctx.translate(cx, cy)
            ctx.scale(1.0, ry / r)
            ctx.arc(0, 0, r, 0, math.pi)
            ctx.restore()
            ctx.line_to(cx - r, cy - hs)
            ctx.line_to(cx + r, cy - hs)
            ctx.close_path()
            g = cairo.LinearGradient(cx - r, 0, cx + r, 0)
            g.add_color_stop_rgb(0, *lighten(col, 0.05))
            g.add_color_stop_rgb(0.25, *lighten(col, 0.28))
            g.add_color_stop_rgb(1, *darken(col, 0.30))
            ctx.set_source(g)
            ctx.fill_preserve()
            if lw > 0:
                ctx.set_source_rgb(*darken(col, 0.62))
                ctx.set_line_width(lw * 0.8)
                ctx.stroke()
            else:
                ctx.new_path()
            # cap
            ctx.save()
            ctx.translate(cx, cy - hs)
            ctx.scale(1.0, ry / r)
            ctx.arc(0, 0, r, 0, TAU)
            ctx.restore()
            ctx.set_source_rgb(*lighten(col, 0.30))
            ctx.fill_preserve()
            if lw > 0:
                ctx.set_source_rgb(*darken(col, 0.62))
                ctx.set_line_width(lw * 0.8)
                ctx.stroke()
            else:
                ctx.new_path()
            if r > 4:
                ctx.save()
                ctx.translate(cx - 0.35 * r, cy - hs - 0.1 * ry)
                ctx.scale(0.32 * r, 0.16 * r)
                ctx.arc(0, 0, 1, 0, TAU)
                ctx.restore()
                ctx.set_source_rgba(1, 1, 1, 0.75)
                ctx.fill()


# ---------------------------------------------------------------------- landmarks (world metres, y up)
def profile_path(ctx, prof, x_c, steps=False):
    pts_r = []
    if steps:
        y0 = 0.0
        for y1, w in prof:
            pts_r += [(w, y0), (w, y1)]
            y0 = y1
    else:
        pts_r = [(w, y) for y, w in prof]
    ctx.move_to(x_c - pts_r[0][0], pts_r[0][1])
    for w, y in pts_r:
        ctx.line_to(x_c + w, y)
    for w, y in reversed(pts_r):
        ctx.line_to(x_c - w, y)
    ctx.close_path()


def draw_eiffel(ctx, s, x_c):
    ctx.save()
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    profile_path(ctx, EIFFEL, x_c)
    ctx.save()                                        # the big arch between the legs
    ctx.translate(x_c, 0)
    ctx.scale(36.0, 40.0)
    ctx.move_to(1, 0)
    ctx.arc(0, 0, 1, 0, math.pi)
    ctx.close_path()
    ctx.restore()
    g = cairo.LinearGradient(x_c - 60, 0, x_c + 60, 0)
    g.add_color_stop_rgb(0, 0.52, 0.38, 0.27)
    g.add_color_stop_rgb(1, 0.30, 0.21, 0.15)
    ctx.set_source(g)
    ctx.fill()
    ctx.restore()
    for y, w in ((57, 38.0), (115, 21.0), (276, 6.5)):                  # platforms
        ctx.rectangle(x_c - w, y - 4.0, 2 * w, 6.0)
    ctx.set_source_rgb(0.26, 0.18, 0.12)
    ctx.fill()
    if s * 330 > 260:                                                  # lattice hint
        ctx.save()
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        profile_path(ctx, EIFFEL, x_c)
        ctx.clip()
        ctx.set_line_width(max(0.6, 1.2 / s))
        ctx.set_source_rgba(0.18, 0.12, 0.08, 0.55)
        for i in range(-40, 40):
            ctx.move_to(x_c + i * 9.0, 0)
            ctx.line_to(x_c + i * 9.0 + 330 * 0.35, 330)
            ctx.move_to(x_c + i * 9.0, 0)
            ctx.line_to(x_c + i * 9.0 - 330 * 0.35, 330)
        ctx.stroke()
        ctx.restore()


def draw_burj(ctx, s, x_c):
    profile_path(ctx, BURJ, x_c, steps=True)
    g = cairo.LinearGradient(x_c - 32, 0, x_c + 32, 0)
    g.add_color_stop_rgb(0, 0.80, 0.87, 0.94)
    g.add_color_stop_rgb(0.45, 0.58, 0.68, 0.80)
    g.add_color_stop_rgb(1, 0.33, 0.40, 0.52)
    ctx.set_source(g)
    ctx.fill()
    if s * 828 > 200:
        ctx.set_source_rgba(1, 1, 1, 0.35)
        ctx.set_line_width(max(0.6, 1.0 / s))
        ctx.move_to(x_c - 6, 0)
        ctx.line_to(x_c - 1.5, 780)
        ctx.stroke()
        ctx.set_source_rgba(0.2, 0.25, 0.35, 0.35)                    # floor bands
        for y in range(20, 740, 24):
            w = next(wd for yt, wd in BURJ if y < yt)
            ctx.move_to(x_c - w, y)
            ctx.line_to(x_c + w, y)
        ctx.stroke()


# ---------------------------------------------------------------------- the scene
class Scene(ScaleScene):
    QUESTION = ("HOW MANY BRICKS", "UNTIL THE BOTTOM BREAKS?")
    UNIT = ("BRICK", "BRICKS")
    WHY = ("Each brick holds about 4,240 N", f"{ABOVE:,} bricks on top weigh that much")
    T_TOTAL = 24.5
    T_REV = 21.8
    GY = 1500.0
    AX = 515.0
    TOP_PX = 770.0
    S0, DRIFT = 40000.0, 0.01
    INTRO_ZOOM, INTRO_T = 0.14, 0.9
    ROOT = 47
    SKY_COOL = (hex_rgb("#4C97DE"), hex_rgb("#CBE6FA"))
    SKY_WARM = (hex_rgb("#D9663E"), hex_rgb("#FFD49A"))
    EQ = ((50, -8.0), (120, -3.5), (300, -1.0), (900, 0.0), (3000, 1.5), (8000, 0.5), (16000, -4.5))

    def __init__(self, seed):
        super().__init__(seed, ANSWER)
        self.logc = np.log(np.array(self.ev_c, dtype=float))
        rng = np.random.default_rng(seed * 13 + 3)
        # cracks on the bottom brick's front face, normalized (u across, v up)
        self.cracks = []
        starts = [(0.06, 0.0), (0.33, 0.0), (0.62, 0.0), (0.93, 0.0), (0.02, 0.85), (0.97, 0.7), (0.5, 1.0)]
        for i, (u, v) in enumerate(starts):
            pts = [(u, v)]
            du = rng.uniform(-0.10, 0.10) + (0.08 if u < 0.5 else -0.08)
            dv = -0.16 if v > 0.5 else 0.16
            for _ in range(6):
                u = clamp(u + du + rng.uniform(-0.07, 0.07), 0.02, 0.98)
                v = clamp(v + dv * rng.uniform(0.6, 1.3), 0.02, 0.98)
                pts.append((u, v))
            self.cracks.append((pts, 0.04 + 0.05 * i / len(starts)))
        # shards of the bottom brick at the reveal: (u0, u1, v0, v1, vx, vy, spin)
        us = [0.0, 0.3 + rng.uniform(-0.05, 0.05), 0.64 + rng.uniform(-0.05, 0.05), 1.0]
        vs = [0.0, 0.5 + rng.uniform(-0.08, 0.08), 1.0]
        self.shards = []
        for a in range(3):
            for b in range(2):
                cu = 0.5 * (us[a] + us[a + 1])
                self.shards.append((us[a], us[a + 1], vs[b], vs[b + 1], (cu - 0.5) * 900 + rng.uniform(-120, 120),
                                    -(260 + 300 * vs[b] + rng.uniform(0, 160)), rng.uniform(-9, 9)))
        self.dust = [(rng.uniform(-1, 1), rng.uniform(0.2, 1.0), rng.uniform(0.5, 1.0)) for _ in range(16)]
        self.stats = self.base_drama(brick_mass_g=M_BRICK * 1000, fail_load_n=F_FAIL, bricks_above=ABOVE,
                                     tower_height_m=round(TOWER_M, 1),
                                     note="answer = ceil(4240 / (0.00115 * 9.81)) + 1 (bottom brick)")

    # ------------------------------------------------------------ state over time
    def cam_n(self, t):
        if t >= self.T_REV - 0.3:
            return float(ANSWER)
        return math.exp(np.interp(t + 0.3, self.ev_t, self.logc))

    def world_height(self, t):
        return (self.cam_n(t) * BH + STUD_H + OY * BW + 0.002) * 1.06

    def strain(self, t):
        frac = (self.count_f(t) - 1.0) / (ANSWER - 1.0)
        return clamp(frac) ** 0.5, frac

    def draw(self, ctx, t, res):
        d = t - self.T_REV
        if 0 <= d < 0.7:
            amp = 16.0 * math.exp(-d / 0.16)
            ctx.save()
            ctx.translate(W / 2, H / 2)
            ctx.scale(1.03, 1.03)
            ctx.translate(-W / 2 + amp * math.sin(d * 93.0), -H / 2 + amp * math.cos(d * 71.0))
            super().draw(ctx, t, res)
            ctx.restore()
        else:
            super().draw(ctx, t, res)

    def draw_ground(self, ctx, t, s):
        k = self.warmth(t)
        gy = self.GY
        g = cairo.LinearGradient(0, gy, 0, H)
        g.add_color_stop_rgb(0, *mix((0.40, 0.64, 0.32), (0.54, 0.56, 0.28), k))
        g.add_color_stop_rgb(1, *mix((0.13, 0.27, 0.13), (0.23, 0.21, 0.11), k))
        ctx.set_source(g)
        ctx.rectangle(0, gy, W, H - gy)
        ctx.fill()
        ctx.set_source_rgba(0, 0, 0, 0.22)
        ctx.rectangle(0, gy, W, 2)
        ctx.fill()

    # ------------------------------------------------------------ world
    def _landmark(self, ctx, s, x_c, half_w, height, fn):
        xa, xb = self.AX + (x_c - half_w) * s, self.AX + (x_c + half_w) * s
        if xb < -5 or xa > W + 5 or height * s < 3:
            return
        ctx.save()
        ctx.translate(self.AX + x_c * s, self.GY)
        ctx.scale(s, -s)
        fn(ctx, 0.0)
        ctx.restore()

    def draw_clouds(self, ctx, s, t, front):
        for i, (x, y, w, f) in enumerate(CLOUDS):
            if f != front or w * s < 10:
                continue
            x = x + 6.0 * t
            sx, sy = self.AX + x * s, self.GY - y * s
            r = 0.5 * w * s
            if sx + r < 0 or sx - r > W or sy + r < 0 or sy - r > H:
                continue
            a = 0.55 if front else 0.85
            for j, (ox, oy, rr) in enumerate(((-0.55, 0.0, 0.45), (0.0, -0.12, 0.6), (0.5, 0.02, 0.48),
                                              (0.2, 0.1, 0.4), (-0.25, 0.12, 0.38))):
                cx, cy, rad = sx + ox * r, sy + oy * r, rr * r
                g = cairo.RadialGradient(cx - 0.3 * rad, cy - 0.35 * rad, 0.05 * rad, cx, cy, rad)
                g.add_color_stop_rgba(0, 1, 1, 1, a)
                g.add_color_stop_rgba(0.75, 0.96, 0.96, 0.98, 0.9 * a)
                g.add_color_stop_rgba(1, 0.9, 0.92, 0.96, 0.0)
                ctx.set_source(g)
                ctx.save()
                ctx.translate(cx, cy)
                ctx.scale(1.0, 0.62)
                ctx.arc(0, 0, rad, 0, TAU)
                ctx.restore()
                ctx.fill()

    def tower_geom(self, s):
        fw = max(BW * s, MINW)
        x0 = self.AX - 0.5 * fw * (1 + OX)
        return fw, x0

    def draw_tower(self, ctx, t, s):
        d = t - self.T_REV
        fw, x0 = self.tower_geom(s)
        bh = BH * s
        lw = clamp(0.012 * fw, 0.0, 3.0) if fw > 40 else 0.0
        e = self.event_index(t)
        n_landed = self.ev_c[e]
        sink = 0.0
        if d >= 0:
            sink = 0.75 * BH * ease_in(clamp(d / 0.18), 2)
        y_off0 = 0.0
        if t < 0.15:                                          # brick 1 lands on frame 0..4
            y_off0 = 0.35 * BH * (1 - (t / 0.15) ** 2)
        segs = []
        for i in range(e + 1):
            c0 = self.ev_c[i - 1] if i > 0 else 0
            segs.append((c0, self.ev_c[i], COLS[i % 3], 0.0))
        fall = None
        if e + 1 < len(self.ev_t):
            te = self.ev_t[e + 1]
            tf = min(0.22, 0.8 * (te - self.ev_t[e]))
            if t >= te - tf:
                u = clamp((t - (te - tf)) / tf)
                nb = self.ev_c[e + 1] - n_landed
                drop = max(0.6 * nb * BH, 150.0 / s) * (1 - u * u)
                fall = (n_landed, self.ev_c[e + 1], COLS[(e + 1) % 3], drop)
        ctx.save()
        if d > 0:
            ang = -0.13 * ease_in(clamp((d - 0.15) / 2.4), 2)
            ctx.translate(self.AX, self.GY)
            ctx.rotate(ang)
            ctx.translate(-self.AX, -self.GY)
        # contact shadow
        g = cairo.RadialGradient(self.AX, self.GY, 0, self.AX, self.GY, 0.75 * fw * (1 + OX) + 8)
        g.add_color_stop_rgba(0, 0, 0, 0, 0.45)
        g.add_color_stop_rgba(1, 0, 0, 0, 0)
        ctx.save()
        ctx.translate(self.AX, self.GY)
        ctx.scale(1.0, 0.18)
        ctx.translate(-self.AX, -self.GY)
        ctx.arc(self.AX, self.GY, 0.75 * fw * (1 + OX) + 8, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
        ctx.restore()
        allsegs = segs + ([fall] if fall else [])
        for j, (c0, c1, col, dr) in enumerate(allsegs):
            off = (y_off0 if c0 == 0 else 0.0) - sink + dr
            ya = self.GY - (c0 * BH + off) * s
            yb = self.GY - (c1 * BH + off) * s
            if ya < -60 or yb > H + 60:
                continue
            ya_c, yb_c = min(ya, H + 60), max(yb, -60.0)
            slab(ctx, x0, ya_c, fw, ya_c - yb_c, col, lw)
            if dr > 0 or j == len(segs) - 1:
                if yb > -60:
                    top_face(ctx, x0, yb, fw, col, lw)
            # seams between bricks inside the segment
            if bh >= 3.5:
                j_lo = max(c0 + 1, int((self.GY - off * s - (H + 60)) / bh))
                j_hi = min(c1 - 1, int((self.GY - off * s + 60) / bh) + 1)
                if j_hi >= j_lo and j_hi - j_lo < 400:
                    ctx.set_line_width(clamp(0.05 * bh, 1.0, 3.5))
                    ctx.set_source_rgba(0, 0, 0, 0.30)
                    for jj in range(j_lo, j_hi + 1):
                        y = self.GY - (jj * BH + off) * s
                        ctx.move_to(x0, y)
                        ctx.line_to(x0 + fw, y)
                        ctx.line_to(x0 + fw * (1 + OX), y - OY * fw)
                    ctx.stroke()
            elif c0 > 0 and ya - yb > 1.5:
                ctx.set_line_width(1.0)
                ctx.set_source_rgba(0, 0, 0, 0.25)
                ctx.move_to(x0, ya)
                ctx.line_to(x0 + fw * (1 + OX), ya - OY * fw)
                ctx.stroke()
        # glossy strip down the whole front
        top_y = self.GY - (n_landed * BH - sink) * s
        ytop = max(top_y, -60.0)
        g = cairo.LinearGradient(x0, 0, x0 + fw, 0)
        g.add_color_stop_rgba(0.06, 1, 1, 1, 0.0)
        g.add_color_stop_rgba(0.12, 1, 1, 1, 0.34)
        g.add_color_stop_rgba(0.20, 1, 1, 1, 0.0)
        ctx.rectangle(x0, ytop, fw, self.GY - ytop - y_off0 * s)
        ctx.set_source(g)
        ctx.fill()
        # glint where the last batch landed
        age = t - self.ev_t[e]
        if e > 0 and age < 0.25 and d < 0 and top_y > -40:
            ctx.save()
            ctx.set_operator(cairo.OPERATOR_ADD)
            glow(ctx, x0 + 0.5 * fw * (1 + OX), top_y, max(40.0, 0.9 * fw), (1.0, 0.95, 0.8),
                 0.55 * math.exp(-age / 0.08))
            ctx.restore()
        ctx.restore()
        return fw, x0

    def draw_dust(self, ctx, t, s, fw):
        d = t - self.T_REV
        if d < 0 or d > 2.6:
            return
        for i, (dx, up, sz) in enumerate(self.dust):
            k = ease_out(clamp(d / 1.6), 2)
            x = self.AX + dx * (60 + 230 * k) * sz
            y = self.GY - up * 90 * k - 6
            r = (25 + 110 * k) * sz
            a = 0.5 * (1 - clamp(d / 2.6)) * sz
            g = cairo.RadialGradient(x, y, 0, x, y, r)
            g.add_color_stop_rgba(0, 0.88, 0.80, 0.68, a)
            g.add_color_stop_rgba(1, 0.88, 0.80, 0.68, 0)
            ctx.set_source(g)
            ctx.arc(x, y, r, 0, TAU)
            ctx.fill()
        for i in range(14):                                        # brick bits
            h1, h2 = hash1(i, 4.0), hash1(i, 9.0)
            vx = (h1 - 0.5) * 520
            vy = -(220 + 260 * h2)
            x = self.AX + vx * d
            y = self.GY - 4 + vy * d + 0.5 * 1400 * d * d
            if y > self.GY + 4:
                continue
            ctx.rectangle(x - 3, y - 2, 6 + 4 * h2, 4 + 3 * h1)
            ctx.set_source_rgb(*COLS[0])
            ctx.fill()

    def draw_inset(self, ctx, t, s):
        fw_main = max(BW * s, MINW)
        a = clamp((150.0 - BH * s) / 45.0)
        if a <= 0.0:
            return
        d = t - self.T_REV
        st, frac = self.strain(t)
        if d >= 0:
            st = 1.0
        ctx.save()
        # connector to the tower base
        fw, x0 = self.tower_geom(s)
        bx, by = self.AX + 0.5 * fw * OX, self.GY - 0.5 * BH * s
        ang = math.atan2(by - CYI, bx - CXI)
        ctx.move_to(CXI + RI * math.cos(ang), CYI + RI * math.sin(ang))
        ctx.line_to(bx, by)
        ctx.set_source_rgba(1, 1, 1, 0.8 * a)
        ctx.set_line_width(3.0)
        ctx.stroke()
        ctx.arc(bx, by, 7, 0, TAU)
        ctx.set_source_rgba(1, 1, 1, 0.9 * a)
        ctx.set_line_width(3.0)
        ctx.stroke()
        # body
        ctx.push_group()
        ctx.arc(CXI, CYI, RI, 0, TAU)
        ctx.clip()
        g = cairo.LinearGradient(0, CYI - RI, 0, CYI + RI)
        g.add_color_stop_rgb(0, 0.78, 0.87, 0.95)
        g.add_color_stop_rgb(1, 0.95, 0.96, 0.97)
        ctx.set_source(g)
        ctx.paint()
        yb0 = CYI + 72.0
        ctx.rectangle(CXI - RI, yb0, 2 * RI, RI)
        ctx.set_source_rgb(0.36, 0.56, 0.30)
        ctx.fill()
        h_i = FW_I * BH / BW
        hf, wf = 1.0 - 0.09 * st, 1.0 + 0.045 * st
        jit = 0.0
        if d < 0 and st > 0.45:
            jit = 2.5 * (st - 0.45) / 0.55 * math.sin(t * 61.0)
        x0i = CXI - 0.5 * FW_I * (1 + OX) + jit
        lw = 2.6
        # bricks above (they ride down as the bottom one squashes, then drop on the crush)
        drop = h_i * (1 - hf)
        if d >= 0:
            drop = h_i * (1 - hf) + h_i * 0.78 * ease_in(clamp(d / 0.16), 2)
            drop -= 6.0 * math.sin(clamp((d - 0.16) / 0.25) * math.pi) * math.exp(-d / 0.5)
        n_above = min(3, max(0, self.count(t) - 1))
        for j in range(1, n_above + 1):
            yb = yb0 - j * h_i + drop
            slab(ctx, x0i, yb, FW_I, h_i, COLS[j % 3], lw)
            if j == n_above:
                top_face(ctx, x0i, yb - h_i, FW_I, COLS[j % 3], lw)
        # the bottom brick
        col0 = mix(COLS[0], (1.0, 0.82, 0.80), 0.55 * st)
        fwb = FW_I * wf
        xb0 = CXI - 0.5 * FW_I * (1 + OX) - 0.5 * (fwb - FW_I) + jit
        hb = h_i * hf
        if d < 0:
            slab(ctx, xb0, yb0, fwb, hb, col0, lw)
            if n_above == 0:
                top_face(ctx, xb0, yb0 - hb, fwb, col0, lw)
            L = clamp((frac - 0.03) / 0.85) ** 0.55
            if L > 0:
                for pts, delay in self.cracks:
                    m = (len(pts) - 1) * clamp((L - delay) / (1 - delay))
                    if m <= 0:
                        continue
                    path = [pts[0]]
                    for q in range(1, len(pts)):
                        if q <= m:
                            path.append(pts[q])
                        else:
                            f = m - (q - 1)
                            p0, p1 = pts[q - 1], pts[q]
                            path.append((p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1])))
                            break
                    for wl, colc in ((3.6, (0.25, 0.03, 0.03, 0.95)), (1.2, (1, 0.95, 0.9, 0.55))):
                        for q, (u, v) in enumerate(path):
                            x, y = xb0 + u * fwb, yb0 - v * hb
                            if q == 0:
                                ctx.move_to(x, y)
                            else:
                                ctx.line_to(x, y)
                        ctx.set_line_width(wl)
                        ctx.set_source_rgba(*colc)
                        ctx.stroke()
        else:
            # rubble left behind, plus the shards flying out
            ctx.rectangle(xb0 - 10, yb0 - 0.22 * h_i + 0.0, fwb * (1 + OX) + 20, 0.22 * h_i)
            ctx.set_source_rgb(*darken(COLS[0], 0.25))
            ctx.fill()
            for (u0, u1, v0, v1, vx, vy, spin) in self.shards:
                cx = xb0 + 0.5 * (u0 + u1) * fwb + vx * d
                cy = yb0 - 0.5 * (v0 + v1) * hb + vy * d + 0.5 * 2400 * d * d
                if cy > CYI + RI + 80:
                    continue
                ww, hh = (u1 - u0) * fwb, (v1 - v0) * hb
                ctx.save()
                ctx.translate(cx, cy)
                ctx.rotate(spin * d)
                ctx.rectangle(-0.5 * ww, -0.5 * hh, ww, hh)
                ctx.set_source_rgb(*col0)
                ctx.fill_preserve()
                ctx.set_source_rgb(*darken(COLS[0], 0.6))
                ctx.set_line_width(2.2)
                ctx.stroke()
                ctx.restore()
            if d < 1.5:
                ctx.save()
                ctx.set_operator(cairo.OPERATOR_ADD)
                glow(ctx, CXI, yb0 - 0.3 * h_i, 160, (1.0, 0.9, 0.7), 0.8 * math.exp(-d / 0.18))
                ctx.restore()
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(a)
        # ring: white, turning hot as the load grows
        pulse = 0.5 + 0.5 * math.sin(t * (6.0 + 14.0 * st)) if st > 0.35 and d < 0 else 1.0
        ring = mix((1.0, 1.0, 1.0), (1.0, 0.25 + 0.2 * pulse, 0.15), clamp((st - 0.2) / 0.6))
        ctx.arc(CXI, CYI, RI, 0, TAU)
        ctx.set_source_rgba(0, 0, 0, 0.35 * a)
        ctx.set_line_width(14.0)
        ctx.stroke()
        ctx.arc(CXI, CYI, RI, 0, TAU)
        ctx.set_source_rgba(*ring, a)
        ctx.set_line_width(7.0 + 3.0 * st * pulse)
        ctx.stroke()
        ctx.restore()

    def draw_world(self, ctx, t, s, res):
        self.draw_clouds(ctx, s, t, 0)
        self._landmark(ctx, s, BURJ_X, 33, 828, lambda c, x: draw_burj(c, s, x))
        self._landmark(ctx, s, EIFFEL_X, 63, 330, lambda c, x: draw_eiffel(c, s, x))
        self._landmark(ctx, s, HOUSE_X + 4.5, 5.2, 9.2, lambda c, x: draw_house(c, x - 4.5))
        self._landmark(ctx, s, TREE_X, 3.4, 12.6, lambda c, x: draw_tree(c, x))
        self._landmark(ctx, s, PERSON_X, 0.4, 1.9, lambda c, x: draw_person(c, x))
        fw, x0 = self.draw_tower(ctx, t, s)
        self.draw_clouds(ctx, s, t, 1)
        self.draw_dust(ctx, t, s, fw)
        self.draw_inset(ctx, t, s)

    # ------------------------------------------------------------ sound
    def event_sound(self, i, t, k, rng):
        """Plastic clacks (more, higher and faster for bigger batches), and the bottom brick creaking under load."""
        nb = self.ev_c[i] - self.ev_c[i - 1]
        f0 = 950.0 * 2 ** (1.1 * k) * (1 + 0.05 * (hash1(i, 3.0) - 0.5))
        m = 1 + int(min(6, 1.1 * math.log10(max(1, nb)) + 2.5 * k))
        out = []
        for j in range(m):
            f = f0 * (1.0 + 0.09 * j) * (1 + 0.06 * (hash1(i, j) - 0.5))
            out.append((clack(rng, f), (0.24 + 0.08 * k) / (1 + 0.28 * j), 0.5 * (hash1(j, i) - 0.5),
                        (0.030 - 0.012 * k) * j))
        out.append((au.glide(260.0 * 2 ** (0.8 * k), 170.0, 0.07, decay=0.03, harm=0.4), 0.14 + 0.08 * k, 0.0, 0.0))
        frac = (self.ev_c[i] - 1) / (ANSWER - 1)
        st = frac ** 0.5
        if st > 0.06:                                               # the bottom brick creaks under the load
            f = 150.0 + 230.0 * st
            sr = au.SR
            n = int(0.32 * sr)
            tt = np.arange(n) / sr
            cr = au.glide(f, f * 0.78, 0.32, decay=0.14, harm=0.9) * (0.6 + 0.4 * np.sign(np.sin(TAU * (38 + 40 * st) * tt)))
            out.append((cr, min(0.30, 0.08 + 0.3 * st), -0.2, 0.05))
        return out

    def reveal_sound(self, fx, rng):
        T = self.T_REV
        sr = au.SR
        au.place(fx, clack(rng, 820.0, 0.1), 0.15, 0.28)                  # brick 1 lands on frame 4
        snap = norm(au.noise(rng, 0.09, 1200, 10000)) * np.exp(-np.arange(int(0.09 * sr)) / sr / 0.018)
        au.place(fx, snap, T - 0.005, 0.55)
        au.place(fx, au.tick(3300.0, 0.08), T, 0.30)
        au.place(fx, au.boom(rng, 1.6, 105.0, 34.0), T, 0.62)
        n = int(2.6 * sr)
        rum = au.swept_noise(rng, 2.6, 700, 120) * np.minimum(1.0, np.arange(n) / (0.05 * sr))
        rum *= np.exp(-np.arange(n) / sr / 1.1)
        au.place(fx, norm(rum), T + 0.02, 0.38)
        for j in range(16):                                               # bits of plastic landing
            tj = T + 0.18 + 1.5 * hash1(j, 2.0) ** 1.4
            au.place(fx, clack(rng, 1500 + 1800 * hash1(j, 5.0), 0.06), tj, 0.16 * (1 - 0.5 * hash1(j, 7.0)),
                     0.8 * (hash1(j, 8.0) - 0.5))
        for j, m_ in enumerate((69, 73, 76, 81)):                         # bright chord as the "why" line lands
            au.place(fx, au.bell(au.hz(m_), 1.6), T + 0.5 + 0.05 * j, 0.10)

    # ------------------------------------------------------------ metadata
    def metadata(self):
        title = "How Many Toy Bricks Until the Bottom One Breaks? \U0001F9F1"
        assert len(title) <= 60 and f"{ANSWER:,}" not in title
        desc = (
            "Comment your number \U0001F447 How many bricks did you guess?\n"
            "Plastic 2x2 toy bricks get stacked on a single brick, one at a time at first, then in bigger and bigger "
            "batches. The counter shows every brick in the tower.\n"
            "Each brick weighs 1.15 g, and the tower is drawn at its true height. The bottom brick carries the weight "
            "of every brick above it and gives way only when that weight beats the crushing load a real 2x2 brick "
            "survived in a lab load test. The circle shows the bottom brick up close as it strains. The person, the "
            "house, the Eiffel Tower, the Burj Khalifa and the clouds show how tall the tower gets as the camera pulls "
            "back.\n"
            "No footage, samples or stock media.\n"
            "#simulation #satisfying #physics")
        assert "le" "go" not in (title + desc).lower() and f"{ANSWER:,}" not in desc
        return dict(
            title=title,
            description=desc,
            tags=["toy bricks", "building bricks", "brick tower", "how many bricks", "tallest tower", "stacking",
                  "compression test", "plastic strength", "scale", "physics simulation", "exponential growth",
                  "prediction", "satisfying", "made with code", "simulation", "physics"],
            pinned_comment="Comment your number \U0001F447 How many toy bricks can you stack before the bottom one "
                           "breaks?",
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of cartoon toy bricks, buildings and "
                                              "sky; nothing realistic that could be mistaken for real people, places "
                                              "or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / self.FPS, 2),
            drama=self.stats,
        )
