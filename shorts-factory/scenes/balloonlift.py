"""BALLOON LIFT (SCALE family, episode 1): how many party balloons does it take to lift a car?

Helium party balloons are tied to a car's roof in escalating batches; the counter climbs; the camera pulls back past
a person, a tree and a house. The car leaves the ground only when the summed net lift beats its weight.

Physics (constants stated here, the answer is computed from them, nothing is tuned to a target):
  * balloon: standard 12-inch (30 cm) latex party balloon, a sphere of diameter 0.30 m -> volume 14.1 L;
  * air 1.204 kg/m^3 (20 C, sea level); balloon helium 0.180 kg/m^3 (slightly pressurized, ~98% pure);
  * gross lift = (1.204 - 0.180) kg/m^3 x 14.1 L = 14.5 g;
  * minus the latex (3.5 g) and the ribbon (1.0 g) -> net lift ~ 10.0 g per balloon;
  * car: a mid-size sedan, 1,500 kg;
  * balloons needed = ceil(1,500 kg / net lift) ~ 150,000 (exact value in ANSWER).
  * cluster size on screen is honest too: n balloons packed loosely (64%) make a ball of radius
    0.15 m x (n / 0.64) ** (1/3), about 9.4 m for the answer: as tall as a 6-storey building.
The seed only changes cosmetics (balloon colors, jitter, bob phases). Deterministic.
"""
import math

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import TAU, clamp, darken, ease_out, ease_out_back, hex_rgb, lighten, mix, ramp
from engine.scale import ScaleScene

# ---------------------------------------------------------------------- physics
D_BALLOON = 0.30            # m
RHO_AIR = 1.204             # kg/m^3
RHO_HE = 0.180              # kg/m^3, balloon-grade helium at slight overpressure
M_LATEX = 3.5               # g
M_RIBBON = 1.0              # g
CAR_KG = 1500.0
V_BALLOON = math.pi / 6 * D_BALLOON ** 3                       # m^3
GROSS_G = (RHO_AIR - RHO_HE) * V_BALLOON * 1000.0              # g
NET_G = GROSS_G - M_LATEX - M_RIBBON                           # g
ANSWER = int(math.ceil(CAR_KG * 1000.0 / NET_G))

# ---------------------------------------------------------------------- look / layout (metres)
CAR_TOP = 1.45
STRING_L = 1.8
SLOT_A = 0.158              # phyllotaxis spacing constant: slot k sits at radius SLOT_A * sqrt(k)
PACK = 0.64
COLORS = [hex_rgb(c) for c in ("#F2363F", "#FF8A1F", "#FFD21F", "#38C76A", "#2F8CFF", "#9B5CF6", "#FF5FA8", "#22C7D9")]
CAR_COL = hex_rgb("#C8102E")
PERSON_X, TREE_X, HOUSE_X = -3.4, -9.6, 6.2
LIFT_D = 2.6                # seconds of liftoff shown
LIFT_H = 7.0                # metres risen by the end of it


def cluster_radius(n):
    return 0.5 * D_BALLOON * (max(1.0, n) / PACK) ** (1.0 / 3.0)


def visible_slots(n):
    """Front-facing balloons drawn for a count n: every balloon while the bunch is small, then the balloons on the
    visible face of the honest ball (so the drawn cluster has the honest size)."""
    n = max(1.0, n)
    honest = (cluster_radius(n) / SLOT_A) ** 2
    return min(n, max(honest, 4.0 * math.sqrt(n)))


def score_seed(seed):
    return dict(seed=seed, score=1.0, answer=ANSWER)


def hash1(a, b=0.0):
    v = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - math.floor(v)


# ---------------------------------------------------------------------- drawing helpers (world metres, y up)
def draw_car(ctx, body_dy, tilt):
    ctx.save()
    ctx.rotate(tilt)
    # wheels first (body sits over the arches)
    for wx in (-1.45, 1.45):
        ctx.new_path()
        ctx.arc(wx, 0.33, 0.33, 0, TAU)
        ctx.set_source_rgb(0.08, 0.08, 0.10)
        ctx.fill()
        g = cairo.RadialGradient(wx - 0.05, 0.38, 0.02, wx, 0.33, 0.21)
        g.add_color_stop_rgb(0, 0.95, 0.96, 0.98)
        g.add_color_stop_rgb(0.7, 0.62, 0.65, 0.70)
        g.add_color_stop_rgb(1, 0.35, 0.37, 0.42)
        ctx.arc(wx, 0.33, 0.21, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
        ctx.set_source_rgb(0.25, 0.26, 0.30)
        ctx.set_line_width(0.035)
        for j in range(5):
            a = TAU * j / 5 + 0.3
            ctx.move_to(wx + 0.05 * math.cos(a), 0.33 + 0.05 * math.sin(a))
            ctx.line_to(wx + 0.19 * math.cos(a), 0.33 + 0.19 * math.sin(a))
        ctx.stroke()
        ctx.arc(wx, 0.33, 0.05, 0, TAU)
        ctx.set_source_rgb(0.2, 0.2, 0.24)
        ctx.fill()
    ctx.translate(0, body_dy)
    # body
    ctx.new_path()
    ctx.move_to(-2.30, 0.40)
    ctx.curve_to(-2.40, 0.60, -2.38, 0.86, -2.22, 0.95)
    ctx.line_to(-1.40, 1.00)
    ctx.curve_to(-1.15, 1.30, -0.95, 1.43, -0.55, 1.45)
    ctx.line_to(0.35, 1.45)
    ctx.curve_to(0.70, 1.43, 0.95, 1.20, 1.25, 1.02)
    ctx.curve_to(1.70, 0.98, 2.15, 0.93, 2.30, 0.80)
    ctx.curve_to(2.41, 0.66, 2.41, 0.50, 2.32, 0.40)
    ctx.line_to(1.90, 0.36)
    ctx.arc(1.45, 0.33, 0.43, 0.05, math.pi - 0.05)
    ctx.line_to(-1.02, 0.36)
    ctx.arc(-1.45, 0.33, 0.43, 0.05, math.pi - 0.05)
    ctx.close_path()
    g = cairo.LinearGradient(0, 0.32, 0, 1.45)
    g.add_color_stop_rgb(0, *darken(CAR_COL, 0.55))
    g.add_color_stop_rgb(0.45, *CAR_COL)
    g.add_color_stop_rgb(0.62, *lighten(CAR_COL, 0.35))
    g.add_color_stop_rgb(0.70, *darken(CAR_COL, 0.15))
    g.add_color_stop_rgb(1, *lighten(CAR_COL, 0.15))
    ctx.set_source(g)
    ctx.fill_preserve()
    ctx.set_source_rgb(*darken(CAR_COL, 0.65))
    ctx.set_line_width(0.035)
    ctx.stroke()
    # windows
    ctx.new_path()
    ctx.move_to(-1.20, 1.03)
    ctx.curve_to(-1.02, 1.28, -0.88, 1.37, -0.52, 1.38)
    ctx.line_to(0.32, 1.38)
    ctx.curve_to(0.62, 1.36, 0.86, 1.18, 1.10, 1.03)
    ctx.close_path()
    g = cairo.LinearGradient(-1.2, 1.38, 1.1, 1.03)
    g.add_color_stop_rgb(0, 0.12, 0.18, 0.28)
    g.add_color_stop_rgb(0.5, 0.30, 0.42, 0.55)
    g.add_color_stop_rgb(1, 0.10, 0.14, 0.22)
    ctx.set_source(g)
    ctx.fill()
    ctx.move_to(-0.35, 1.37)
    ctx.line_to(-0.05, 1.37)
    ctx.line_to(-0.40, 1.05)
    ctx.line_to(-0.62, 1.05)
    ctx.close_path()
    ctx.set_source_rgba(1, 1, 1, 0.22)
    ctx.fill()
    ctx.rectangle(-0.16, 1.02, 0.09, 0.37)                       # B-pillar
    ctx.set_source_rgb(*darken(CAR_COL, 0.5))
    ctx.fill()
    # door seams, handle, lights, highlight
    ctx.set_source_rgba(*darken(CAR_COL, 0.6), 0.8)
    ctx.set_line_width(0.022)
    for x in (-1.15, -0.11, 1.02):
        ctx.move_to(x, 0.42)
        ctx.line_to(x + 0.02, 1.0)
    ctx.stroke()
    for x in (-0.45, 0.60):
        ctx.rectangle(x, 0.84, 0.20, 0.045)
    ctx.set_source_rgb(*lighten(CAR_COL, 0.5))
    ctx.fill()
    ctx.save()
    ctx.translate(2.27, 0.74)
    ctx.scale(0.10, 0.07)
    ctx.arc(0, 0, 1, 0, TAU)
    ctx.restore()
    ctx.set_source_rgb(1.0, 0.96, 0.78)
    ctx.fill()
    ctx.rectangle(-2.36, 0.74, 0.08, 0.13)
    ctx.set_source_rgb(0.85, 0.08, 0.08)
    ctx.fill()
    ctx.move_to(-2.15, 0.90)
    ctx.curve_to(-1.0, 0.93, 1.0, 0.93, 2.15, 0.86)
    ctx.set_source_rgba(1, 1, 1, 0.45)
    ctx.set_line_width(0.03)
    ctx.stroke()
    ctx.restore()


def draw_person(ctx, x):
    ctx.save()
    ctx.translate(x, 0)
    ctx.set_source_rgb(0.16, 0.19, 0.27)
    for lx in (-0.11, 0.03):
        ctx.rectangle(lx, 0.0, 0.10, 0.86)
    ctx.fill()
    g = cairo.LinearGradient(-0.22, 0, 0.22, 0)
    g.add_color_stop_rgb(0, 0.15, 0.42, 0.62)
    g.add_color_stop_rgb(1, 0.08, 0.26, 0.42)
    ctx.set_source(g)
    ctx.move_to(-0.21, 0.84)
    ctx.line_to(0.21, 0.84)
    ctx.line_to(0.23, 1.43)
    ctx.curve_to(0.1, 1.48, -0.1, 1.48, -0.23, 1.43)
    ctx.close_path()
    ctx.fill()
    ctx.set_line_width(0.09)
    ctx.set_source_rgb(0.12, 0.36, 0.55)
    ctx.move_to(-0.24, 1.40)
    ctx.line_to(-0.30, 0.92)
    ctx.move_to(0.24, 1.40)
    ctx.line_to(0.34, 1.80)                                       # waving / pointing up
    ctx.stroke()
    ctx.set_source_rgb(0.88, 0.70, 0.55)
    ctx.arc(0.0, 1.60, 0.12, 0, TAU)
    ctx.fill()
    ctx.arc(0.35, 1.83, 0.05, 0, TAU)
    ctx.fill()
    ctx.set_source_rgb(0.20, 0.14, 0.10)
    ctx.arc(0.0, 1.65, 0.12, 0.1, math.pi - 0.1)
    ctx.fill()
    ctx.restore()


def draw_tree(ctx, x):
    ctx.save()
    ctx.translate(x, 0)
    g = cairo.LinearGradient(-0.35, 0, 0.35, 0)
    g.add_color_stop_rgb(0, 0.40, 0.26, 0.15)
    g.add_color_stop_rgb(1, 0.24, 0.15, 0.08)
    ctx.set_source(g)
    ctx.move_to(-0.38, 0)
    ctx.line_to(-0.22, 6.5)
    ctx.line_to(0.22, 6.5)
    ctx.line_to(0.38, 0)
    ctx.close_path()
    ctx.fill()
    for cx, cy, r in ((-1.6, 6.8, 2.4), (1.7, 7.0, 2.3), (0.0, 7.6, 3.0), (-0.9, 9.6, 2.3), (1.0, 9.8, 2.1),
                      (0.0, 11.0, 1.6)):
        g = cairo.RadialGradient(cx - 0.5 * r, cy + 0.5 * r, 0.1 * r, cx, cy, r)
        g.add_color_stop_rgb(0, 0.46, 0.74, 0.32)
        g.add_color_stop_rgb(0.7, 0.20, 0.50, 0.20)
        g.add_color_stop_rgb(1, 0.12, 0.34, 0.14)
        ctx.set_source(g)
        ctx.arc(cx, cy, r, 0, TAU)
        ctx.fill()
    ctx.restore()


def draw_house(ctx, x0):
    ctx.save()
    ctx.translate(x0, 0)
    g = cairo.LinearGradient(0, 0, 9, 0)
    g.add_color_stop_rgb(0, 0.96, 0.89, 0.76)
    g.add_color_stop_rgb(1, 0.80, 0.70, 0.56)
    ctx.set_source(g)
    ctx.rectangle(0, 0, 9.0, 5.4)
    ctx.fill()
    ctx.set_source_rgb(0.45, 0.30, 0.25)
    ctx.rectangle(6.6, 6.8, 0.8, 2.4)                             # chimney
    ctx.fill()
    g = cairo.LinearGradient(0, 5.4, 0, 9.0)
    g.add_color_stop_rgb(0, 0.42, 0.18, 0.14)
    g.add_color_stop_rgb(1, 0.62, 0.28, 0.20)
    ctx.set_source(g)
    ctx.move_to(-0.6, 5.3)
    ctx.line_to(4.5, 9.0)
    ctx.line_to(9.6, 5.3)
    ctx.close_path()
    ctx.fill()
    ctx.set_source_rgb(0.35, 0.22, 0.16)
    ctx.rectangle(1.2, 0, 1.1, 2.2)                               # door
    ctx.fill()
    for wx, wy in ((3.6, 2.6), (6.2, 2.6), (3.6, 0.6), (6.2, 0.6)):
        ctx.rectangle(wx, wy, 1.5, 1.4)
        ctx.set_source_rgb(0.95, 0.95, 0.98)
        ctx.fill()
        ctx.rectangle(wx + 0.1, wy + 0.1, 1.3, 1.2)
        g = cairo.LinearGradient(0, wy, 0, wy + 1.4)
        g.add_color_stop_rgb(0, 0.40, 0.55, 0.70)
        g.add_color_stop_rgb(1, 0.70, 0.82, 0.92)
        ctx.set_source(g)
        ctx.fill()
    ctx.restore()


def make_sprite(col, w):
    """A glossy balloon, w px wide. Returns (surface, knot_x, knot_y)."""
    w = max(3, int(w))
    h = 1.15 * w
    sw, sh = w + 4, int(h + 0.18 * w + 5)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, sw, sh)
    c = cairo.Context(surf)
    cx, cy = sw / 2, 2 + h / 2
    c.save()
    c.translate(cx, cy)
    c.scale(w / 2, h / 2)
    c.arc(0, 0, 1, 0, TAU)
    c.restore()
    g = cairo.RadialGradient(cx - 0.22 * w, cy - 0.25 * h, 0.03 * w, cx, cy, 0.62 * h)
    g.add_color_stop_rgb(0, *lighten(col, 0.55))
    g.add_color_stop_rgb(0.40, *col)
    g.add_color_stop_rgb(1, *darken(col, 0.50))
    c.set_source(g)
    c.fill()
    if w >= 7:
        c.save()
        c.translate(cx - 0.20 * w, cy - 0.24 * h)
        c.rotate(-0.6)
        c.scale(0.12 * w, 0.07 * w)
        c.arc(0, 0, 1, 0, TAU)
        c.restore()
        c.set_source_rgba(1, 1, 1, 0.85)
        c.fill()
    ky = cy + h / 2
    c.move_to(cx, ky - 1)
    c.line_to(cx - 0.07 * w - 0.6, ky + 0.12 * w + 0.6)
    c.line_to(cx + 0.07 * w + 0.6, ky + 0.12 * w + 0.6)
    c.close_path()
    c.set_source_rgb(*darken(col, 0.3))
    c.fill()
    surf.flush()
    return surf, cx, ky + 0.12 * w


# ---------------------------------------------------------------------- the scene
class Scene(ScaleScene):
    QUESTION = ("HOW MANY BALLOONS", "TO LIFT A CAR?")
    UNIT = ("BALLOON", "BALLOONS")
    WHY = ("Each balloon lifts about 10 g", "1,500 kg = 150,000 balloons")
    ROOT = 45
    EQ = ((50, -9.0), (120, -4.5), (300, -1.0), (900, 0.0), (3000, 2.0), (8000, 1.0), (16000, -4.0))

    def __init__(self, seed):
        super().__init__(seed, ANSWER)
        rng = np.random.default_rng(seed * 7 + 1)
        self.mmax = int(visible_slots(ANSWER)) + 2
        k = np.arange(self.mmax)
        r = SLOT_A * np.sqrt(k + 0.5)
        th = k * 2.399963
        self.sx = r * np.cos(th) + rng.uniform(-0.025, 0.025, self.mmax)
        self.sy = r * np.sin(th) * 0.97 + rng.uniform(-0.025, 0.025, self.mmax)
        self.scol = rng.integers(0, len(COLORS), self.mmax)
        self.ssz = rng.uniform(0.93, 1.07, self.mmax)
        self.sph = rng.uniform(0, TAU, self.mmax)
        # birth time of each slot: the event that first makes it visible (staggered a little within the batch)
        self.sbirth = np.full(self.mmax, 1e9)
        prev = 0
        for i, (te, c) in enumerate(zip(self.ev_t, self.ev_c)):
            m = int(visible_slots(c))
            if i == 0:
                self.sbirth[:m] = -10.0
            elif m > prev:
                span = min(0.14, 0.6 * (te - self.ev_t[i - 1]))
                for j in range(prev, min(m, self.mmax)):
                    self.sbirth[j] = te + span * (j - prev) / max(1, m - prev)
            prev = max(prev, m)
        self.stats = self.base_drama(net_lift_g=round(NET_G, 2), gross_lift_g=round(GROSS_G, 2), car_kg=CAR_KG,
                                     cluster_diameter_m=round(2 * cluster_radius(ANSWER), 1))

    def episode_resources(self, res):
        res["sprites"] = {}

    # ------------------------------------------------------------ state over time
    def lift(self, t):
        d = t - self.T_REV
        return 0.0 if d <= 0 else LIFT_H * clamp(d / LIFT_D) ** 1.6

    def cluster_r(self, t):
        return SLOT_A * math.sqrt(max(1.0, visible_slots(self.count_f(t)))) + 0.17

    def world_height(self, t):
        return max(4.1, CAR_TOP + STRING_L + 2 * self.cluster_r(t) + 0.5) + self.lift(t)

    def sprite(self, res, ci, w):
        key = (int(ci), int(w))
        sp = res["sprites"].get(key)
        if sp is None:
            if len(res["sprites"]) > 600:
                res["sprites"].clear()
            sp = res["sprites"][key] = make_sprite(COLORS[ci], w)
        return sp

    # ------------------------------------------------------------ drawing
    def draw_world(self, ctx, t, s, res):
        d = t - self.T_REV
        k = self.progress(t)
        lift = self.lift(t)
        frac = min(1.0, self.count_f(t) * NET_G / (CAR_KG * 1000.0))
        body_dy = 0.06 * frac ** 2                                   # springs unload as the lift grows
        tilt = 0.0
        if d < 0 and frac > 0.7:                                     # it strains just before it goes
            q = (frac - 0.7) / 0.3
            body_dy += 0.02 * q * math.sin(t * 37.0)
            tilt = 0.012 * q * math.sin(t * 9.0)
        if d >= 0:
            tilt = 0.02 * math.sin(d * 3.1) * math.exp(-d / 1.6)
        # reference objects behind
        ctx.save()
        self.world_ctx(ctx, s)
        draw_tree(ctx, TREE_X)
        draw_house(ctx, HOUSE_X)
        # car shadow (soft, shrinks as it rises)
        sh = math.exp(-lift / 2.5)
        g = cairo.RadialGradient(0, 0, 0.2, 0, 0, 2.8)
        g.add_color_stop_rgba(0, 0, 0, 0, 0.45 * sh)
        g.add_color_stop_rgba(1, 0, 0, 0, 0)
        ctx.save()
        ctx.scale(1.0, 0.12)
        ctx.arc(0, 0, 2.8, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
        ctx.restore()
        draw_person(ctx, PERSON_X)
        ctx.restore()
        # strings, then the car, then the balloons
        R = self.cluster_r(t)
        sway = 0.10 * math.sin(0.9 * t) + 0.04 * math.sin(2.3 * t + 1.0)
        ccx = sway * min(1.0, 0.5 + R)
        ccy = CAR_TOP + body_dy + lift + STRING_L + R
        tie = self.to_screen(0.0, CAR_TOP + body_dy + lift, s)
        live = np.nonzero(self.sbirth <= t)[0]
        bob = 0.05 * np.sin(1.7 * t + self.sph[live])
        px = self.AX + (ccx + self.sx[live]) * s
        py = self.GY - (ccy + self.sy[live] + bob) * s
        bw = D_BALLOON * s * self.ssz[live]
        age = t - self.sbirth[live]
        grow = np.where(age < 0.25, [ease_out_back(clamp(a / 0.25), 2.2) for a in age], 1.0)
        bw = bw * grow
        knot_y = py + 0.5 * 1.15 * bw + 0.12 * bw
        ctx.set_line_width(max(1.0, 0.012 * s))
        nstr = len(live) if len(live) <= 60 else 60
        idx = range(len(live)) if len(live) <= 60 else range(0, 60)
        ctx.set_source_rgba(0.97, 0.97, 1.0, 0.75)
        for j in idx:
            x0, y0 = px[j], knot_y[j]
            mx = 0.5 * (x0 + tie[0]) + 0.08 * (x0 - tie[0])
            ctx.move_to(x0, y0)
            ctx.curve_to(mx, y0 + 0.4 * (tie[1] - y0), tie[0], tie[1] - 0.3 * (tie[1] - y0), tie[0], tie[1])
        ctx.stroke()
        ctx.save()
        self.world_ctx(ctx, s)
        ctx.translate(0, lift)
        draw_car(ctx, body_dy, tilt)
        ctx.restore()
        # tie knot on the roof
        ctx.arc(tie[0], tie[1], max(2.0, 0.06 * s), 0, TAU)
        ctx.set_source_rgb(0.95, 0.95, 0.95)
        ctx.fill()
        # balloons: outer (newest) first, inner on top
        for j in range(len(live) - 1, -1, -1):
            w = bw[j]
            if w < 2.5:
                continue
            surf, kx, ky = self.sprite(res, self.scol[live[j]], round(w))
            x = round(px[j] - kx)
            y = round(knot_y[j] - ky)
            ctx.set_source_surface(surf, x, y)
            ctx.rectangle(x, y, surf.get_width(), surf.get_height())
            ctx.fill()
        # dust puff at liftoff
        if 0 <= d < 1.2:
            for i in range(18):
                h1, h2 = hash1(i, 1.0), hash1(i, 2.0)
                wx = (-1.45 if i % 2 else 1.45) + (h1 - 0.5) * 0.8
                vx = (h1 - 0.5) * 4.0 + (-1.2 if wx < 0 else 1.2)
                x = wx + vx * ease_out(d / 1.2, 2)
                y = 0.15 + 0.9 * h2 * ease_out(d / 1.2, 2)
                r = (0.25 + 0.35 * h2) * (0.6 + 0.8 * d)
                sx_, sy_ = self.to_screen(x, y, s)
                a = 0.45 * (1 - d / 1.2)
                g = cairo.RadialGradient(sx_, sy_, 0, sx_, sy_, r * s)
                g.add_color_stop_rgba(0, 0.85, 0.78, 0.66, a)
                g.add_color_stop_rgba(1, 0.85, 0.78, 0.66, 0)
                ctx.set_source(g)
                ctx.arc(sx_, sy_, r * s, 0, TAU)
                ctx.fill()

    # ------------------------------------------------------------ sound
    def event_sound(self, i, t, k, rng):
        """A rubbery squeak (balloon tied on) whose pitch rises with the build, plus a soft bump."""
        f0 = 330.0 * 2 ** (1.7 * k) * (1 + 0.06 * hash1(i, 3.0))
        n_new = self.ev_c[i] - self.ev_c[i - 1]
        out = [(au.glide(f0, f0 * 1.45, 0.11, decay=0.06, harm=0.55), 0.22 - 0.06 * k, 0.3 * (hash1(i) - 0.5), 0.0),
               (au.tick(2200 + 900 * k, 0.05), 0.10, 0.0, 0.0)]
        if n_new > 1:                                             # batches: a quick second squeak, wider pan
            out.append((au.glide(f0 * 1.26, f0 * 1.8, 0.08, decay=0.04, harm=0.5), 0.13, -0.4 * (hash1(i, 5) - 0.5),
                        0.045))
        return out

    def reveal_sound(self, fx, rng):
        T = self.T_REV
        sr = au.SR
        tt = np.arange(int(0.9 * sr)) / sr
        creak = au.glide(150.0, 80.0, 0.9, decay=0.35, harm=0.9) * (1 + 0.5 * np.sin(TAU * 31 * tt))
        au.place(fx, creak, T - 0.02, 0.30)
        wh = au.swept_noise(rng, 2.2, 250, 2600) * np.minimum(1.0, np.arange(int(2.2 * sr)) / (0.3 * sr))
        wh *= np.exp(-np.arange(len(wh)) / sr / 1.2)
        au.place(fx, wh, T + 0.05, 0.40, 0.0)
        for j, m_ in enumerate((69, 74, 78, 81)):                # bright chord as the "why" line lands
            au.place(fx, au.bell(au.hz(m_), 1.6), T + 0.5 + 0.05 * j, 0.10)

    # ------------------------------------------------------------ metadata
    def metadata(self):
        title = "How Many Balloons to Lift a Car? \U0001F388"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "Comment your number \U0001F447\n"
                "Helium party balloons get tied to a 1,500 kg car, one at a time at first, then in bigger and bigger "
                "batches. The counter shows every balloon tied on.\n"
                "Each one is a standard 30 cm latex party balloon, and the car leaves the ground only when the total "
                "lift of all the balloons beats the weight of the car. The person, the tree and the house show the "
                "real size of the bunch as the camera pulls back.\n"
                "No footage, samples or stock media.\n"
                "#simulation #satisfying #physics"),
            tags=["balloons", "helium balloons", "how many balloons", "lift a car", "balloon car", "scale",
                  "physics simulation", "exponential growth", "prediction", "satisfying", "made with code",
                  "simulation", "physics"],
            pinned_comment="Comment your number \U0001F447 How many balloons does it take to lift a car?",
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of a cartoon car and balloons; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / self.FPS, 2),
            drama=self.stats,
        )
