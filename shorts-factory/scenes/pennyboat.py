"""PENNY BOAT (SCALE family, episode 2): how many pennies does it take to sink a boat?

US pennies are poured into a small aluminum jon boat floating on calm fresh water, in escalating batches; the counter
climbs, the pile grows and the boat sits lower and lower. It swamps only when the honest buoyancy says so: when the
boat plus pennies weigh more than the water the hull displaces with the waterline at the gunwale.

Physics (constants stated here, the answer is computed from them, nothing is tuned to a target):
  * penny: US cent (1982 and later), 2.5 g, 19.05 mm across, 1.52 mm thick -> envelope volume 0.433 cm^3;
  * boat: a 12-ft aluminum jon boat ("1236" class): length 3.66 m, bottom width 0.91 m, beam at the gunwale
    1.22 m (flared sides), depth 0.38 m, flat bottom with the bow raked up 0.24 m over its last 0.88 m
    (the hull shape replaces a block coefficient: volume is integrated over the drawn hull), boat with seats 50 kg;
  * fresh water 1000 kg/m^3, calm, no margin: the boat swamps the moment the waterline reaches the gunwale;
  * displacement at the gunwale = 1,396 kg (1.396 m^3), so the hull can carry 1,346 kg of pennies;
  * pennies needed = ceil(1,345.7 kg / 2.5 g) = 538,273 (exact value in ANSWER), about $5,383;
  * draft on screen at every moment = the depth at which the displaced water equals boat + pennies in the boat
    (inverted from the same hull integral; 1.9 cm empty), so the boat sinks exactly as the real mass says;
  * the pile is honest too: n pennies poured loose (packing fraction 0.60) make a ridge across the boat with
    20 degree slopes (angle of repose of smooth coins); at the answer it is 0.389 m^3, 2.0 m long and 0.36 m tall:
    it still fits under the 0.38 m gunwale.
The seed only changes cosmetics (coin jitter, glints, splash droplets). Deterministic.
"""
import bisect
import math

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import TAU, clamp, darken, ease_in, ease_out, glow, hex_rgb, lerp, lighten, mix, ramp
from engine.scale import H as SCR_H, W as SCR_W, ScaleScene

# ---------------------------------------------------------------------- physics
PENNY_G = 2.5
PENNY_D = 0.01905
PENNY_T = 0.00152
V_COIN = math.pi * (PENNY_D / 2) ** 2 * PENNY_T          # m^3
PACK = 0.60
REPOSE = math.radians(20.0)
RHO_W = 1000.0
L = 3.66                       # boat length, m
B0 = 0.91                      # bottom width
BEAM = 1.22                    # beam at the gunwale
HD = 0.38                      # depth (keel to gunwale)
X_RAKE = L / 2 - 0.88          # bow rake starts here (boat coords, u = 0 at mid-length)
BOW_RISE = 0.24
BOAT_KG = 50.0
INSET = (BEAM - B0) / 2        # flare per side
FLARE = INSET / HD
WI = 0.5 * (B0 + BEAM)         # mean inside width the pile ridge spans


def hb(u):
    """Height of the hull bottom above the keel at station u (flat, then the raked bow)."""
    q = np.clip((np.asarray(u, float) - X_RAKE) / (L / 2 - X_RAKE), 0.0, 1.0)
    return BOW_RISE * q ** 1.4


_U = np.linspace(-L / 2, L / 2, 2001)
_HB = hb(_U)
DRAFTS = np.linspace(0.0, HD, 761)


def _disp_kg(d):
    area = np.where(d > _HB, B0 * (d - _HB) + FLARE * (d * d - _HB * _HB), 0.0)
    return RHO_W * float(np.sum(0.5 * (area[1:] + area[:-1]) * np.diff(_U)))


DISP = np.array([_disp_kg(d) for d in DRAFTS])
CAPACITY_KG = DISP[-1] - BOAT_KG
ANSWER = int(math.ceil(CAPACITY_KG * 1000.0 / PENNY_G))


def draft_for(mass_kg):
    return float(np.interp(mass_kg, DISP, DRAFTS))


def pile_dims(n):
    """(half-length a, height h, bulk volume) of the honest ridge of n loose pennies."""
    vb = max(0.0, n) * V_COIN / PACK
    a = math.sqrt(vb / (WI * math.tan(REPOSE)))
    return a, a * math.tan(REPOSE), vb


PILE_A, PILE_H, PILE_V = pile_dims(ANSWER)
assert PILE_H < HD and PILE_A < 1.3, "the pile must fit in the hull"

# ---------------------------------------------------------------------- look / layout (metres)
OBX, OBY = -0.12, 1.00         # oblique projection of the depth axis z (0 = near side, BEAM = far side)
PILE_C = -0.25                 # pile centre along the boat
POUR_Z = BEAM / 2
TFALL = 0.45                   # s, ~1 m free fall from the bucket lip
HZ = 760.0                     # horizon on screen
DUCK = (0.90, -0.40)
DOCK_X, DOCK_Z, DOCK_Y = (-3.0, -1.35), (-2.3, -1.2), 0.45
PERSON = (-2.1, -1.6)
BUCKET = (300.0, 778.0)        # screen position of the pouring bucket (under the counter)
COIN_DRAW = 2.6               # falling coins are drawn 2.6x (drawing choice only; pile volume stays honest)
COPPER = hex_rgb("#C8693A")
COPPER_HI = hex_rgb("#FFC08A")
COPPER_LO = hex_rgb("#6E2E14")
ALU = hex_rgb("#B9C2CC")
WATER_HI = hex_rgb("#7FC4D8")
WATER_LO = hex_rgb("#0F4A63")


def score_seed(seed):
    return dict(seed=seed, score=1.0, answer=ANSWER)


def hash1(a, b=0.0):
    v = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - math.floor(v)


def clink(f, dur=0.16):
    """A small metallic coin clink: inharmonic partials, fast decay."""
    t = np.arange(int(dur * au.SR)) / au.SR
    x = np.zeros_like(t)
    for r, a, dc in ((1.0, 1.0, 0.055), (2.41, 0.55, 0.040), (3.93, 0.32, 0.026), (5.72, 0.18, 0.018)):
        x += a * np.sin(TAU * f * r * t + r) * np.exp(-t / dc)
    x *= np.minimum(1.0, t / 0.0008)
    return x / np.abs(x).max()


def norm(x):
    return x / max(1e-9, np.abs(x).max())


# ---------------------------------------------------------------------- the scene
class Scene(ScaleScene):
    QUESTION = ("HOW MANY PENNIES", "TO SINK A BOAT?")
    UNIT = ("PENNY", "PENNIES")
    WHY = ("Each penny weighs 2.5 g", f"the boat holds up {CAPACITY_KG:,.0f} kg = {ANSWER:,}")
    GY = 1270.0
    AX = 575.0
    S0, DRIFT = 285.0, 0.004
    INTRO_ZOOM, INTRO_T = 0.12, 0.9
    ROOT = 43
    SKY_COOL = (hex_rgb("#4F9BE0"), hex_rgb("#CDE8F8"))
    SKY_WARM = (hex_rgb("#D9683F"), hex_rgb("#FFD29A"))
    EQ = ((50, -9.0), (120, -4.5), (300, -1.0), (900, 0.0), (3000, 1.5), (8000, 0.0), (16000, -5.0))

    def __init__(self, seed):
        super().__init__(seed, ANSWER)
        rng = np.random.default_rng(seed * 11 + 5)
        # coins lying on the floor around the pour point (individual coins while there are few)
        self.nly = 140
        self.ly_u = PILE_C + rng.normal(0, 1, self.nly) * 0.05 * np.sqrt(np.arange(self.nly) + 1) ** 0.6
        self.ly_z = POUR_Z + rng.normal(0, 1, self.nly) * 0.04 * np.sqrt(np.arange(self.nly) + 1) ** 0.6
        self.ly_z = np.clip(self.ly_z, INSET + 0.03, BEAM - INSET - 0.03)
        self.ly_c = rng.uniform(0.75, 1.15, self.nly)
        # texture points on the pile slopes (p along slope, q across the boat)
        self.tx_p = rng.uniform(0, 1, 700)
        self.tx_q = rng.uniform(0, 1, 700)
        self.tx_c = rng.uniform(0.6, 1.25, 700)
        self.tx_g = rng.uniform(0, TAU, 700)
        self.drafts = [draft_for(BOAT_KG + c * PENNY_G / 1000.0) for c in self.ev_c]
        self.stats = self.base_drama(capacity_kg=round(CAPACITY_KG, 1), displacement_at_gunwale_kg=round(DISP[-1], 1),
                                     boat_kg=BOAT_KG, pile_m3=round(PILE_V, 3), pile_height_m=round(PILE_H, 3),
                                     pile_length_m=round(2 * PILE_A, 2), empty_draft_cm=round(100 * self.drafts[0], 1))

    # ------------------------------------------------------------ state over time
    def world_height(self, t):
        return 0.5

    def mass(self, t):
        return BOAT_KG + self.count_f(t) * PENNY_G / 1000.0

    def bob(self, t):
        y = 0.006 * math.sin(2.2 * t) + 0.003 * math.sin(3.7 * t + 1.0)
        e = self.event_index(t)
        for i in range(max(1, e - 6), e + 1):
            age = t - self.ev_t[i]
            if age < 0 or i == len(self.ev_t) - 1:
                continue
            amp = min(0.03, 0.003 + 0.8 * (self.drafts[i] - self.drafts[i - 1]) + 0.004 * self.progress(t))
            y -= amp * math.exp(-age / 0.45) * math.cos(10.0 * age)
        return y

    def sink(self, d):
        return 0.0 if d <= 0 else 1.9 * ease_in(clamp((d - 0.30) / 2.3), 2.2)

    def inner_water(self, d):
        return 0.0 if d <= 0 else HD * ease_out(clamp((d - 0.02) / 0.95), 1.6)

    def keel_y(self, t):
        d = t - self.T_REV
        dr = draft_for(self.mass(t)) if d < 0 else HD
        return -dr + (self.bob(t) if d < 0 else 0.0) - self.sink(d)

    def shake(self, t):
        d = t - self.T_REV
        k = self.progress(t)
        age, e = self.since_event(t)
        a = 1.2 * k * k + (3.5 * k * math.exp(-age / 0.08) if e > 0 and d < 0 else 0.0)
        if d >= 0:
            a = 9.0 * math.exp(-d / 0.45)
        return a * math.sin(t * 71.0), a * math.sin(t * 53.0 + 1.3)

    # ------------------------------------------------------------ projection
    def P(self, x, y, z):
        s = self._s
        return self.AX + (x + OBX * z) * s, self.GY - (y + OBY * z) * s

    def poly(self, ctx, pts):
        ctx.new_path()
        for i, (x, y, z) in enumerate(pts):
            sx, sy = self.P(x, y, z)
            if i == 0:
                ctx.move_to(sx, sy)
            else:
                ctx.line_to(sx, sy)
        ctx.close_path()

    # ------------------------------------------------------------ frame
    def draw(self, ctx, t, res):
        s = self.scale(t)
        self._s = s
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.draw_sky(ctx, t, s)
        self.draw_ground(ctx, t, s)
        ctx.save()
        ctx.translate(*self.shake(t))
        self.draw_world(ctx, t, s, res)
        ctx.restore()
        d = t - self.T_REV
        if d >= 0:
            self.draw_flash(ctx, t, s, d)
        self.draw_question(ctx, res["font"], t, d)
        self.draw_counter(ctx, res["font"], t, d)

    def draw_ground(self, ctx, t, s):
        """Far shore and the lake (the water plane, seen from above at an angle)."""
        k = self.warmth(t)
        # far shore: hills then a tree line
        ctx.new_path()
        ctx.move_to(0, HZ)
        for i in range(0, 23):
            x = i * 50
            ctx.line_to(x, HZ - 70 - 38 * math.sin(i * 0.55 + 0.7) - 18 * math.sin(i * 1.7))
        ctx.line_to(SCR_W, HZ)
        ctx.close_path()
        ctx.set_source_rgb(*mix((0.45, 0.60, 0.70), (0.70, 0.52, 0.48), k))
        ctx.fill()
        ctx.new_path()
        ctx.move_to(0, HZ + 2)
        for i in range(0, 55):
            x = i * 20
            ctx.line_to(x, HZ - 22 - 14 * hash1(i, 4.0) - 6 * math.sin(i * 0.9))
        ctx.line_to(SCR_W, HZ + 2)
        ctx.close_path()
        ctx.set_source_rgb(*mix((0.16, 0.34, 0.24), (0.30, 0.26, 0.16), k))
        ctx.fill()
        self.water_plane(ctx, t)

    def water_plane(self, ctx, t):
        k = self.warmth(t)
        top = mix(WATER_HI, (0.96, 0.72, 0.52), 0.55 * k)
        g = cairo.LinearGradient(0, HZ, 0, SCR_H)
        g.add_color_stop_rgb(0, *top)
        g.add_color_stop_rgb(0.35, *mix(mix(WATER_HI, WATER_LO, 0.45), (0.55, 0.42, 0.40), 0.35 * k))
        g.add_color_stop_rgb(1, *mix(WATER_LO, (0.20, 0.18, 0.25), 0.3 * k))
        ctx.set_source(g)
        ctx.rectangle(0, HZ, SCR_W, SCR_H - HZ)
        ctx.fill()
        # shimmer streaks drifting slowly (denser near the horizon)
        for i in range(46):
            h1, h2 = hash1(i, 7.0), hash1(i, 9.0)
            yy = HZ + 8 + (SCR_H - HZ) * h2 ** 1.7
            w = 30 + 140 * h1 * (0.4 + (yy - HZ) / (SCR_H - HZ))
            xx = (h1 * 1400 + 9 * t * (0.5 + h2)) % 1400 - 160
            a = 0.10 + 0.10 * math.sin(t * 1.3 + i)
            ctx.set_source_rgba(1, 1, 1, max(0.0, a))
            ctx.rectangle(xx, yy, w, 2.0 + 3.0 * h2)
            ctx.fill()

    # ------------------------------------------------------------ reference objects
    def draw_dock(self, ctx, t):
        x0, x1 = DOCK_X
        z0, z1 = DOCK_Z
        y = DOCK_Y
        for px in (x0 + 0.15, x1 - 0.15):                         # pilings
            for pz in (z0 + 0.1, z1 - 0.1):
                a, b = self.P(px, y, pz)
                c, e = self.P(px, -0.05, pz)
                g = cairo.LinearGradient(a - 9, 0, a + 9, 0)
                g.add_color_stop_rgb(0, 0.30, 0.20, 0.12)
                g.add_color_stop_rgb(1, 0.16, 0.10, 0.06)
                ctx.set_source(g)
                ctx.rectangle(a - 0.09 * self._s, b, 0.18 * self._s, e - b)
                ctx.fill()
        # deck front face and top planks
        self.poly(ctx, [(x0, y, z0), (x1, y, z0), (x1, y - 0.12, z0), (x0, y - 0.12, z0)])
        ctx.set_source_rgb(0.36, 0.24, 0.14)
        ctx.fill()
        n = 7
        for i in range(n):
            za, zb = lerp(z0, z1, i / n), lerp(z0, z1, (i + 0.93) / n)
            self.poly(ctx, [(x0, y, za), (x1, y, za), (x1, y, zb), (x0, y, zb)])
            c = mix((0.72, 0.53, 0.33), (0.58, 0.41, 0.24), hash1(i, 3.0))
            g = cairo.LinearGradient(*self.P(x0, y, za), *self.P(x1, y, za))
            g.add_color_stop_rgb(0, *lighten(c, 0.12))
            g.add_color_stop_rgb(1, *darken(c, 0.15))
            ctx.set_source(g)
            ctx.fill()
        # the person on the dock (1.75 m), soft shadow first
        px, pz = PERSON
        sx, sy = self.P(px, y, pz)
        ctx.save()
        ctx.translate(sx + 6, sy)
        ctx.scale(1.0, 0.35)
        ctx.arc(0, 0, 0.32 * self._s, 0, TAU)
        ctx.restore()
        ctx.set_source_rgba(0, 0, 0, 0.25)
        ctx.fill()
        ctx.save()
        ctx.translate(sx, sy)
        ctx.scale(self._s, -self._s)
        self.draw_person(ctx, t)
        ctx.restore()

    def draw_person(self, ctx, t):
        ctx.set_source_rgb(0.16, 0.19, 0.27)
        for lx in (-0.11, 0.03):
            ctx.rectangle(lx, 0.0, 0.10, 0.86)
        ctx.fill()
        g = cairo.LinearGradient(-0.22, 0, 0.22, 0)
        g.add_color_stop_rgb(0, 0.86, 0.30, 0.22)
        g.add_color_stop_rgb(1, 0.60, 0.16, 0.12)
        ctx.set_source(g)
        ctx.move_to(-0.21, 0.84)
        ctx.line_to(0.21, 0.84)
        ctx.line_to(0.23, 1.43)
        ctx.curve_to(0.1, 1.48, -0.1, 1.48, -0.23, 1.43)
        ctx.close_path()
        ctx.fill()
        ctx.set_line_width(0.09)
        ctx.set_source_rgb(0.70, 0.20, 0.15)
        sw = 0.05 * math.sin(t * 2.0)
        ctx.move_to(-0.24, 1.40)
        ctx.line_to(-0.30, 0.92)
        ctx.move_to(0.24, 1.40)
        ctx.line_to(0.42 + sw, 1.10)                              # hand out over the water
        ctx.stroke()
        ctx.set_source_rgb(0.88, 0.70, 0.55)
        ctx.arc(0.0, 1.60, 0.12, 0, TAU)
        ctx.fill()
        ctx.arc(0.43 + sw, 1.08, 0.05, 0, TAU)
        ctx.fill()
        ctx.set_source_rgb(0.20, 0.14, 0.10)
        ctx.arc(0.0, 1.65, 0.12, 0.1, math.pi - 0.1)
        ctx.fill()

    def draw_duck(self, ctx, t):
        dx, dz = DUCK
        dx += 0.04 * math.sin(t * 0.7)
        bob = 0.012 * math.sin(t * 2.6 + 0.5)
        sx, sy = self.P(dx, bob, dz)
        s = self._s
        for r in range(3):                                         # its own little ripples
            ph = (t * 0.6 + r / 3.0) % 1.0
            ctx.save()
            ctx.translate(sx, sy + 2)
            ctx.scale(1.0, OBY * 0.45)
            ctx.arc(0, 0, (0.20 + 0.35 * ph) * s, 0, TAU)
            ctx.restore()
            ctx.set_source_rgba(1, 1, 1, 0.35 * (1 - ph))
            ctx.set_line_width(2.0)
            ctx.stroke()
        ctx.save()
        ctx.translate(sx, sy)
        ctx.scale(s, -s)
        # body
        ctx.save()
        ctx.translate(0, 0.07)
        ctx.scale(0.17, 0.085)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        g = cairo.LinearGradient(0, 0.0, 0, 0.16)
        g.add_color_stop_rgb(0, 0.42, 0.34, 0.24)
        g.add_color_stop_rgb(1, 0.70, 0.60, 0.46)
        ctx.set_source(g)
        ctx.fill()
        ctx.move_to(-0.17, 0.09)
        ctx.line_to(-0.24, 0.14)
        ctx.line_to(-0.15, 0.12)
        ctx.close_path()
        ctx.fill()
        # head (green mallard) and beak
        ctx.arc(0.13, 0.19, 0.055, 0, TAU)
        g = cairo.RadialGradient(0.11, 0.21, 0.005, 0.13, 0.19, 0.06)
        g.add_color_stop_rgb(0, 0.35, 0.75, 0.45)
        g.add_color_stop_rgb(1, 0.05, 0.32, 0.18)
        ctx.set_source(g)
        ctx.fill()
        ctx.rectangle(0.09, 0.125, 0.07, 0.02)
        ctx.set_source_rgb(0.95, 0.95, 0.92)
        ctx.fill()
        ctx.move_to(0.18, 0.20)
        ctx.line_to(0.25, 0.185)
        ctx.line_to(0.18, 0.17)
        ctx.close_path()
        ctx.set_source_rgb(0.98, 0.72, 0.15)
        ctx.fill()
        ctx.arc(0.145, 0.205, 0.009, 0, TAU)
        ctx.set_source_rgb(0, 0, 0)
        ctx.fill()
        ctx.restore()

    # ------------------------------------------------------------ coins
    def coin(self, ctx, sx, sy, r, squash, rot, shade, glint):
        ctx.save()
        ctx.translate(sx, sy)
        ctx.rotate(rot)
        ctx.scale(r, max(0.15, squash) * r)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        ctx.set_source_rgb(*mix(COPPER_LO, COPPER_HI, clamp(0.35 + 0.35 * shade + 0.3 * glint)))
        ctx.fill()
        if glint > 0.6:
            a = (glint - 0.6) / 0.4
            ctx.set_source_rgba(1.0, 0.95, 0.80, 0.85 * a)
            L_ = 3.0 + 7.0 * a
            ctx.set_line_width(1.4)
            ctx.move_to(sx - L_, sy)
            ctx.line_to(sx + L_, sy)
            ctx.move_to(sx, sy - L_)
            ctx.line_to(sx, sy + L_)
            ctx.stroke()

    def batch_n(self, i):
        return self.ev_c[i] - (self.ev_c[i - 1] if i > 0 else 0)

    def bucket_state(self, t):
        """Tilt (radians) and lip position (screen px) of the pouring bucket. It tips further as the batches grow."""
        k = self.progress(t)
        age, e = self.since_event(t)
        d = t - self.T_REV
        jolt = 0.06 * math.exp(-age / 0.12) * (1 if e > 0 and d < 0 else 0)
        ang = math.radians(100.0 + 22.0 * k) + 0.025 * math.sin(2.4 * t) + jolt
        if d > 0:
            ang -= math.radians(30.0) * ease_out(clamp(d / 0.8), 2)      # done: tipped back upright-ish
        bx, by = BUCKET
        lx, ly = 62.0 * 1.25, -52.0 * 1.25                                             # lower lip corner in bucket coords
        ca, sa = math.cos(ang), math.sin(ang)
        return ang, (bx + lx * ca - ly * sa, by + lx * sa + ly * ca)

    def draw_bucket(self, ctx, t):
        ang, _ = self.bucket_state(t)
        bx, by = BUCKET
        ctx.save()
        ctx.translate(bx, by)
        ctx.rotate(ang)
        ctx.scale(1.25, 1.25)
        # soft shadow
        ctx.save()
        ctx.translate(8, 10)
        ctx.move_to(-62, -52)
        ctx.line_to(62, -52)
        ctx.line_to(46, 52)
        ctx.line_to(-46, 52)
        ctx.close_path()
        ctx.restore()
        ctx.set_source_rgba(0, 0, 0, 0.22)
        ctx.fill()
        # body (copper)
        ctx.move_to(-62, -52)
        ctx.line_to(62, -52)
        ctx.line_to(46, 52)
        ctx.line_to(-46, 52)
        ctx.close_path()
        g = cairo.LinearGradient(-62, 0, 62, 0)
        g.add_color_stop_rgb(0, *darken(COPPER, 0.45))
        g.add_color_stop_rgb(0.30, *lighten(COPPER, 0.25))
        g.add_color_stop_rgb(0.42, *lighten(COPPER_HI, 0.35))
        g.add_color_stop_rgb(0.60, *COPPER)
        g.add_color_stop_rgb(1, *darken(COPPER, 0.55))
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.18, 0.08, 0.04)
        ctx.set_line_width(4)
        ctx.stroke()
        for yy in (-30, 30):                                             # hoops
            w_ = 62 - (yy + 52) * 16 / 104
            ctx.move_to(-w_, yy)
            ctx.line_to(w_, yy)
            ctx.set_source_rgba(0.25, 0.10, 0.05, 0.8)
            ctx.set_line_width(5)
            ctx.stroke()
        # opening: rim ellipse full of pennies
        ctx.save()
        ctx.translate(0, -52)
        ctx.scale(62, 14)
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.restore()
        g = cairo.LinearGradient(-62, -52, 62, -52)
        g.add_color_stop_rgb(0, *COPPER_LO)
        g.add_color_stop_rgb(0.5, *COPPER_HI)
        g.add_color_stop_rgb(1, *COPPER)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.18, 0.08, 0.04)
        ctx.set_line_width(4)
        ctx.stroke()
        for j in range(7):
            gx = -45 + 90 * hash1(j, 31.0)
            a = max(0.0, math.sin(t * 3.1 + j * 2.1)) ** 6
            ctx.set_source_rgba(1, 0.95, 0.8, 0.9 * a)
            ctx.arc(gx, -52 + 8 * (hash1(j, 32.0) - 0.5), 3.0, 0, TAU)
            ctx.fill()
        # handle
        ctx.new_path()
        ctx.arc(0, -40, 70, math.pi + 0.25, TAU - 0.25)
        ctx.set_source_rgb(0.30, 0.32, 0.36)
        ctx.set_line_width(5)
        ctx.stroke()
        ctx.restore()

    def coin_path(self, t, tau, h1, h2, keel, ph, spread):
        """Screen position of a coin tau seconds after leaving the lip: a free-fall parabola onto the pile."""
        _, (lx, ly) = self.bucket_state(t - tau)
        x = PILE_C + (h1 - 0.5) * 2 * spread
        z = POUR_Z + (h2 - 0.5) * 2 * spread
        y_land = keel + ph * (1.0 - 0.6 * abs(h1 - 0.5)) + 0.01
        ex, ey = self.P(x, y_land, z)
        q = tau / TFALL
        return lerp(lx, ex, q), ly + (ey - ly) * q * q

    def draw_falling(self, ctx, t, keel):
        """The pour: each batch leaves the bucket lip TFALL before it lands (its event time). A copper stream whose
        width grows with the batch size, plus individual glinting coins."""
        s = self._s
        r = 0.5 * PENNY_D * s * COIN_DRAW
        i0 = bisect.bisect_right(self.ev_t, t)
        i1 = bisect.bisect_right(self.ev_t, t + TFALL + 0.2)
        _, ph_now, _ = pile_dims(self.count_f(t))
        # the stream (only for real batches)
        nb_next = self.batch_n(i0) if 1 <= i0 < len(self.ev_t) else 0
        sw = 0.0 if nb_next < 6 else min(30.0, 2.5 * math.log1p(nb_next))
        if sw > 0 and t < self.T_REV:
            pts = [self.coin_path(t, TFALL * q, 0.5, 0.5, keel, ph_now, 0.0) for q in np.linspace(0, 1, 14)]
            for wmul, col, a in ((1.0, COPPER_LO, 0.55), (0.55, COPPER, 0.75), (0.18, COPPER_HI, 0.8)):
                ctx.new_path()
                for j, (px, py) in enumerate(pts):
                    ctx.line_to(px, py)
                ctx.set_source_rgba(*col, a)
                ctx.set_line_width(sw * wmul)
                ctx.stroke()
        for i in range(max(1, i0), min(i1, len(self.ev_t))):
            te = self.ev_t[i]
            nb = self.batch_n(i)
            nv = int(min(40, 2 + 3.4 * math.log1p(nb))) if nb > 1 else 1
            spread = min(0.20, 0.015 + 0.012 * math.log1p(nb) ** 1.5)
            gap = te - self.ev_t[i - 1]
            _, ph, _ = pile_dims(self.ev_c[i - 1])
            for j in range(nv):
                h1, h2, h3 = hash1(i, j + 0.1), hash1(j, i + 0.7), hash1(i + j, 3.3)
                lag = (j / max(1, nv)) * min(0.16, 0.7 * gap)
                tau = t - (te + lag - TFALL)
                if tau < 0 or tau > TFALL:
                    continue
                sx, sy = self.coin_path(t, tau, h1, h2, keel, ph, spread)
                sx += (h3 - 0.5) * min(18.0, sw)
                c = math.cos(11.0 * tau + TAU * h3)
                self.coin(ctx, sx, sy, r, abs(c), 0.4 * h1, c, abs(c) ** 4)

    def draw_bounces(self, ctx, t, keel):
        """Coins skittering off the pile right after each batch lands, plus glints."""
        e = self.event_index(t)
        s = self._s
        r = 0.5 * PENNY_D * s
        _, ph, _ = pile_dims(self.count_f(t))
        for i in range(max(1, e - 2), e + 1):
            age = t - self.ev_t[i]
            if age < 0 or age > 0.40 or i == len(self.ev_t) - 1:
                continue
            nb = self.batch_n(i)
            m = int(min(16, 2 + 2.2 * math.log1p(nb)))
            for j in range(m):
                h1, h2 = hash1(i, j + 3.1), hash1(j + 9, i)
                vx = (h1 - 0.5) * (0.6 + 0.12 * math.log1p(nb))
                vy = 0.6 + 1.2 * h2
                x = PILE_C + vx * age
                y = keel + ph + 0.01 + vy * age - 4.9 * age * age
                if y < keel + 0.005:
                    continue
                sx, sy = self.P(x, y, POUR_Z + (h2 - 0.5) * 0.3)
                c = math.cos(20 * age + 6 * h1)
                self.coin(ctx, sx, sy, r * COIN_DRAW * 0.8, abs(c), h2, c, abs(c) ** 6 * (1 - age / 0.4))
            # copper flash on the pile where the batch lands (bigger for bigger batches)
            gx, gy = self.P(PILE_C, keel + ph, POUR_Z)
            ctx.save()
            ctx.set_operator(cairo.OPERATOR_ADD)
            glow(ctx, gx, gy, (14 + 9 * math.log1p(nb)) * (0.6 + age / 0.4 * 0.8), (1.0, 0.62, 0.30),
                 0.55 * (1 - age / 0.4))
            ctx.restore()

    # ------------------------------------------------------------ the boat
    def draw_boat(self, ctx, t, keel, d):
        s = self._s
        us = np.linspace(-L / 2, L / 2, 48)
        hbs = hb(us)
        Y = lambda v: keel + v                                       # noqa: E731
        far_g = [(u, Y(HD), BEAM) for u in us]
        far_c = [(u, Y(v), BEAM - INSET) for u, v in zip(us, hbs)]
        near_g = [(u, Y(HD), 0.0) for u in us]
        near_c = [(u, Y(v), INSET) for u, v in zip(us, hbs)]
        k = self.progress(t)
        # 1. far wall, inside (in shadow)
        self.poly(ctx, far_g + far_c[::-1])
        g = cairo.LinearGradient(*self.P(0, Y(HD), BEAM), *self.P(0, Y(0), BEAM - INSET))
        g.add_color_stop_rgb(0, *darken(ALU, 0.10))
        g.add_color_stop_rgb(1, *darken(ALU, 0.42))
        ctx.set_source(g)
        ctx.fill()
        ctx.set_line_width(max(1.0, 0.012 * s))
        ctx.set_source_rgba(0.25, 0.28, 0.32, 0.55)
        for u in np.arange(-L / 2 + 0.45, L / 2 - 0.2, 0.45):     # ribs
            ctx.move_to(*self.P(u, Y(HD), BEAM))
            ctx.line_to(*self.P(u, Y(float(hb(u))), BEAM - INSET))
        ctx.stroke()
        # 2. floor
        self.poly(ctx, far_c + near_c[::-1])
        g = cairo.LinearGradient(*self.P(-L / 2, Y(0), BEAM / 2), *self.P(L / 2, Y(0), BEAM / 2))
        g.add_color_stop_rgb(0, *darken(ALU, 0.22))
        g.add_color_stop_rgb(0.6, *darken(ALU, 0.08))
        g.add_color_stop_rgb(1, *darken(ALU, 0.25))
        ctx.set_source(g)
        ctx.fill()
        ctx.set_source_rgba(0.30, 0.33, 0.37, 0.5)
        for u in np.arange(-L / 2 + 0.3, X_RAKE + 0.5, 0.3):        # floor ribs
            v = float(hb(u))
            ctx.move_to(*self.P(u, Y(v), INSET))
            ctx.line_to(*self.P(u, Y(v), BEAM - INSET))
        ctx.stroke()
        # 3. bow transom, inside
        vb = float(hb(L / 2))
        self.poly(ctx, [(L / 2, Y(vb), INSET), (L / 2, Y(HD), 0), (L / 2, Y(HD), BEAM), (L / 2, Y(vb), BEAM - INSET)])
        ctx.set_source_rgb(*darken(ALU, 0.30))
        ctx.fill()
        # 4. seats (stern and bow benches)
        for uc in (-1.45, 1.22):
            for a_, col in ((0.30, None),):
                zi = INSET * (1 - a_ / HD)
                self.poly(ctx, [(uc - 0.13, Y(a_), zi), (uc + 0.13, Y(a_), zi), (uc + 0.13, Y(a_), BEAM - zi),
                                (uc - 0.13, Y(a_), BEAM - zi)])
                g = cairo.LinearGradient(*self.P(uc - 0.13, Y(a_), 0), *self.P(uc + 0.13, Y(a_), 0))
                g.add_color_stop_rgb(0, *lighten(ALU, 0.25))
                g.add_color_stop_rgb(1, *darken(ALU, 0.15))
                ctx.set_source(g)
                ctx.fill()
                self.poly(ctx, [(uc - 0.13, Y(a_), zi), (uc + 0.13, Y(a_), zi), (uc + 0.13, Y(a_ - 0.04), zi),
                                (uc - 0.13, Y(a_ - 0.04), zi)])
                ctx.set_source_rgb(*darken(ALU, 0.35))
                ctx.fill()
        # 5. pennies: loose coins on the floor, then the honest pile
        n = self.count_f(t)
        a, h, _ = pile_dims(n)
        r = 0.5 * PENNY_D * s
        nl = int(min(self.nly, n))
        for j in range(nl):
            u, z = self.ly_u[j], self.ly_z[j]
            if abs(u - PILE_C) < 0.7 * a:
                continue
            sx, sy = self.P(u, Y(0.0) + 0.002, z)
            gl = max(0.0, math.sin(t * 2.3 + j * 1.7)) ** 12
            self.coin(ctx, sx, sy, r, OBY * 0.75, 0.0, self.ly_c[j] - 0.4, gl)
        if a * s > 3.0:
            self.draw_pile(ctx, t, keel, a, h)
        # 6. water coming in at the reveal
        wi = self.inner_water(d)
        if wi > 0.002:
            zi = INSET * (1 - wi / HD)
            ui = [u for u in us if hb(u) < wi] + [L / 2]
            top = [(u, Y(wi), zi) for u in ui] + [(u, Y(wi), BEAM - zi) for u in ui[::-1]]
            self.poly(ctx, top)
            g = cairo.LinearGradient(*self.P(0, Y(wi), BEAM), *self.P(0, Y(wi), 0))
            g.add_color_stop_rgba(0, *mix(WATER_HI, WATER_LO, 0.2), 0.80)
            g.add_color_stop_rgba(1, *mix(WATER_HI, WATER_LO, 0.55), 0.92)
            ctx.set_source(g)
            ctx.fill()
            # sheets of water pouring over the far gunwale and down the inside of the far wall
            if d < 1.3:
                aa = 1.0 - ramp(d, 0.8, 1.3)
                for j in range(14):
                    u = -L / 2 + 0.2 + (L - 0.4) * (j + 0.5 * hash1(j, 2.0)) / 14
                    x0, y0 = self.P(u, Y(HD) + 0.02, BEAM)
                    x1, y1 = self.P(u + 0.05, Y(wi), BEAM - zi)
                    ctx.set_source_rgba(0.88, 0.96, 1.0, 0.75 * aa)
                    ctx.set_line_width((0.06 + 0.05 * hash1(j, 5.0)) * s)
                    ctx.move_to(x0, y0)
                    ctx.curve_to(x0, y0 + 0.3 * (y1 - y0), x1, y1 - 0.3 * (y1 - y0), x1, y1)
                    ctx.stroke()
        # 7. stern transom, outside
        self.poly(ctx, [(-L / 2, Y(0), INSET), (-L / 2, Y(HD), 0), (-L / 2, Y(HD), BEAM), (-L / 2, Y(0), BEAM - INSET)])
        g = cairo.LinearGradient(*self.P(-L / 2, Y(HD), 0), *self.P(-L / 2, Y(0), BEAM))
        g.add_color_stop_rgb(0, *darken(ALU, 0.25))
        g.add_color_stop_rgb(1, *darken(ALU, 0.50))
        ctx.set_source(g)
        ctx.fill()
        # 8. near wall, outside (brushed aluminum)
        self.poly(ctx, near_g + near_c[::-1])
        x0, y0 = self.P(0, Y(HD), 0)
        x1, y1 = self.P(0, Y(0), INSET)
        g = cairo.LinearGradient(x0, y0, x1, y1)
        g.add_color_stop_rgb(0, *lighten(ALU, 0.45))
        g.add_color_stop_rgb(0.25, *lighten(ALU, 0.10))
        g.add_color_stop_rgb(0.55, *ALU)
        g.add_color_stop_rgb(1, *darken(ALU, 0.35))
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.save()
        ctx.clip()
        for j in range(9):                                           # brushed streaks
            v = HD * (0.15 + 0.8 * hash1(j, 8.0))
            ctx.set_source_rgba(1, 1, 1, 0.10 + 0.12 * hash1(j, 1.0))
            ctx.set_line_width(1.5)
            ctx.move_to(*self.P(-L / 2, Y(v), INSET * (1 - v / HD)))
            ctx.line_to(*self.P(L / 2, Y(v), INSET * (1 - v / HD)))
            ctx.stroke()
        # moving sun glint along the hull
        gx = self.AX + ((t * 0.12) % 1.4 - 0.7) * L * s
        glow(ctx, gx, y0 + 0.12 * s, 0.6 * s, (1.0, 1.0, 1.0), 0.25 + 0.15 * k)
        ctx.restore()
        # rivets along the near side
        for u in np.arange(-L / 2 + 0.12, L / 2 - 0.05, 0.16):
            for v in (HD - 0.045, float(hb(u)) + 0.04):
                if v < float(hb(u)) + 0.02:
                    continue
                sx, sy = self.P(u, Y(v), INSET * (1 - v / HD))
                ctx.arc(sx, sy, max(1.2, 0.009 * s), 0, TAU)
                ctx.set_source_rgb(*darken(ALU, 0.45))
                ctx.fill()
                ctx.arc(sx - 0.4, sy - 0.6, max(0.6, 0.004 * s), 0, TAU)
                ctx.set_source_rgba(1, 1, 1, 0.8)
                ctx.fill()
        # 9. outlines and gunwale rails
        ctx.set_source_rgba(0.10, 0.12, 0.16, 0.9)
        ctx.set_line_width(max(1.5, 0.010 * s))
        self.poly(ctx, near_g + near_c[::-1])
        ctx.stroke()
        rail = max(3.0, 0.035 * s)
        for pts, w_, col in ((far_g, 0.7 * rail, darken(ALU, 0.05)), (near_g, rail, lighten(ALU, 0.35))):
            ctx.new_path()
            for i, p in enumerate(pts):
                ctx.line_to(*self.P(*p))
            ctx.set_line_width(w_ + 2.5)
            ctx.set_source_rgb(0.10, 0.12, 0.16)
            ctx.stroke_preserve()
            ctx.set_line_width(w_)
            ctx.set_source_rgb(*col)
            ctx.stroke()
        for u_ in (-L / 2, L / 2):                                   # end caps
            ctx.move_to(*self.P(u_, Y(HD), 0))
            ctx.line_to(*self.P(u_, Y(HD), BEAM))
            ctx.set_line_width(0.8 * rail + 2.0)
            ctx.set_source_rgb(0.10, 0.12, 0.16)
            ctx.stroke_preserve()
            ctx.set_line_width(0.8 * rail)
            ctx.set_source_rgb(*lighten(ALU, 0.2))
            ctx.stroke()
        # 10. the hull below the waterline, seen through the water
        dr = min(HD, -keel)
        if dr > 0:
            wl, cl = [], []
            for u, v in zip(us, hbs):
                if v < dr:
                    p = (dr - v) / (HD - v)
                    wl.append((u, 0.0, lerp(INSET, 0.0, p)))
                    cl.append((u, Y(v), INSET))
            if wl:
                self.poly(ctx, cl + wl[::-1])
                ctx.set_source_rgba(*mix(WATER_HI, WATER_LO, 0.5), 0.62)
                ctx.fill()
                p = dr / HD
                self.poly(ctx, [(-L / 2, Y(0), INSET), (-L / 2, 0.0, lerp(INSET, 0, p)),
                                (-L / 2, 0.0, lerp(BEAM - INSET, BEAM, p)), (-L / 2, Y(0), BEAM - INSET)])
                ctx.fill()
                ctx.new_path()                                       # waterline foam
                for i, q in enumerate(wl):
                    ctx.line_to(*self.P(*q))
                ctx.set_source_rgba(1, 1, 1, 0.75)
                ctx.set_line_width(max(1.5, 0.012 * s))
                ctx.stroke()

    def draw_pile(self, ctx, t, keel, a, h):
        s = self._s
        k = self.progress(t)
        Y = lambda v: keel + v                                       # noqa: E731
        z0 = INSET + 0.02
        z1 = BEAM - INSET - 0.02
        zt0 = INSET * (1 - h / HD) + 0.02
        zt1 = BEAM - zt0
        c = PILE_C
        # left slope (shade), right slope (lit), then the near face
        faces = ((-1, 0.55), (1, 1.0))
        for side, lit in faces:
            pts = [(c + side * a, Y(0), z0), (c, Y(h), zt0), (c, Y(h), zt1), (c + side * a, Y(0), z1)]
            self.poly(ctx, pts)
            g = cairo.LinearGradient(*self.P(c + side * a, Y(0), BEAM / 2), *self.P(c, Y(h), BEAM / 2))
            g.add_color_stop_rgb(0, *mix(COPPER_LO, COPPER, 0.55 * lit + 0.1))
            g.add_color_stop_rgb(0.7, *mix(COPPER, COPPER_HI, 0.35 * lit))
            g.add_color_stop_rgb(1, *mix(COPPER, COPPER_HI, 0.55 * lit))
            ctx.set_source(g)
            ctx.fill()
            # coin texture and glints on the slope (honest coin size)
            m = int(min(340, 40 + 0.012 * (a * h * s * s)))
            r = 0.5 * PENNY_D * s
            for j in range(m):
                jj = j + (0 if side < 0 else 350)
                p, q = self.tx_p[jj], self.tx_q[jj]
                u = lerp(c + side * a, c, p)
                v = h * p
                zz0 = lerp(z0, zt0, p)
                z = lerp(zz0, BEAM - zz0, q)
                sx, sy = self.P(u, Y(v), z)
                gl = max(0.0, math.sin(t * (1.5 + self.tx_c[jj]) + self.tx_g[jj])) ** 14 * (0.6 + 0.4 * k)
                self.coin(ctx, sx, sy, r, 0.6, self.tx_g[jj], (self.tx_c[jj] - 0.9) * 1.6 + 0.4 * lit - 0.2, gl)
        # near face (vertical triangle against the near wall)
        self.poly(ctx, [(c - a, Y(0), z0), (c, Y(h), zt0), (c + a, Y(0), z0)])
        ctx.set_source_rgb(*mix(COPPER_LO, COPPER, 0.4))
        ctx.fill()
        # ridge highlight
        ctx.move_to(*self.P(c, Y(h), zt0))
        ctx.line_to(*self.P(c, Y(h), zt1))
        ctx.set_source_rgba(*COPPER_HI, 0.8)
        ctx.set_line_width(max(1.0, 0.012 * s))
        ctx.stroke()

    # ------------------------------------------------------------ water effects
    def draw_rings(self, ctx, t, keel):
        """Rings spreading from the hull after each batch (bigger as the batches grow) and the hull's shadow."""
        s = self._s
        cx, cy = self.P(0.0, 0.0, BEAM / 2)
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(1.0, OBY * 0.38)
        g = cairo.RadialGradient(0, 0, 0.3 * s, 0, 0, 2.6 * s)
        g.add_color_stop_rgba(0, 0, 0.05, 0.1, 0.30)
        g.add_color_stop_rgba(1, 0, 0.05, 0.1, 0.0)
        ctx.set_source(g)
        ctx.arc(0, 0, 2.6 * s, 0, TAU)
        ctx.fill()
        e = self.event_index(t)
        for i in range(max(1, e - 8), e + 1):
            age = t - self.ev_t[i]
            if age < 0 or age > 1.6:
                continue
            strength = min(1.0, 0.25 + 0.1 * math.log1p(self.batch_n(i)) + 0.4 * self.progress(self.ev_t[i]))
            R = (L / 2 + 0.15 + 0.9 * age) * s
            ctx.new_path()
            ctx.save()
            ctx.scale(1.0, 0.36 * (1 + 0.15 * age))
            ctx.arc(0, 0, R, 0, TAU)
            ctx.restore()
            ctx.set_source_rgba(1, 1, 1, 0.45 * strength * (1 - age / 1.6))
            ctx.set_line_width(2.0 + 2.5 * strength)
            ctx.stroke()
        ctx.restore()

    def draw_splash(self, ctx, t, keel, d):
        """Droplets thrown up along the waterline when a batch lands (grows with the batch) and the big swamp."""
        s = self._s
        e = self.event_index(t)
        ctx.set_source_rgba(0.92, 0.98, 1.0, 0.85)
        for i in range(max(1, e - 2), e + 1):
            age = t - self.ev_t[i]
            if age < 0 or age > 0.55 or i == len(self.ev_t) - 1:
                continue
            amp = clamp(0.15 + 35.0 * (self.drafts[i] - self.drafts[i - 1]) + 0.5 * self.progress(self.ev_t[i]))
            m = int(4 + 22 * amp)
            for j in range(m):
                h1, h2, h3 = hash1(i, j + 0.3), hash1(j + 4, i), hash1(i * 3 + j, 1.0)
                u = -L / 2 + L * h1
                z = -0.03 if h3 < 0.6 else BEAM + 0.03
                vy = (0.5 + 1.3 * h2) * (0.4 + amp)
                y = vy * age - 4.9 * age * age
                if y < 0:
                    continue
                sx, sy = self.P(u + (h3 - 0.5) * 0.4 * age, y, z)
                ctx.arc(sx, sy, (0.010 + 0.012 * h3) * s, 0, TAU)
                ctx.fill()
        if 0 <= d < 1.6:                                              # the swamp: water surges over both rails
            a = 1.0 - d / 1.6
            for j in range(70):
                h1, h2, h3 = hash1(j, 11.0), hash1(j, 12.0), hash1(j, 13.0)
                tt = (d - 0.25 * h3)
                if tt < 0:
                    continue
                u = -L / 2 + L * h1
                y = keel + HD + (0.9 * h2 + 0.3) * tt - 4.9 * tt * tt
                if y < keel + HD - 0.05:
                    continue
                sx, sy = self.P(u, y, BEAM * (h3 > 0.5) + (0.2 * tt if h3 <= 0.5 else -0.2 * tt))
                ctx.set_source_rgba(0.9, 0.97, 1.0, 0.85 * a)
                ctx.arc(sx, sy, (0.015 + 0.02 * h2) * s, 0, TAU)
                ctx.fill()

    def draw_sinking(self, ctx, t, keel, d):
        """Foam on the surface where the boat went down, and bubbles."""
        if d < 0.5:
            return
        s = self._s
        cx, cy = self.P(0.0, 0.0, BEAM / 2)
        for r in range(4):
            ph = clamp((d - 0.5 - 0.25 * r) / 2.0)
            if ph <= 0:
                continue
            ctx.save()
            ctx.translate(cx, cy)
            ctx.scale(1.0, OBY * 0.36)
            ctx.arc(0, 0, (L / 2 + 0.2 + 1.6 * ph) * s, 0, TAU)
            ctx.restore()
            ctx.set_source_rgba(1, 1, 1, 0.55 * (1 - ph))
            ctx.set_line_width(3.0 + 5.0 * (1 - ph))
            ctx.stroke()
        for j in range(40):
            h1, h2, h3 = hash1(j, 21.0), hash1(j, 22.0), hash1(j, 23.0)
            age = (d - 0.5 - 1.5 * h3)
            if age < 0:
                continue
            u = -L / 2 + 0.2 + (L - 0.4) * h1
            z = INSET + (BEAM - 2 * INSET) * h2
            y = keel + 0.15 + 0.9 * age
            if y > 0:
                sx, sy = self.P(u, 0.0, z)
                pop = clamp((y - 0) / 0.08)
                if pop < 1:
                    ctx.arc(sx, sy, (0.03 + 0.05 * pop) * s, 0, TAU)
                    ctx.set_source_rgba(1, 1, 1, 0.7 * (1 - pop))
                    ctx.set_line_width(2.0)
                    ctx.stroke()
                continue
            sx, sy = self.P(u + 0.04 * math.sin(age * 9 + j), y, z)
            ctx.arc(sx, sy, (0.012 + 0.02 * h2) * s, 0, TAU)
            ctx.set_source_rgba(0.9, 0.97, 1.0, 0.65)
            ctx.set_line_width(1.6)
            ctx.stroke()

    def draw_world(self, ctx, t, s, res):
        d = t - self.T_REV
        keel = self.keel_y(t)
        self.draw_rings(ctx, t, keel)
        gd = -(keel + HD)                                             # gunwale depth below the surface
        if gd > 0.0:                                                  # under water: the boat fades into the lake
            ctx.push_group()
            self.draw_boat(ctx, t, keel, d)
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(1.0 - clamp(1.7 * gd) ** 0.8 * 0.97)
        else:
            self.draw_boat(ctx, t, keel, d)
        if d < 0.6:
            self.draw_bounces(ctx, t, keel)
        self.draw_splash(ctx, t, keel, d)
        self.draw_sinking(ctx, t, keel, d)
        self.draw_duck(ctx, t)
        self.draw_dock(ctx, t)
        if d < 0:
            self.draw_falling(ctx, t, keel)
        self.draw_bucket(ctx, t)

    # ------------------------------------------------------------ sound
    def event_sound(self, i, t, k, rng):
        """Metallic clinks (more of them, higher, for bigger batches) plus the hiss of a pour that grows."""
        nb = self.batch_n(i)
        f0 = 1250.0 * 2 ** (0.9 * k)
        m = 1 + int(min(5, 1.3 * math.log10(max(1, nb)) + 2.5 * k))
        out = []
        for j in range(m):
            f = f0 * (1 + 0.10 * (hash1(i, j) - 0.5)) * (1.0 + 0.12 * j)
            out.append((clink(f), (0.20 + 0.06 * k) / (1 + 0.25 * j), 0.6 * (hash1(j, i) - 0.5),
                        0.028 * j + 0.01 * hash1(i, j + 2)))
        if nb > 3:
            hiss = norm(au.noise(rng, 0.16, 2500, 9000)) * np.exp(-np.arange(int(0.16 * au.SR)) / au.SR / 0.06)
            out.append((hiss, 0.04 + 0.12 * k, 0.0, 0.0))
        dd = self.drafts[i] - self.drafts[i - 1]
        if k > 0.45:                                                  # the hull taking the weight: a soft slap
            slap = norm(au.noise(rng, 0.12, 350, 1800)) * np.exp(-np.arange(int(0.12 * au.SR)) / au.SR / 0.035)
            out.append((slap, min(0.30, 0.06 + 6.0 * dd + 0.12 * k), 0.0, 0.01))
        return out

    def reveal_sound(self, fx, rng):
        T = self.T_REV
        sr = au.SR
        n = int(2.6 * sr)
        rush = au.swept_noise(rng, 2.6, 350, 1500) * np.minimum(1.0, np.arange(n) / (0.12 * sr))
        rush *= np.exp(-np.arange(n) / sr / 1.3)
        au.place(fx, norm(rush), T + 0.0, 0.50, 0.0)
        au.place(fx, au.boom(rng, 1.4, 110.0, 40.0), T, 0.55)
        for j in range(7):                                            # glugs as the air leaves
            f = 260.0 - 18 * j + 30 * hash1(j, 3.0)
            au.place(fx, au.glide(f, f * 0.62, 0.16, decay=0.07, harm=0.6), T + 0.55 + 0.23 * j + 0.05 * hash1(j),
                     0.22, 0.4 * (hash1(j, 8.0) - 0.5))
        for j, m_ in enumerate((67, 71, 74, 79)):                    # bright chord as the "why" line lands
            au.place(fx, au.bell(au.hz(m_), 1.6), T + 0.5 + 0.05 * j, 0.10)

    # ------------------------------------------------------------ metadata
    def metadata(self):
        title = "How Many Pennies to Sink a Boat? \U0001FA99"
        assert len(title) <= 60 and str(ANSWER) not in title
        return dict(
            title=title,
            description=(
                "Comment your number \U0001F447\n"
                "US pennies get poured into a 12-foot aluminum jon boat on calm fresh water, a few at first, then in "
                "bigger and bigger batches. The counter shows every penny in the boat.\n"
                "Each penny weighs 2.5 g. The boat sits exactly as low as the real weight says, and it sinks only when "
                "boat plus pennies outweigh the water the hull can push aside with the water at the rim. The pile is "
                "drawn at its real size; the duck and the person on the dock show the scale.\n"
                "No footage, samples or stock media.\n"
                "#simulation #satisfying #physics"),
            tags=["pennies", "how many pennies", "sink a boat", "penny boat", "buoyancy", "archimedes", "jon boat",
                  "scale", "physics simulation", "exponential growth", "prediction", "satisfying", "made with code",
                  "simulation", "physics"],
            pinned_comment="Comment your number \U0001F447 How many pennies does it take to sink a boat?",
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of a cartoon boat, coins and lake; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / self.FPS, 2),
            drama=self.stats,
        )
