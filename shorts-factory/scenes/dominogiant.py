"""DOMINO GIANT: a chain of dominoes where each one is 1.5 times bigger than the last, until one out-tops a tower.

Side view. Domino 1 is 5 cm tall; every next one is 1.5 times bigger in every dimension and stands half a
domino-height beyond the previous one. Each domino is a rigid block that pivots on its lower front edge
under gravity; frictionless contact between neighbors transfers the push (stiff, damped contact, so hits do not
bounce). The line ends at a 300 m tower (a lighter rigid block): the first domino taller than the tower knocks it
over. The count is fixed by the rule and the physics, so the seed only changes cosmetic things (accent colors, dust,
stars). Playback speed changes along the way (slow motion for the small ones, fast forward for the giants) and is
shown on screen.

simulate() is deterministic and needs no rendering; drama() recomputes the answer from the rule.
"""
import bisect
import functools
import math
import os
import random

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_in_out, ease_out, ease_out_back, glow, hex_rgb, lerp,
                        lighten, mix, polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30

# ---------------------------------------------------------------------- the rule
G = 9.81
H1 = 0.05                     # domino 1 is 5 cm tall
GROW = 1.5                    # each next one is 1.5 times bigger in every dimension
GAP = 0.5                     # clear gap to the next one: half the height of the smaller domino
THICK = 0.10                  # thickness / height
DEPTH = 0.5                   # width (into the screen) / height
TOWER_H = 300.0
TOWER_W = 46.0
TOWER_RHO = 0.25              # the tower is mostly air: a quarter of the density of a domino
KS, ZETA = 3000.0, 0.7        # contact stiffness (per weight / height) and damping ratio
FLICK = 1.6                   # starting flick: 1.6 times the spin needed to carry domino 1 over its balance point
WINDOW = 3                    # dominoes this far behind the front follow the neighbor they lean on

# ---------------------------------------------------------------------- the edit (video time per fall)
V1, VQ = 0.44, 1.055          # video seconds for the first fall, growth per fall (the rhythm slows down gently)
V_LAST = 2.4                  # the last giant leaning into the tower
V_FALL = 1.6                  # the tower going down
V_POST = 3.5                  # result card
SPEED_POST = 5.5
XFADE = 0.4                   # dissolve back to the opening frame

HS = 390.0                    # on-screen height of the domino that has just been hit
X0S = 300.0                   # its pivot on screen
GY = 1440.0                   # ground line


def count_needed():
    """Dominoes in the line: the last one is the first that is taller than the tower."""
    n, h = 1, H1
    while h <= TOWER_H:
        h *= GROW
        n += 1
    return n


def build():
    n = count_needed()
    B = []
    x = 0.0
    for i in range(n):
        h = H1 * GROW ** i
        d = THICK * h
        if i:
            x = B[-1]["x"] + GAP * B[-1]["h"] + d
        m = h * d * DEPTH * h
        B.append(dict(h=h, d=d, x=x, m=m, I=m * (h * h + d * d) / 3))
    x = B[-1]["x"] + GAP * B[-1]["h"] + TOWER_W
    m = TOWER_H * TOWER_W * TOWER_W * TOWER_RHO
    B.append(dict(h=TOWER_H, d=TOWER_W, x=x, m=m, I=m * (TOWER_H ** 2 + TOWER_W ** 2) / 3))
    return B


def contacts(A, th, B, ph):
    """Contact points between A and its right neighbor B: (depth, rAx, rAy, rBx, rBy, nx, ny); n pushes A back."""
    out = []
    ct, st, cp, sp = math.cos(th), math.sin(th), math.cos(ph), math.sin(ph)
    dx = B["x"] - A["x"]
    hA, dA, hB, dB = A["h"], A["d"], B["h"], B["d"]
    px, py = hA * st, hA * ct                       # A's top front corner against B's back face
    qx, qy = px - dx, py
    lx = qx * cp - qy * sp
    ly = qx * sp + qy * cp
    if 0.0 < ly < hB and -dB < lx < 0.0:
        out.append((lx + dB, px, py, qx, qy, -cp, sp))
    for yb in (hB, 0.0):                            # B's back corners against A's front face
        wx = -dB * cp + yb * sp
        wy = dB * sp + yb * cp
        ux = wx + dx
        ax = ux * ct - wy * st
        ay = ux * st + wy * ct
        if 0.0 < ay < hA and -dA < ax < 0.0:
            out.append((-ax, ux, wy, wx, wy, -ct, st))
    return out


def rest_angle(A, B, ph, lo):
    """How far A can lean (from lo up to flat) before it touches B standing at angle ph."""
    if contacts(A, lo, B, ph):
        return lo
    step = math.radians(1.5)
    a = lo
    while a < math.pi / 2:
        b = min(a + step, math.pi / 2)
        if contacts(A, b, B, ph):
            for _ in range(22):
                mid = 0.5 * (a + b)
                if contacts(A, mid, B, ph):
                    b = mid
                else:
                    a = mid
            return a
        a = b
    return math.pi / 2


@functools.lru_cache(maxsize=1)
def simulate():
    B = build()
    nb = len(B)
    th = [0.0] * nb
    om = [0.0] * nb
    L = math.hypot(B[0]["h"], B[0]["d"])
    om[0] = FLICK * math.sqrt(3 * G / L * (1 - math.cos(math.atan(THICK))))
    hit = [None] * nb                 # time each body is first touched by the one before it
    hitpos = [None] * nb
    hit[0], hitpos[0] = 0.0, (0.0, B[0]["h"])
    last_c = [-1e9] * nb
    clacks = []                       # later contact onsets (a domino landing on the next one): (t, i, strength)
    lead = lo = 0
    t = 0.0
    keys_t, keys = [0.0], [list(th)]
    slam = None
    t_rest = None
    while True:
        dt = 0.2 / math.sqrt(3 * KS * G / B[lo]["h"])
        hi = min(lead + 1, nb - 1)
        tq = [0.0] * nb
        for i in range(lo, hi + 1):
            b = B[i]
            tq[i] = b["m"] * G * (-0.5 * b["d"] * math.cos(th[i]) + 0.5 * b["h"] * math.sin(th[i]))
        for i in range(lo, hi):
            A, Bb = B[i], B[i + 1]
            cs = contacts(A, th[i], Bb, th[i + 1])
            if not cs:
                continue
            # contact stiffness follows the lighter body, except against the tower: it ends up lying on the
            # ground, where it cannot give way, so the full weight of the last domino bears on it
            ms = A["m"] if i + 1 == nb - 1 else min(A["m"], Bb["m"])
            k = KS * ms * G / A["h"]
            c = 2 * ZETA * math.sqrt(k * ms / 3)
            first = True
            for pen, rax, ray, rbx, rby, nx, ny in cs:
                closing = -((om[i] * ray - om[i + 1] * rby) * nx + (-om[i] * rax + om[i + 1] * rbx) * ny)
                f = k * pen + c * closing
                if f <= 0:
                    continue
                fx, fy = f * nx, f * ny
                tq[i] += ray * fx - rax * fy
                tq[i + 1] -= rby * fx - rbx * fy
                if hit[i + 1] is None:
                    hit[i + 1] = t
                    hitpos[i + 1] = (A["x"] + rax, ray)
                    lead = i + 1
                elif first and t - last_c[i] > 0.08 * math.sqrt(A["h"] / G):
                    s = closing / math.sqrt(G * A["h"])
                    if s > 0.04:
                        clacks.append((t, i, s))
                first = False
            last_c[i] = t
        for i in range(lo, hi + 1):
            om[i] += tq[i] / B[i]["I"] * dt
            th[i] += om[i] * dt
            if th[i] < 0.0:
                th[i] = 0.0
                om[i] = max(0.0, om[i])
            elif th[i] > math.pi / 2:
                th[i] = math.pi / 2
                if i == nb - 1 and slam is None:
                    slam = t
                om[i] = 0.0
        t += dt
        while lo < lead - WINDOW:
            om[lo] = 0.0
            lo += 1
        if t - keys_t[-1] >= 0.012 * math.sqrt(B[lead]["h"] / G):
            for i in range(lo - 1, -1, -1):          # settled dominoes follow the one they lean on
                if th[i] < math.pi / 2:
                    th[i] = rest_angle(B[i], B[i + 1], th[i + 1], th[i])
            keys_t.append(t)
            keys.append(list(th))
            if slam is not None:
                sc = math.sqrt(G / B[-1]["h"])
                if all(abs(om[i]) < 0.004 * sc for i in range(lo, nb)):
                    if t_rest is None:
                        t_rest = t
                    elif t - t_rest > 1.0:
                        break
                else:
                    t_rest = None
            if t > 400 or (lead < nb - 1 and t - hit[lead] > 40 * math.sqrt(B[lead]["h"] / G)):
                break
    return dict(B=B, hit=hit, hitpos=hitpos, clacks=clacks, keys_t=np.array(keys_t), keys=np.array(keys),
                slam=slam, t_end=t)


# ---------------------------------------------------------------------- monotone cubic through the knots
def _slopes(xs, ys):
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    dl = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0], m[-1] = dl[0], dl[-1]
    for i in range(1, n - 1):
        if dl[i - 1] * dl[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / dl[i - 1] + w2 / dl[i])
    return m


def _curve(xs, ys, m, x):
    """Value and slope at x."""
    x = min(max(x, xs[0]), xs[-1])
    i = min(len(xs) - 2, max(0, bisect.bisect_right(xs, x) - 1))
    h = xs[i + 1] - xs[i]
    s = (x - xs[i]) / h
    s2, s3 = s * s, s * s * s
    v = (2 * s3 - 3 * s2 + 1) * ys[i] + (s3 - 2 * s2 + s) * h * m[i] + (-2 * s3 + 3 * s2) * ys[i + 1] + (s3 - s2) * h * m[i + 1]
    dv = ((6 * s2 - 6 * s) * ys[i] / h + (3 * s2 - 4 * s + 1) * m[i] + (-6 * s2 + 6 * s) * ys[i + 1] / h
          + (3 * s2 - 2 * s) * m[i + 1])
    return v, dv


def timeline(sim):
    """Video time of every hit, and the knots that map video time to simulated time."""
    hit, nb = sim["hit"], len(sim["B"])
    if any(h is None for h in hit) or sim["slam"] is None:
        return None
    vk, tk = [0.0], [0.0]
    for j in range(1, nb):
        vk.append(vk[-1] + (V1 * VQ ** (j - 1) if j < nb - 1 else V_LAST))
        tk.append(hit[j])
    vhit = list(vk)
    vk.append(vk[-1] + V_FALL)
    tk.append(sim["slam"])
    v_slam = vk[-1]
    vk.append(vk[-1] + V_POST)
    tk.append(sim["slam"] + V_POST * SPEED_POST)
    return dict(vk=vk, tk=tk, mt=_slopes(vk, tk), vhit=vhit, v_slam=v_slam, dur=vk[-1])


def drama(seed):
    sim = simulate()
    tl = timeline(sim)
    n = count_needed()
    hs = [H1 * GROW ** i for i in range(n)]
    rnd = random.Random(seed * 7919 + 3)
    pal = rnd.randrange(len(PALETTES))
    stars = [(rnd.uniform(30, 1050), rnd.uniform(170, 760)) for _ in range(22)]
    spread = min(math.hypot(a[0] - b[0], a[1] - b[1]) for k, a in enumerate(stars) for b in stars[k + 1:])
    cosmetic = round(min(1.0, spread / 60.0), 3)                  # evenly scattered stars look tidier
    valid = tl is not None and len(sim["B"]) == n + 1
    dur = tl["dur"] if valid else None
    short_m = TOWER_H - hs[-2]                                    # the runner-up falls this far short of the tower
    short_px = short_m * HS / hs[-2]                              # on screen while it is the domino being hit
    score = 0.0
    if not valid:
        score -= 100.0
    else:
        score += 6.0 if 24.5 <= dur <= 31.5 else -40.0
        score += 4.0 if short_px >= 40.0 else -20.0               # the near miss must be readable on a phone
        score += cosmetic
    stage = [round(sim["hit"][j] - sim["hit"][j - 1], 3) for j in range(1, len(sim["hit"]))] if valid else []
    return dict(
        seed=seed, score=round(score, 3), valid=valid,
        winner=str(n), dominoes_needed=n,
        growth=GROW, first_height_m=H1, tower_height_m=TOWER_H,
        last_domino_height_m=round(hs[-1], 1), runner_up_height_m=round(hs[-2], 1),
        runner_up_short_by_m=round(short_m, 1), runner_up_short_by_px=round(short_px, 1),
        last_domino_over_by_m=round(hs[-1] - TOWER_H, 1),
        duration_s=round(dur, 2) if dur else None,
        simulated_seconds_to_tower_hit=round(sim["hit"][-1], 2) if valid else None,
        simulated_seconds_tower_fall=round(sim["slam"] - sim["hit"][-1], 2) if valid else None,
        first_fall_real_s=stage[0] if stage else None, last_fall_real_s=stage[-1] if stage else None,
        playback_speed_range=[round(stage[0] / V1, 2), round(stage[-2] / (V1 * VQ ** (n - 2)), 2)] if valid else None,
        palette=pal, cosmetic_score=cosmetic,
        note="count fixed by the rule; the seed only changes accent colors, dust and stars",
    )


def score_seed(seed):
    return drama(seed)


# ====================================================================== look
INK = hex_rgb("#0B0D17")
CREAM = hex_rgb("#F6F1E4")
IVORY = hex_rgb("#F5EFDF")
AMBER = hex_rgb("#FFC94D")
MUTED = hex_rgb("#C9BFD9")
SIL = hex_rgb("#241631")          # scale-marker silhouettes
TOWER_C = hex_rgb("#2A3560")
GROUND = hex_rgb("#17121F")
PALETTES = [(hex_rgb("#2EC4B6"), hex_rgb("#FF5E5B")), (hex_rgb("#4CC9F0"), hex_rgb("#F72585")),
            (hex_rgb("#B8E04A"), hex_rgb("#9B5DE5")), (hex_rgb("#FFB703"), hex_rgb("#3A86FF"))]
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
HXL = 56.0                        # left edge of the header and counter
CX = 475.0                        # center of the safe area

MARKERS = (("COIN", 0.024, -0.0265), ("HAND", 0.18, 0.2), ("PERSON", 1.7, 2.91), ("HOUSE", 8.0, 15.0),
           ("TREE", 20.0, 33.9))


def marker_path(ctx, kind):
    """Unit silhouette: base on y = 0, height 1 (y up is negative), centered on x = 0."""
    if kind == "COIN":
        ctx.arc(0, -0.5, 0.5, 0, TAU)
    elif kind == "HAND":
        rounded_rect(ctx, -0.21, -0.52, 0.42, 0.52, 0.08)
        for x, top in ((-0.205, 0.86), (-0.095, 0.97), (0.012, 1.0), (0.12, 0.9)):
            rounded_rect(ctx, x, -top, 0.09, top - 0.4, 0.045)
        ctx.save()
        ctx.translate(0.2, -0.3)
        ctx.rotate(0.6)
        rounded_rect(ctx, -0.045, -0.36, 0.1, 0.4, 0.05)
        ctx.restore()
    elif kind == "PERSON":
        ctx.arc(0, -0.92, 0.078, 0, TAU)
        rounded_rect(ctx, -0.115, -0.82, 0.23, 0.36, 0.05)
        rounded_rect(ctx, -0.105, -0.5, 0.09, 0.5, 0.03)
        rounded_rect(ctx, 0.015, -0.5, 0.09, 0.5, 0.03)
        rounded_rect(ctx, -0.185, -0.8, 0.055, 0.36, 0.027)
        rounded_rect(ctx, 0.13, -0.8, 0.055, 0.36, 0.027)
    elif kind == "HOUSE":
        ctx.rectangle(-0.36, -0.6, 0.72, 0.6)
        polygon(ctx, [(-0.46, -0.58), (0, -1.0), (0.46, -0.58)])
        ctx.rectangle(0.18, -0.92, 0.09, 0.22)
    elif kind == "TREE":
        ctx.rectangle(-0.04, -0.45, 0.08, 0.45)
        for x, y, r in ((0, -0.72, 0.28), (-0.17, -0.56, 0.2), (0.17, -0.56, 0.2), (0, -0.5, 0.2)):
            ctx.new_sub_path()
            ctx.arc(x, y, r, 0, TAU)


def make_background(font, stars):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = np.clip(yy / GY, 0, 1)[..., None]
    top, mid, hor = (np.array(hex_rgb(c), np.float32) for c in ("#10162E", "#4A3470", "#E2725B"))
    k1 = np.clip(v / 0.62, 0, 1)
    k2 = np.clip((v - 0.62) / 0.38, 0, 1) ** 1.3
    img = (top * (1 - k1) + mid * k1) * (1 - k2) + hor * k2
    d = np.hypot((xx - 700) / 900, (yy - GY) / 420)
    img += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array([0.22, 0.14, 0.02], np.float32)
    vig = np.clip(1.10 - 0.42 * (((xx - 520) / 760) ** 2 + ((yy - 900) / 1300) ** 2), 0.4, 1.0)
    img *= vig[..., None]
    img += np.random.default_rng(11).normal(0, 0.006, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    del img, yy, xx, v, d, vig, k1, k2
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    for x, y in stars:
        a = 0.55 * (1 - y / 800)
        ctx.arc(x, y, 2.2, 0, TAU)
        ctx.set_source_rgba(*CREAM, a)
        ctx.fill()
    # header: the question, the rule, the fine print (left aligned)
    font.draw(ctx, "HOW MANY DOMINOES", HXL, 262, 96, CREAM, stroke=INK, stroke_w=10, shadow=6)
    font.draw(ctx, "TO TOPPLE A TOWER?", HXL, 362, 96, CREAM, stroke=INK, stroke_w=10, shadow=6)
    txt, size = "EACH ONE IS 1.5x BIGGER", 50
    tw = font.width(txt, size, 1.5)
    rounded_rect(ctx, HXL, 388, tw + 48, 70, 14)
    ctx.set_source_rgba(*AMBER, 1)
    ctx.fill()
    font.draw(ctx, txt, HXL + 24, 442, size, INK, tracking=1.5)
    font.draw(ctx, "DOMINO 1 IS 5 CM  •  THE TOWER IS 300 M", HXL, 502, 31, CREAM, tracking=1.0, stroke=INK,
              stroke_w=5)
    font.draw(ctx, "THE FIRST ONE TALLER THAN THE TOWER KNOCKS IT OVER", HXL, 542, 31, CREAM, tracking=1.0,
              stroke=INK, stroke_w=5)
    surf.flush()
    return surf, buf


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((45, -14.0), (90, -8.0), (200, -3.0), (400, 0.0), (900, 1.0), (3000, 3.5), (8000, 2.0), (16000, -1.0))
    DRIVE = 4.5

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate()
        self.tl = timeline(self.sim)
        if self.tl is None:
            raise ValueError("the chain did not reach the tower")
        self.stats = drama(seed)
        self.B = self.sim["B"]
        self.nb = len(self.B)
        self.N = self.nb - 1                              # dominoes (the answer)
        self.dur = self.tl["dur"]
        self.nframes = int(round(self.dur * FPS))
        self.vhit = self.tl["vhit"]
        self.v_slam = self.tl["v_slam"]
        self.v_res = self.v_slam + 0.35
        self._res = None
        self._prep()

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    def _prep(self):
        tl, sim = self.tl, self.sim
        nf = self.nframes
        self.fT = np.zeros(nf)
        self.fspeed = np.zeros(nf)
        for f in range(nf):
            self.fT[f], self.fspeed[f] = _curve(tl["vk"], tl["tk"], tl["mt"], f / FPS)
        k = np.ones(9) / 9
        self.fspeed = np.convolve(np.pad(self.fspeed, 4, mode="edge"), k, mode="valid")
        self.th = np.stack([np.interp(self.fT, sim["keys_t"], sim["keys"][:, i]) for i in range(self.nb)], axis=1)
        self.uk = [float(j + 1) for j in range(self.N)]
        self.um = _slopes(self.vhit[:self.N], self.uk)
        rnd = random.Random(self.seed * 7919 + 3)                 # same stream as drama(): palette, then stars
        self.pal = PALETTES[rnd.randrange(len(PALETTES))]
        self.stars = [(rnd.uniform(30, 1050), rnd.uniform(170, 760)) for _ in range(22)]
        self.puffs = [[(rnd.uniform(-2.6, -0.5), rnd.uniform(0.5, 1.0), rnd.uniform(0.6, 1.0)) for _ in range(7)]
                      for _ in range(self.nb)]
        self.dust = [(rnd.random(), rnd.uniform(0.4, 1.0), rnd.uniform(0.5, 1.0), rnd.uniform(-1, 1)) for _ in range(34)]
        # end pose: the whole fallen line and the tower in frame
        xl = 230.0                                                # the far end of the line is a speck by now
        xr = self.B[-1]["x"] + TOWER_H + 20.0
        self.s_end = 890.0 / (xr - xl)
        self.ax_end = 40.0 - xl * self.s_end

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            surf2 = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            font = Font(FONT_PATH)
            bg, keep = make_background(font, self.stars)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), surf2=surf2,
                             ctx2=cairo.Context(surf2), bg=bg, keep=keep, font=font)
        return self._res

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        self.draw(ctx, f, res)
        ctx.restore()
        k = ramp(f / FPS, self.dur - XFADE, self.dur)             # dissolve back to the opening frame
        if k > 0:
            c2 = res["ctx2"]
            c2.save()
            self.draw(c2, 0, res)
            c2.restore()
            res["surf2"].flush()
            ctx.save()
            ctx.set_source_surface(res["surf2"], 0, 0)
            ctx.paint_with_alpha(ease_in_out(k))
            ctx.restore()
        res["surf"].flush()
        return bytes(res["surf"].get_data())

    # ------------------------------------------------------------ camera
    def camera(self, v):
        """Scale (px per m) and screen x of world x = 0."""
        v_last = self.vhit[self.N - 1]                            # the last domino is hit here
        u = _curve(self.vhit[:self.N], self.uk, self.um, v)[0]
        u += 0.25 * ease_out(ramp(v, v_last, v_last + 1.0))
        s_a = HS / (H1 * GROW ** (u - 1))
        ax_a = X0S - 1.3 * H1 * (GROW ** (u - 1) - 1) * s_a
        k = ease_in_out(ramp(v, v_last + 0.1, v_last + 1.9))
        if k <= 0:
            return s_a, ax_a
        z = 1 - 0.08 * ramp(v, v_last + 1.5, self.dur)            # slow drift out, so the end card is never still
        s_b = self.s_end * z
        ax_b = CX + (self.ax_end - CX) * z
        return lerp(s_a, s_b, k), lerp(ax_a, ax_b, k)

    def shake(self, v):
        sx = sy = 0.0
        for j in range(1, self.nb):
            d = v - self.vhit[j]
            if 0 <= d < 0.3:
                a = (1.0 + 0.3 * j) * math.exp(-d / 0.07)
                sx += a * math.sin(d * 83)
                sy += a * math.cos(d * 67)
        d = v - self.v_slam
        if 0 <= d < 0.9:
            a = 17.0 * math.exp(-d / 0.2)
            sx += a * math.sin(d * 61)
            sy += a * math.cos(d * 47)
        return sx, sy

    def count(self, v):
        return sum(1 for j in range(self.N) if self.vhit[j] <= v + 1e-9)

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, res):
        v = f / FPS
        font = res["font"]
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        S, ax = self.camera(v)
        th = self.th[min(f, self.nframes - 1)]
        sx, sy = self.shake(v)
        ctx.save()
        ctx.translate(sx, sy)
        self._markers(ctx, S, ax)
        self._tower_line(ctx, v, S, ax, font)
        self._ground(ctx, S, ax)
        self._slam_fx(ctx, v, S, ax)
        self._tower(ctx, S, ax, th[-1])
        for i in range(self.N - 1, -1, -1):
            self._domino(ctx, i, S, ax, th[i], font)
        self._hit_fx(ctx, v, S, ax)
        ctx.restore()
        self._labels(ctx, v, S, ax, font)
        self._hud(ctx, f, v, S, font)
        self._result(ctx, v, font)

    def _ground(self, ctx, S, ax):
        g = cairo.LinearGradient(0, GY, 0, H)
        g.add_color_stop_rgb(0, *lighten(GROUND, 0.10))
        g.add_color_stop_rgb(0.05, *GROUND)
        g.add_color_stop_rgb(1, *darken(GROUND, 0.5))
        ctx.rectangle(-60, GY, W + 120, H - GY + 60)
        ctx.set_source(g)
        ctx.fill()
        ctx.rectangle(-60, GY - 2, W + 120, 4)
        ctx.set_source_rgba(*mix(GROUND, CREAM, 0.35), 1)
        ctx.fill()
        # ruler ticks every 10^k meters: they crowd together as the camera pulls back
        for k in range(-3, 4):
            p = 10.0 ** k
            px = p * S
            if not 14 < px < 1400:
                continue
            al = ramp(px, 14, 40) * (1 - ramp(px, 600, 1400))
            ln = 7 + 13 * ramp(px, 14, 300)
            m0 = math.floor((-60 - ax) / px)
            m1 = math.ceil((W + 60 - ax) / px)
            for m in range(m0, m1 + 1):
                x = ax + m * px
                ctx.move_to(x, GY + 3)
                ctx.line_to(x, GY + 3 + ln)
            ctx.set_source_rgba(*CREAM, 0.55 * al)
            ctx.set_line_width(2.5)
            ctx.stroke()

    def _markers(self, ctx, S, ax):
        for kind, hm, xm in MARKERS:
            hp = hm * S
            x = ax + xm * S
            if hp < 3 or hp > 2600 or x < -hp or x > W + hp:
                continue
            ctx.save()
            ctx.translate(x, GY)
            ctx.scale(hp, hp)
            ctx.new_path()
            marker_path(ctx, kind)
            ctx.set_source_rgba(*SIL, 0.7)
            ctx.fill()
            if kind == "COIN":
                ctx.arc(0, -0.5, 0.38, 0, TAU)
                ctx.set_source_rgba(*mix(SIL, CREAM, 0.25), 0.9)
                ctx.set_line_width(0.035)
                ctx.stroke()
            ctx.restore()

    def _labels(self, ctx, v, S, ax, font):
        for kind, hm, xm in MARKERS:
            hp = hm * S
            x = ax + xm * S
            al = ramp(hp, 26, 60) * (1 - ramp(hp, 520, 900))
            if al <= 0 or not 30 < x < 930:
                continue
            font.draw(ctx, kind, clamp(x, 70, 880), max(600.0, GY - hp - 18), 32, CREAM, anchor="c", alpha=al,
                      tracking=1.5, stroke=INK, stroke_w=6)
        xt = ax + (self.B[-1]["x"] - TOWER_W / 2) * S
        al = 1 - ramp(xt, 900, 1000)
        al *= 1 - ramp(v, self.vhit[-1], self.vhit[-1] + 0.4)
        if al > 0:
            font.draw(ctx, "TOWER 300 M", clamp(xt, 120, 830), GY - TOWER_H * S - 24, 36, AMBER, anchor="c", alpha=al,
                      tracking=1.5, stroke=INK, stroke_w=6)

    def _tower_line(self, ctx, v, S, ax, font):
        """Dashed line at the height of the tower top: what a domino has to beat."""
        xb = ax + (self.B[-1]["x"] - TOWER_W) * S
        al = (1 - ramp(xb, 1000, 1120)) * (1 - ramp(v, self.vhit[-1], self.vhit[-1] + 0.4))
        if al <= 0:
            return
        y = GY - TOWER_H * S
        ctx.save()
        ctx.set_dash([18, 14])
        ctx.move_to(-20, y)
        ctx.line_to(min(xb, W + 20), y)
        ctx.set_source_rgba(*AMBER, 0.85 * al)
        ctx.set_line_width(4)
        ctx.stroke()
        ctx.restore()

    def _tower(self, ctx, S, ax, th):
        b = self.B[-1]
        x = ax + b["x"] * S
        hp, dp = TOWER_H * S, TOWER_W * S
        if x - dp - hp > W + 60 or hp < 2:
            return
        ctx.save()
        ctx.translate(x, GY)
        ctx.rotate(th)
        ctx.rectangle(-dp, -hp, dp, hp)
        g = cairo.LinearGradient(-dp, 0, 0, 0)
        g.add_color_stop_rgb(0, *lighten(TOWER_C, 0.16))
        g.add_color_stop_rgb(1, *darken(TOWER_C, 0.25))
        ctx.set_source(g)
        ctx.fill()
        n = 20
        for k in range(1, n):                                    # floors
            y = -hp * k / n
            ctx.move_to(-dp * 0.84, y)
            ctx.line_to(-dp * 0.16, y)
        ctx.set_source_rgba(*mix(TOWER_C, AMBER, 0.75), 0.75)
        ctx.set_line_width(max(1.0, hp * 0.012))
        ctx.stroke()
        ctx.rectangle(-dp, -hp, dp, hp * 0.035)
        ctx.set_source_rgba(*AMBER, 1)
        ctx.fill()
        ctx.rectangle(-dp, -hp, dp, hp)
        ctx.set_source_rgba(*INK, 1)
        ctx.set_line_width(clamp(hp * 0.012, 1.5, 4))
        ctx.stroke()
        ctx.restore()

    def _domino(self, ctx, i, S, ax, th, font):
        b = self.B[i]
        hp, dp = b["h"] * S, b["d"] * S
        x = ax + b["x"] * S
        if hp < 1.2 or x + hp < -40 or x - dp - hp > W + 40:
            return
        ctx.save()
        ctx.translate(x, GY)
        ctx.rotate(th)
        r = min(dp * 0.22, 9.0)
        rounded_rect(ctx, -dp, -hp, dp, hp, r)
        g = cairo.LinearGradient(-dp, 0, 0, 0)
        g.add_color_stop_rgb(0, *IVORY)
        g.add_color_stop_rgb(1, *darken(IVORY, 0.2))
        ctx.set_source(g)
        ctx.fill()
        if hp > 14:
            ctx.save()
            rounded_rect(ctx, -dp, -hp, dp, hp, r)
            ctx.clip()
            ctx.rectangle(-dp, -hp, dp, hp * 0.085)                # colored head, alternating
            ctx.set_source_rgba(*self.pal[i % 2], 1)
            ctx.fill()
            ctx.rectangle(-dp, -hp * 0.5 - max(1.0, hp * 0.006), dp, max(2.0, hp * 0.012))
            ctx.set_source_rgba(*INK, 0.75)
            ctx.fill()
            ctx.restore()
        rounded_rect(ctx, -dp, -hp, dp, hp, r)
        ctx.set_source_rgba(*INK, 1)
        ctx.set_line_width(clamp(hp * 0.012, 1.0, 4.0))
        ctx.stroke()
        if dp >= 22:                                              # its place in the line
            fs = dp * (0.62 if i < 9 else 0.56)
            font.draw(ctx, str(i + 1), -dp / 2, -hp * 0.75 + fs * 0.4, fs, INK, anchor="c")
        ctx.restore()

    def _hit_fx(self, ctx, v, S, ax):
        for j in range(1, self.nb):
            d = v - self.vhit[j]
            if not 0 <= d < 0.45:
                continue
            wx, wy = self.sim["hitpos"][j]
            x, y = ax + wx * S, GY - wy * S
            if x > W + 100:
                continue
            k = d / 0.45
            big = 1.0 + 0.05 * j
            ctx.arc(x, y, (10 + 46 * ease_out(k)) * big, 0, TAU)
            ctx.set_source_rgba(*CREAM, 0.8 * (1 - k))
            ctx.set_line_width(5 * (1 - k) + 1)
            ctx.stroke()
            for a, sp, sz in self.puffs[j]:
                dist = (30 + 90 * sp) * big * ease_out(k)
                ctx.arc(x + math.cos(a) * dist, y + math.sin(a) * dist + 60 * k * k, max(0.3, 6 * sz * (1 - k)), 0, TAU)
                ctx.set_source_rgba(*self.pal[j % 2], 0.9 * (1 - k))
                ctx.fill()

    def _slam_fx(self, ctx, v, S, ax):
        d = v - self.v_slam
        if not 0 <= d < 2.2:
            return
        x0 = ax + self.B[-1]["x"] * S
        ln = TOWER_H * S
        k = d / 2.2
        for u, sp, sz, dr in self.dust:
            x = x0 + u * ln + dr * 60 * ease_out(k)
            y = GY - 8 - (40 + 150 * sp) * ease_out(min(1, d / 1.4))
            r = (16 + 44 * sz) * (0.5 + 0.8 * ease_out(k))
            glow(ctx, x, y, r, mix(CREAM, hex_rgb("#E2A07A"), 0.5), 0.34 * (1 - k) ** 1.5)
        if d < 0.7:
            kk = d / 0.7
            ctx.save()
            ctx.translate(x0 + ln / 2, GY)
            ctx.scale(1.0, 0.22)
            ctx.arc(0, 0, ln * (0.5 + 0.7 * ease_out(kk)), math.pi, TAU)
            ctx.restore()
            ctx.set_source_rgba(*CREAM, 0.8 * (1 - kk))
            ctx.set_line_width(7 * (1 - kk) + 1)
            ctx.stroke()

    def _hud(self, ctx, f, v, S, font):
        out = 1 - ramp(v, self.v_res - 0.15, self.v_res + 0.1)
        if out > 0:                                               # running count of toppled dominoes
            n = self.count(v)
            d = v - self.vhit[n - 1]
            pop = 1 + 0.16 * (1 - ease_out(min(1.0, d / 0.2))) if n > 1 else 1.0
            font.draw(ctx, "TOPPLED", HXL, 660, 44, MUTED, alpha=out, tracking=2.0, stroke=INK, stroke_w=5)
            ctx.save()
            ctx.translate(HXL, 850)
            ctx.scale(pop, pop)
            font.draw(ctx, str(n), 0, 0, 200, CREAM, alpha=out, stroke=INK, stroke_w=12, shadow=6)
            ctx.restore()
        # playback speed
        sp = float(self.fspeed[min(f, self.nframes - 1)])
        txt = f"x{sp:.2f}" if sp < 0.995 else (f"x{sp:.1f}" if sp < 9.95 else f"x{sp:.0f}")
        word = "SLOW MOTION" if sp < 0.9 else ("REAL TIME" if sp < 1.1 else "FAST FORWARD")
        y = 1516
        wt = font.width("TIME " + txt, 40, 1.0)
        rounded_rect(ctx, HXL - 14, y - 46, 54 + wt + 18 + font.width(word, 28, 1.2) + 30, 62, 14)
        ctx.set_source_rgba(0.03, 0.03, 0.07, 0.6)
        ctx.fill()
        for k in range(2):                                        # play-speed chevrons
            x = HXL + 4 + 20 * k
            polygon(ctx, [(x, y - 30), (x + 18, y - 15), (x, y)])
            ctx.set_source_rgba(*AMBER, 1)
            ctx.fill()
        font.draw(ctx, "TIME " + txt, HXL + 54, y, 40, AMBER, tracking=1.0)
        font.draw(ctx, word, HXL + 54 + wt + 18, y - 2, 28, CREAM, tracking=1.2)
        # scale bar
        val = 1e-3
        for e in range(-3, 5):
            for mth in (1, 2, 5):
                if mth * 10.0 ** e * S <= 230:
                    val = mth * 10.0 ** e
        bw = val * S
        lab = f"{val * 100:g} CM" if val < 0.995 else (f"{val:g} M" if val < 999 else f"{val / 1000:g} KM")
        x1 = 912.0
        ctx.move_to(x1 - bw, y - 26)
        ctx.line_to(x1 - bw, y - 8)
        ctx.line_to(x1, y - 8)
        ctx.line_to(x1, y - 26)
        ctx.set_source_rgba(*CREAM, 0.95)
        ctx.set_line_width(5)
        ctx.stroke()
        font.draw(ctx, lab, x1 - bw - 16, y - 4, 32, CREAM, anchor="r", tracking=1.2)

    def _result(self, ctx, v, font):
        k1 = ramp(v, self.v_res, self.v_res + 0.25)
        if k1 <= 0:
            return
        s = 0.75 + 0.25 * ease_out_back(k1)
        ctx.save()
        ctx.translate(CX, 880)
        ctx.scale(s, s)
        font.draw(ctx, str(self.N), 0, 0, 330, AMBER, anchor="c", alpha=k1, stroke=INK, stroke_w=16, shadow=8)
        ctx.restore()
        k2 = ramp(v, self.v_res + 0.3, self.v_res + 0.55)
        font.draw(ctx, "DOMINOES", CX, 990, 110, CREAM, anchor="c", alpha=k2, stroke=INK, stroke_w=11, shadow=5)
        k3 = ramp(v, self.v_res + 0.8, self.v_res + 1.05)
        font.draw(ctx, f"NUMBER {self.N} IS THE FIRST ONE TALLER THAN THE TOWER", CX, 1052, 36, CREAM, anchor="c",
                  alpha=k3, stroke=INK, stroke_w=7, tracking=0.8)
        k4 = ramp(v, self.v_res + 1.5, self.v_res + 1.75)
        font.draw(ctx, "WHAT WAS YOUR GUESS?", CX, 1128, 58, AMBER, anchor="c", alpha=k4, stroke=INK, stroke_w=9)

    # ------------------------------------------------------------ sound
    def audio(self):
        rng = np.random.default_rng(self.seed + 911)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        nb, N = self.nb, self.N

        def tt(d):
            return np.arange(int(d * sr)) / sr

        def thump(j, scale=1.0):
            """Impact of body j being hit: one whole tone lower per domino, longer and duller as they grow."""
            f0 = 1200.0 * 2 ** (-(j - 1) / 6.0)
            d = (0.10 + 0.04 * j) * scale
            t_ = tt(d)
            ph = TAU * np.cumsum(f0 * (1 + 0.5 * np.exp(-t_ * 60))) / sr
            body = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.32 * np.sin(3 * ph) + 0.2 * np.sin(5.1 * ph)) * \
                np.exp(-t_ * 34 / (1 + 0.28 * j))
            click = au.noise(rng, d, max(250.0, 2600 / (1 + 0.2 * j)), max(1600.0, 9000 / (1 + 0.13 * j))) * \
                np.exp(-t_ * 170 / (1 + 0.32 * j))
            return (body + 0.85 * click) * np.minimum(1.0, t_ / 0.002)

        def pan_of(v, wx):
            S, ax = self.camera(v)
            return clamp((ax + wx * S - 480) / 700, -0.6, 0.6)

        for j in range(1, nb):
            v = self.vhit[j]
            p = pan_of(v, self.sim["hitpos"][j][0])
            au.place(fx, thump(j), v, 0.34 + 0.022 * j, p)
            if j < N:
                au.place(fx, au.tick(2300 + 20 * j, 0.03), v + 0.02, 0.05, -0.3)   # counter click
        # dominoes landing on the next one (real contact onsets from the simulation)
        fv = np.arange(self.nframes) / FPS
        last = {}
        for t, i, s in self.sim["clacks"]:
            if t >= self.fT[-1]:
                continue
            v = float(np.interp(t, self.fT, fv))
            if v - last.get(i, -9) < 0.12:
                continue
            last[i] = v
            au.place(fx, thump(i + 1, 0.6), v, min(0.22, 0.06 + 0.25 * s), pan_of(v, self.B[i]["x"]))
        # rumble that grows with the size of what is falling, swelling toward each hit
        ts = np.arange(n) / sr
        br = np.cumsum(rng.standard_normal(n))
        br = au.bandpass(br - np.convolve(br, np.ones(2400) / 2400, mode="same"), 110, 700)
        br /= max(1e-9, np.abs(br).max())
        vk = np.array(self.vhit + [self.v_slam])
        idx = np.clip(np.searchsorted(vk, ts, side="right") - 1, 0, len(vk) - 2)
        frac = (ts - vk[idx]) / (vk[idx + 1] - vk[idx])
        env = (0.01 + 0.11 * ((idx + frac) / nb) ** 2.2) * (0.35 + 0.65 * np.clip(frac, 0, 1) ** 1.5)
        env *= ts < self.v_slam
        bed[:n, 0] += br * env
        bed[:n, 1] += br * env
        del br, ts, idx, frac, env
        # the last giant leans in, then the tower goes down
        v0 = self.vhit[N - 1]
        d = self.vhit[N] - v0
        au.place(fx, au.swept_noise(rng, d, 520, 150) * np.sin(np.pi * tt(d) / d) ** 1.5, v0, 0.16, 0.1)
        d = self.v_slam - self.vhit[N]
        au.place(fx, au.swept_noise(rng, d, 300, 900) * (tt(d) / d) ** 2, self.vhit[N], 0.22, 0.2)
        # slam: 40 Hz sine with a long decay (plus harmonics a phone can play) under a brown-noise wash
        t_ = tt(2.2)
        sub = (np.sin(TAU * 40 * t_) + 0.7 * np.sin(TAU * 80 * t_) * np.exp(-t_ * 2.5)
               + 0.5 * np.sin(TAU * 160 * t_) * np.exp(-t_ * 5) + 0.3 * np.sin(TAU * 240 * t_) * np.exp(-t_ * 7)) * \
            np.exp(-t_ * 1.7) * np.minimum(1.0, t_ / 0.004)
        au.place(fx, sub, self.v_slam, 0.42, 0.15)
        au.place(fx, au.boom(rng, 1.2, 140, 42), self.v_slam, 0.5, 0.15)
        w = np.cumsum(rng.standard_normal(int(1.8 * sr)))
        w = au.bandpass(w - np.convolve(w, np.ones(4800) / 4800, mode="same"), 80, 2600)
        w /= max(1e-9, np.abs(w).max())
        tw = tt(1.8)
        au.place(fx, w[:len(tw)] * np.minimum(1, tw / 0.01) * np.exp(-tw * 2.4), self.v_slam, 0.5, 0.2)
        # result: three marimba notes, then a soft bell on the question
        for k, mnote in enumerate((67, 72, 76)):
            au.place(fx, au.marimba(au.hz(mnote), 0.9), self.v_res + 0.3 * k + (0.2 if k == 2 else 0), 0.20, -0.2 + 0.2 * k)
        au.place(fx, au.bell(au.hz(79), 1.6), self.v_res + 1.5, 0.10)
        au.place(fx, au.marimba(au.hz(60), 0.6), self.v_res + 2.35, 0.12)
        # quiet bed
        au.place(bed, au.pad(au.hz(50), self.dur, 0.8, 0.8), 0.0, 0.014)
        au.place(bed, au.pad(au.hz(57), self.dur, 0.8, 0.8), 0.0, 0.010)
        mix_ = au.reverb(fx, rng, decay=0.9, wet=0.08) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]
        out = au.eq(out, self.EQ)
        pk = np.abs(out).max()
        return np.tanh(out * (self.DRIVE / max(1e-9, pk)))        # soft-clip the peaks so mastering reaches -14 LUFS

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        title = "Each Domino Is 1.5x Bigger… How Many to Topple a Tower? 🤔"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "How many dominoes does it take to topple a tower? Lock in a number before the first one lands!\n\n"
                "The first domino is 5 cm tall. Every next domino is 1.5 times bigger in every dimension, and each "
                "one stands half a domino-height away from the one that will hit it. One flick starts the line: "
                "each domino tips over its lower edge, falls under gravity and pushes the next, bigger one. The "
                "line ends at a 300 m tower, and the first domino that is taller than the tower knocks it over. "
                "The counter shows how many dominoes have toppled so far, the dashed line marks the height of the "
                "tower, and a coin, a hand, a person, a house and a tree show the scale as the camera pulls back. "
                "Big dominoes fall much more slowly than small ones, so the playback speed changes along the way "
                "and is always shown on screen: slow motion at the start, fast forward for the giants.\n"
                "This is a simplified rigid-block model: every domino is a solid block that pivots on its edge, "
                "contact is frictionless and does not bounce, and the tower is treated as a lighter block. "
                f"Physics, picture and sound are generated by code (seed {self.seed}), with no footage, samples or "
                "stock media.\n"
                "What number did you guess?\n"
                "#simulation #satisfying #dominoes"),
            tags=["domino", "dominoes", "chain reaction", "physics simulation", "exponential growth", "prediction",
                  "satisfying", "made with code", "simulation"],
            pinned_comment=("How many dominoes did you guess? Each one is 1.5 times bigger than the last, starting "
                            "at 5 cm, and the tower is 300 m. Post your number before you watch the end."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of abstract shapes; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=self.stats,
        )
