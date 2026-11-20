"""TIPPING POINT: a steel beam on a knife-edge pivot, a counterweight under the pivot pulls it back toward level.

Every 0.4 s a 1 to 5 kg weight drops at a random spot (size and number show the mass; a laser line marks the
spot before it falls). Torque is mass times distance from the pivot. Weights hold by friction until the tilt
passes 12 degrees (they creep from 9), then slide downhill; a weight that slides off the end falls into the pit.
The first tip to touch its contact pad (18 degrees) loses: its whole load pours into the pit.

simulate(seed) is deterministic and needs no rendering, so make.py can score hundreds of seeds for
drama (drama()) and render only the best one (Scene).
"""
import math
import os
import random

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_in_out, ease_out, ease_out_back, glow, hex_rgb, lerp,
                        lighten, mix, polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
SUB = 4
DT = 1.0 / (FPS * SUB)

DROP = 0.4                    # one weight every 0.4 s
L0 = -0.005                   # first landing just before frame 0: the loop's last frame shows it falling
FALL = 0.28                   # release-to-landing time
XMAX = 0.92                   # drop spots span +-92% of the half-length
TH_CREEP = math.radians(9.5)
TH_S = math.radians(12.0)     # slide threshold
TH_STOP = math.radians(10.0)  # sliding stops again below this
TH_T = math.radians(18.0)     # a tip touches its contact pad at this tilt
K = 80.0                      # counterweight restoring torque (per unit sin(tilt))
S = 15.5                      # torque-to-acceleration scale
I0 = 20.0                     # beam inertia
ZETA = 0.22                   # damping ratio
JM = 1.2                      # landing impulse
G = 6.0                       # sliding acceleration scale before contact
G2 = 14.0                     # avalanche after contact
VC = 0.03                     # creep speed at 12 degrees (half-lengths per second)
TIP = 1.026                   # beam end (half-lengths, including the end cap)
T_MAX = 27.0
SETTLE = 2.75                 # seconds from the beam emptying to the loop point
W_ZETA, W_PER = 0.16, 0.95    # whip-back oscillation after the load is gone

A = 380.0                     # half-length in px
NAMES = ("LEFT", "RIGHT")     # symbols: circle, square


def bsize(m):
    s = 27 + 6.5 * m
    return s, 0.72 * s        # width, body height (px)


def simulate(seed, record=False):
    rng = random.Random(seed * 9173 + 5)
    drops = []
    k = 0
    while L0 + k * DROP <= T_MAX:
        drops.append((L0 + k * DROP, rng.uniform(-XMAX, XMAX), rng.randint(1, 5)))
        k += 1
    xs, vs, ms, ids = [], [], [], []
    th = om = 0.0
    frames, offs = [], []
    contact = loser = t_empty = None
    sliding = False
    nd = 0
    step = 0
    nmax = int((T_MAX + 8) * FPS * SUB)
    w_om = TAU / W_PER
    t_end = None
    while step < nmax:
        t = step * DT
        if contact is None and t > T_MAX:
            break
        if True:                                  # weights already falling when a tip touches still land
            while nd < len(drops) and drops[nd][0] <= t + 1e-9 and (contact is None or drops[nd][0] - FALL < contact):
                tl, x, m = drops[nd]
                I = I0 + sum(mm * xx * xx for mm, xx in zip(ms, xs)) + m * x * x
                om += JM * m * x * math.cos(th) / I
                xs.append(x)
                vs.append(0.0)
                ms.append(m)
                ids.append(nd)
                nd += 1
        if step % SUB == 0:
            frames.append((th, om, tuple(zip(ids, xs)) if record else len(xs),
                           (sum(abs(v) for v in vs) / len(vs)) if vs else 0.0))
            if t_end is not None and t >= t_end - 1e-9:
                break
        if t_empty is None:
            if contact is None:
                T = sum(m * x for m, x in zip(ms, xs)) * math.cos(th)
                I = I0 + sum(m * x * x for m, x in zip(ms, xs))
                c = 2 * ZETA * math.sqrt(S * K * I)
                om += (S * (T - K * math.sin(th)) - c * om) / I * DT
                th += om * DT
                if abs(th) >= TH_T:
                    contact, loser = t, (1 if th > 0 else 0)
                    th = math.copysign(TH_T, th)
                    om = 0.0
            # weights: stick, creep or slide
            a = abs(th)
            dirn = 1.0 if th > 0 else -1.0
            if a > TH_S or contact is not None:
                sliding = True
            elif a < TH_STOP:
                sliding = False
            for i in range(len(xs)):
                if contact is not None:
                    vs[i] += dirn * G2 * (math.sin(a) - math.sin(TH_STOP)) * DT
                elif sliding:
                    vs[i] += dirn * G * (math.sin(a) - math.sin(TH_STOP)) * DT
                    if vs[i] * dirn < 0:
                        vs[i] *= math.exp(-DT / 0.05)
                else:
                    tgt = dirn * VC * ((a - TH_CREEP) / (TH_S - TH_CREEP)) ** 2 if a > TH_CREEP else 0.0
                    vs[i] += (tgt - vs[i]) * (1 - math.exp(-DT / 0.06))
                xs[i] += vs[i] * DT
            j = 0
            while j < len(xs):
                if abs(xs[j]) > TIP:
                    offs.append((ids[j], t, 1 if xs[j] > 0 else 0, vs[j], th))
                    del xs[j], vs[j], ms[j], ids[j]
                else:
                    j += 1
            if contact is not None and not xs:
                t_empty = t
                t_end = t_empty + SETTLE
        else:
            u = t - t_empty                       # the emptied beam whips back and rings down
            th0 = TH_T if loser == 1 else -TH_T
            env = math.exp(-W_ZETA * w_om * u)
            th = th0 * env * math.cos(w_om * u)
            k2 = ease_in_out(ramp(t, t_end - 0.9, t_end))
            th *= 1 - k2
            om = 0.0
        step += 1
    return dict(seed=seed, drops=drops, frames=frames, offs=offs, contact=contact, loser=loser,
                t_empty=t_empty, t_end=t_end, ndrops=nd)


def lead_series(frames):
    """Lower side per frame (1 = right down, 0 = left down, -1 = level); it flips only past 2.5 degrees."""
    out, lead = [], -1
    h = math.radians(2.5)
    for f, fr in enumerate(frames):
        th = fr[0]
        if th > h:
            lead = 1
        elif th < -h:
            lead = 0
        out.append(lead)
    return out


def drama(sim):
    frames = sim["frames"]
    tc = sim["contact"]
    lead = lead_series(frames)
    fc = int(round(tc * FPS)) if tc is not None else len(frames)
    rev = [round(f / FPS, 2) for f in range(1, min(fc, len(lead)))
           if lead[f - 1] >= 0 and lead[f] >= 0 and lead[f] != lead[f - 1]]
    # near-threshold moments: an excursion past 8 degrees that came back under 6 without touching
    near, peak, inside = [], 0.0, False
    for f in range(min(fc, len(frames))):
        a = abs(frames[f][0])
        if a > math.radians(8.0):
            inside = True
            peak = max(peak, a)
        elif inside and a < math.radians(6.0):
            near.append(dict(at_s=round(f / FPS, 2), peak_deg=round(math.degrees(peak), 1)))
            inside, peak = False, 0.0
    slid = sum(1 for n in near if n["peak_deg"] > 12.0)
    valid = tc is not None and sim["t_end"] is not None
    dur = sim["t_end"] if valid else None
    score = 0.0
    if not valid:
        score -= 100.0
    else:
        score += 4.0 if 24.5 <= dur <= 31.5 else -40.0 - 5 * abs(dur - 28)
        score += 1.5 if 20.0 <= tc <= 26.0 else 0.0
    nrev = len(rev)
    real = [r for r in rev if r >= 5.0]            # flips in the first seconds are cheap; count later ones
    score += 1.6 * min(len(real), 6) + (0.0 if len(real) >= 3 else -10.0)
    late = [r for r in rev if tc and r > tc / 2]
    score += 2.0 if late else -3.0
    last_gap = (tc - rev[-1]) if (rev and tc) else None
    if last_gap is not None and last_gap < 2.0:
        score -= 2.5                     # no last-second flips presented as typical
    score += 2.5 * min(len(near), 3) + (0.0 if near else -8.0)
    score += 1.0 * min(slid, 2)
    loser = sim["loser"]
    winner = 1 - loser if loser is not None else None
    half = lead[min(len(lead) - 1, fc // 2)] if lead else -1
    pre = lead[:fc]
    winner_was_lower_s = round(sum(1 for x in pre if x == winner) / FPS, 1) if winner is not None else 0
    score += (3.0 if winner_was_lower_s >= 6.0 else -3.0) if winner is not None else 0.0
    return dict(
        seed=sim["seed"], score=round(score, 3), valid=valid,
        winner=NAMES[winner] if winner is not None else None, winner_index=winner,
        loser=NAMES[loser] if loser is not None else None,
        contact_s=round(tc, 2) if tc is not None else None,
        load_empty_s=round(sim["t_empty"], 2) if sim["t_empty"] is not None else None,
        duration_s=round(dur, 2) if dur else None,
        tilt_reversals=nrev, reversal_times_s=rev,
        last_reversal_to_contact_s=round(last_gap, 2) if last_gap is not None else None,
        near_threshold_moments=near, slides_survived=slid,
        lower_side_at_half=NAMES[half] if half >= 0 else None,
        winner_was_lower_for_s=winner_was_lower_s,
        weights_dropped=sim["ndrops"],
    )


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
SIDE = [hex_rgb("#3CC8F5"), hex_rgb("#FF7043")]     # LEFT circle, RIGHT square
INK = hex_rgb("#091017")
CREAM = hex_rgb("#F1EEE6")
STEEL = hex_rgb("#8EA2B2")
MUTED = hex_rgb("#7F93A3")
AMBER = hex_rgb("#FFC23D")
DANGER = hex_rgb("#FF3B4E")
SLATE = hex_rgb("#26333D")
LASER = (0.55, 1.0, 0.75)
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")

PX, PY = 475.0, 960.0         # pivot
BT = 26.0                     # beam thickness
PADU = 345.0                  # contact pads sit under this point of the beam (px from the pivot)
PAD_W, PAD_H = 56.0, 22.0
PAD_TOP = PY + PADU * math.sin(TH_T) + BT / 2 * math.cos(TH_T)
BENCH_Y = PAD_TOP + PAD_H
BENCH_B = BENCH_Y + 88
PIT = [(4.0, 118.0), (832.0, 946.0)]                # pit openings (x ranges), outboard of the pads
PIT_B = BENCH_B
RAIL_Y = 494.0
PARK = PX
GX, GY, GR = 475.0, 1468.0, 188.0                   # tilt gauge
GSC = 3.0                                           # the gauge dial shows 3x the real angle
HX = 475.0                                          # header center (keeps text clear of the right margin)


def symbol(ctx, k, x, y, s):
    if k == 0:
        ctx.new_sub_path()
        ctx.arc(x, y, s, 0, TAU)
    else:
        ctx.rectangle(x - s * 0.88, y - s * 0.88, s * 1.76, s * 1.76)


def pad_x(k):
    return PX + (1 if k else -1) * PADU * math.cos(TH_T)


def gauge_pt(deg, r):
    a = math.radians(clamp(deg, -22, 22) * GSC)
    return GX + r * math.sin(a), GY - r * math.cos(a)


def make_background(font):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (yy / H)[..., None]
    img = np.array([0.070, 0.112, 0.140], np.float32) * (1 - v) + np.array([0.030, 0.046, 0.062], np.float32) * v
    d = np.hypot(xx - PX, yy - 820) / 820
    img += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array([0.050, 0.080, 0.090], np.float32)
    grid = ((xx % 40) < 1.5) | ((yy % 40) < 1.5)
    major = ((xx % 200) < 2) | ((yy % 200) < 2)
    img += (grid.astype(np.float32) * 0.014 + major.astype(np.float32) * 0.018)[..., None] * \
        np.array([0.6, 0.9, 1.0], np.float32)
    vig = np.clip(1.12 - 0.5 * (((xx - 520) / 720) ** 2 + ((yy - 960) / 1150) ** 2), 0.35, 1.0)
    img *= vig[..., None]
    img += np.random.default_rng(5).normal(0, 0.005, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    del img, yy, xx, v, d, grid, major, vig
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    # header
    font.draw(ctx, "LEFT OR RIGHT?", HX, 250, 106, CREAM, anchor="c", stroke=INK, stroke_w=10, shadow=6)
    font.draw(ctx, "WHICH SIDE STAYS UP?", HX, 326, 58, CREAM, anchor="c", stroke=INK, stroke_w=8, shadow=4)
    txt, size = "FIRST TIP TO TOUCH THE FLOOR LOSES", 38
    tw = font.width(txt, size, 1.2)
    rounded_rect(ctx, HX - tw / 2 - 22, 348, tw + 44, 54, 12)
    ctx.set_source_rgba(*CREAM, 1)
    ctx.fill()
    font.draw(ctx, txt, HX, 390, size, INK, anchor="c", tracking=1.2)
    font.draw(ctx, "1 TO 5 KG EVERY 0.4 S   •   PAST 12° THEY SLIDE", HX, 446, 30, MUTED, anchor="c", tracking=1.0)
    # rail for the dispenser carriage
    for col, lw in ((INK, 12), (STEEL, 6)):
        ctx.move_to(70, RAIL_Y)
        ctx.line_to(880, RAIL_Y)
        ctx.set_source_rgba(*col, 1)
        ctx.set_line_width(lw)
        ctx.stroke()
    for x in (70, 880):
        rounded_rect(ctx, x - 9, RAIL_Y - 18, 18, 36, 4)
        ctx.set_source_rgba(*darken(STEEL, 0.35), 1)
        ctx.fill()
    # bench, pits and pads
    g = cairo.LinearGradient(0, BENCH_Y, 0, BENCH_B)
    g.add_color_stop_rgb(0, *lighten(SLATE, 0.18))
    g.add_color_stop_rgb(0.08, *SLATE)
    g.add_color_stop_rgb(1, *darken(SLATE, 0.45))
    ctx.rectangle(0, BENCH_Y, W, BENCH_B - BENCH_Y)
    ctx.set_source(g)
    ctx.fill()
    ctx.rectangle(0, BENCH_Y, W, 3)
    ctx.set_source_rgba(*lighten(SLATE, 0.5), 1)
    ctx.fill()
    for x0, x1 in PIT:
        g = cairo.LinearGradient(0, BENCH_Y, 0, PIT_B)
        g.add_color_stop_rgb(0, 0.02, 0.03, 0.04)
        g.add_color_stop_rgb(1, 0.0, 0.0, 0.0)
        ctx.rectangle(x0, BENCH_Y - 1, x1 - x0, PIT_B - BENCH_Y + 1)
        ctx.set_source(g)
        ctx.fill()
        for xe in (x0, x1):
            ctx.rectangle(xe - 2, BENCH_Y - 1, 4, PIT_B - BENCH_Y + 1)
            ctx.set_source_rgba(*lighten(SLATE, 0.3), 1)
            ctx.fill()
        font.draw(ctx, "PIT", (x0 + x1) / 2, PIT_B - 14, 22, darken(STEEL, 0.4), anchor="c", tracking=2)
    for k in (0, 1):
        x = pad_x(k)
        rounded_rect(ctx, x - PAD_W / 2, PAD_TOP, PAD_W, PAD_H + 2, 5)
        ctx.set_source_rgba(*darken(SIDE[k], 0.25), 1)
        ctx.fill_preserve()
        ctx.set_source_rgba(*INK, 1)
        ctx.set_line_width(3)
        ctx.stroke()
        symbol(ctx, k, x, PAD_TOP + PAD_H / 2 + 1, 6.5)
        ctx.set_source_rgba(*INK, 1)
        ctx.fill()
        font.draw(ctx, "CONTACT 18°", x, BENCH_Y + 40, 22, darken(CREAM, 0.35), anchor="c", tracking=1)
    # stand: column and knife edge
    g = cairo.LinearGradient(PX - 20, 0, PX + 20, 0)
    g.add_color_stop_rgb(0, *darken(STEEL, 0.55))
    g.add_color_stop_rgb(0.45, *darken(STEEL, 0.15))
    g.add_color_stop_rgb(1, *darken(STEEL, 0.6))
    ctx.rectangle(PX - 17, PY + 34, 34, BENCH_Y - PY - 34)
    ctx.set_source(g)
    ctx.fill()
    rounded_rect(ctx, PX - 62, BENCH_Y - 14, 124, 16, 5)
    ctx.set_source_rgba(*darken(STEEL, 0.4), 1)
    ctx.fill()
    polygon(ctx, [(PX, PY - 4), (PX - 30, PY + 40), (PX + 30, PY + 40)])
    ctx.set_source_rgba(*darken(STEEL, 0.2), 1)
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK, 1)
    ctx.set_line_width(3)
    ctx.stroke()
    # gauge face
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    ctx.arc(GX, GY, GR + 30, math.radians(-90 - 74), math.radians(-90 + 74))
    ctx.set_source_rgba(0.02, 0.04, 0.06, 0.55)
    ctx.set_line_width(100)
    ctx.stroke()
    for d0, d1, col, al in ((-9.5, 9.5, STEEL, 0.45), (9.5, 12, AMBER, 0.9), (-12, -9.5, AMBER, 0.9),
                            (12, 18, DANGER, 0.9), (-18, -12, DANGER, 0.9)):
        ctx.arc(GX, GY, GR, math.radians(d0 * GSC) - math.pi / 2, math.radians(d1 * GSC) - math.pi / 2)
        ctx.set_source_rgba(*col, al)
        ctx.set_line_width(10)
        ctx.stroke()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for d in range(-18, 19, 3):
        big = d % 6 == 0
        x0, y0 = gauge_pt(d, GR + 8)
        x1, y1 = gauge_pt(d, GR + (26 if big else 17))
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y1)
        ctx.set_source_rgba(*CREAM, 0.8 if big else 0.4)
        ctx.set_line_width(4 if big else 2)
        ctx.stroke()
        if big:
            x2, y2 = gauge_pt(d, GR + 50)
            col = DANGER if abs(d) >= 12 else CREAM
            font.draw(ctx, f"{abs(d)}°", x2, y2 + 10, 28, col, anchor="c")
    for d in (-12, 12):                                  # threshold marks: weights slide past 12
        x0, y0 = gauge_pt(d, GR - 14)
        x1, y1 = gauge_pt(d, GR + 30)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y1)
        ctx.set_source_rgba(*DANGER, 1)
        ctx.set_line_width(6)
        ctx.stroke()
    for k in (0, 1):                                    # side symbols at the ends of the dial
        x, y = gauge_pt(-22 if k == 0 else 22, GR - 2)
        ctx.arc(x, y, 17, 0, TAU)
        ctx.set_source_rgba(*SIDE[k], 1)
        ctx.fill()
        symbol(ctx, k, x, y, 7)
        ctx.set_source_rgba(*INK, 1)
        ctx.fill()
    # side panels
    for k in (0, 1):
        x0 = 26.0 if k == 0 else 690.0
        rounded_rect(ctx, x0, 1228, 236, 196, 22)
        ctx.set_source_rgba(0.02, 0.04, 0.06, 0.62)
        ctx.fill_preserve()
        ctx.set_source_rgba(*SIDE[k], 0.9)
        ctx.set_line_width(4)
        ctx.stroke()
        ctx.arc(x0 + 46, 1282, 24, 0, TAU)
        ctx.set_source_rgba(*SIDE[k], 1)
        ctx.fill()
        symbol(ctx, k, x0 + 46, 1282, 10)
        ctx.set_source_rgba(*INK, 1)
        ctx.fill()
        font.draw(ctx, NAMES[k], x0 + 84, 1302, 52, SIDE[k], tracking=1.5)
    surf.flush()
    return surf, buf


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((60, -9.0), (160, -6.0), (300, -2.5), (900, 0.0), (3000, 3.0), (8000, 2.0), (16000, -1.0))
    DRIVE = 3.5

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed, record=True)
        if self.sim["t_end"] is None:
            raise ValueError(f"seed {seed}: no tip touches within {T_MAX} s")
        self.stats = drama(self.sim)
        self.loser = self.sim["loser"]
        self.winner = 1 - self.loser
        self.tc = self.sim["contact"]
        self.te = self.sim["t_empty"]
        self.dur = self.sim["t_end"]
        self.nframes = int(round(self.dur * FPS))
        self._res = None
        self._prep()

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    def _prep(self):
        sim = self.sim
        drops = sim["drops"]
        self.nrel = sim["ndrops"]
        self.size = [bsize(m) for _, _, m in drops]
        # stack heights per frame (render only): a weight rests on the highest earlier weight under it
        self.base = []
        prev = {}
        for fr in sim["frames"]:
            cur = {}
            placed = []
            for i, x in sorted(fr[2]):
                w, h = self.size[i]
                u0, u1 = x * A - w / 2 + 3, x * A + w / 2 - 3
                b = 0.0
                for v0, v1, top in placed:
                    if v0 < u1 and u0 < v1:
                        b = max(b, top)
                if i in prev and prev[i] > b:
                    b = max(b, prev[i] - 26.0)
                cur[i] = b
                placed.append((u0, u1, b + h))
            self.base.append(cur)
            prev = cur
        nfr = len(sim["frames"])
        self.rel = [tl - FALL for tl, _, _ in drops[:self.nrel]]
        self.tgt = []
        for k in range(self.nrel):
            tl, x, m = drops[k]
            f = min(nfr - 1, int(math.ceil(tl * FPS - 1e-6)))
            th = sim["frames"][f][0]
            b = self.base[f].get(k, 0.0)
            u, v = x * A, -(BT / 2 + b + self.size[k][1] / 2)
            c, s = math.cos(th), math.sin(th)
            self.tgt.append((PX + u * c - v * s, PY + u * s + v * c, th))
        self.offs = {o[0]: o for o in sim["offs"]}
        self.off_base = {}
        for i, t, side, v, th in sim["offs"]:
            f = max(0, min(nfr - 1, int(t * FPS)))
            self.off_base[i] = self.base[f].get(i, self.base[max(0, f - 1)].get(i, 0.0))
        rnd = random.Random(self.seed * 31 + 7)
        self.spin = {i: rnd.uniform(-7, 7) for i in self.offs}
        self.sparks = [(rnd.uniform(-math.pi, 0), rnd.uniform(250, 700), rnd.uniform(3, 6)) for _ in range(26)]

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            font = Font(FONT_PATH)
            bg, keep = make_background(font)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep, font=font)
        return self._res

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        self.draw(ctx, f, res)
        ctx.restore()
        res["surf"].flush()
        return bytes(res["surf"].get_data())

    # ------------------------------------------------------------ helpers
    def fr(self, f):
        fr = self.sim["frames"]
        return fr[max(0, min(f, len(fr) - 1))]

    def theta(self, f):
        return self.fr(f)[0]

    def carriage(self, t):
        """Carriage x, and the weight hanging under it as (index, alpha, scale) or None."""
        rel, n = self.rel, self.nrel
        if t >= self.dur - 0.62:
            tt = t - self.dur
            x = lerp(PARK, self.tgt[0][0], ease_in_out(ramp(tt, -0.55, rel[0] - 0.04)))
            hang = (0, 1.0, ease_out_back(ramp(tt, -0.60, -0.50))) if tt < rel[0] else None
            return x, hang
        k = -1
        while k + 1 < n and rel[k + 1] <= t:
            k += 1
        if k < 0:
            return self.tgt[0][0], None
        if k + 1 < n:
            x = lerp(self.tgt[k][0], self.tgt[k + 1][0], ease_in_out(ramp(t, rel[k] + 0.03, rel[k + 1] - 0.04)))
            return x, (k + 1, 1.0, ease_out_back(ramp(t, rel[k], rel[k] + 0.10)))
        t0 = max(rel[k], self.tc) + 0.15
        x = lerp(self.tgt[k][0], PARK, ease_in_out(ramp(t, t0, t0 + 0.7)))
        hang = None
        if k + 1 < len(self.sim["drops"]):
            al = 1 - ramp(t, self.tc, self.tc + 0.35)
            if al > 0:
                hang = (k + 1, al, ease_out_back(ramp(t, rel[k], rel[k] + 0.10)))
        return x, hang

    def shake(self, t):
        d = t - self.tc
        if 0 <= d < 0.6:
            a = 9.0 * math.exp(-d / 0.12)
            return a * math.sin(d * 71), a * math.cos(d * 59)
        return 0.0, 0.0

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, res):
        t = f / FPS
        font = res["font"]
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        th = self.theta(f)
        cx, hang = self.carriage(t)
        sx, sy = self.shake(t)
        ctx.save()
        ctx.translate(sx, sy)
        self._laser(ctx, t, cx, hang)
        self._pads(ctx, t, font)
        self._beam(ctx, f, t, th, font)
        self._falling(ctx, t, font)
        self._pit_falls(ctx, t, font)
        self._carriage(ctx, t, cx, hang, font)
        self._contact_fx(ctx, t, font)
        ctx.restore()
        self._gauge(ctx, f, t, th, font)
        self._panels(ctx, f, t, th, font)
        self._result(ctx, t, font)

    def _weight(self, ctx, i, x, y, rot=0.0, alpha=1.0, scale=1.0, font=None):
        w, h = self.size[i]
        m = self.sim["drops"][i][2]
        ctx.save()
        ctx.translate(x, y)
        if rot:
            ctx.rotate(rot)
        if scale != 1.0:
            ctx.scale(max(0.01, scale), max(0.01, scale))
        kw = w * 0.34
        rounded_rect(ctx, -kw / 2, -h / 2 - 9, kw, 12, 3)
        ctx.set_source_rgba(*darken(STEEL, 0.15), alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(*INK, alpha)
        ctx.set_line_width(2.5)
        ctx.stroke()
        rounded_rect(ctx, -w / 2, -h / 2, w, h, 7)
        g = cairo.LinearGradient(-w / 2, 0, w / 2, 0)
        g.add_color_stop_rgba(0, 0.66, 0.72, 0.77, alpha)
        g.add_color_stop_rgba(0.35, 0.90, 0.93, 0.95, alpha)
        g.add_color_stop_rgba(1, 0.50, 0.56, 0.62, alpha)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgba(*INK, alpha)
        ctx.set_line_width(3)
        ctx.stroke()
        fs = h * 0.78
        font.draw(ctx, str(m), 0, fs * 0.36, fs, INK, anchor="c", alpha=alpha)
        ctx.restore()

    def _laser(self, ctx, t, cx, hang):
        if hang is None:
            return
        i, al, sc = hang
        if i >= self.nrel:
            return
        tx, ty, _ = self.tgt[i]
        h = self.size[i][1]
        y0 = RAIL_Y + 40 + h + 4
        y1 = ty + h / 2
        if y1 <= y0:
            return
        k = al * ramp(sc, 0.3, 1.0)
        ctx.save()
        ctx.set_dash([10, 9], -(t * 120) % 19)
        ctx.move_to(cx, y0)
        ctx.line_to(cx, y1)
        ctx.set_source_rgba(*LASER, 0.42 * k)
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.restore()
        ctx.move_to(cx - 14, y1)
        ctx.line_to(cx + 14, y1)
        ctx.set_source_rgba(*LASER, 0.8 * k)
        ctx.set_line_width(4)
        ctx.stroke()

    def _carriage(self, ctx, t, cx, hang, font):
        if hang is not None:
            i, al, sc = hang
            w, h = self.size[i]
            self._weight(ctx, i, cx, RAIL_Y + 40 + h / 2, alpha=al, scale=sc, font=font)
        ctx.move_to(cx - 16, RAIL_Y + 14)
        ctx.line_to(cx - 10, RAIL_Y + 34)
        ctx.move_to(cx + 16, RAIL_Y + 14)
        ctx.line_to(cx + 10, RAIL_Y + 34)
        ctx.set_source_rgba(*darken(STEEL, 0.1), 1)
        ctx.set_line_width(5)
        ctx.stroke()
        rounded_rect(ctx, cx - 44, RAIL_Y - 17, 88, 34, 9)
        g = cairo.LinearGradient(0, RAIL_Y - 17, 0, RAIL_Y + 17)
        g.add_color_stop_rgb(0, *lighten(SLATE, 0.35))
        g.add_color_stop_rgb(1, *darken(SLATE, 0.3))
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgba(*INK, 1)
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.arc(cx, RAIL_Y, 6, 0, TAU)
        ctx.set_source_rgba(*LASER, 0.9 if hang else 0.3)
        ctx.fill()

    def _falling(self, ctx, t, font):
        """Weights between release and landing."""
        tt = t - self.dur if t >= self.dur - 0.62 else t
        for i in range(self.nrel):
            r = self.rel[i]
            if r > tt + 0.01:
                break
            if not (r <= tt < r + FALL):
                continue
            u = (tt - r) / FALL
            tx, ty, th = self.tgt[i]
            h = self.size[i][1]
            y0 = RAIL_Y + 40 + h / 2
            self._weight(ctx, i, tx, lerp(y0, ty, u * u), rot=th * u * u, font=font)

    def _beam(self, ctx, f, t, th, font):
        ctx.save()
        ctx.translate(PX, PY)
        ctx.rotate(th)
        # counterweight under the pivot (rotates with the beam)
        ctx.move_to(0, 0)
        ctx.line_to(0, 92)
        ctx.set_source_rgba(*INK, 1)
        ctx.set_line_width(12)
        ctx.stroke_preserve()
        ctx.set_source_rgba(*STEEL, 1)
        ctx.set_line_width(6)
        ctx.stroke()
        ctx.arc(0, 100, 22, 0, TAU)
        g = cairo.RadialGradient(-7, 92, 2, 0, 100, 22)
        g.add_color_stop_rgb(0, *lighten(STEEL, 0.4))
        g.add_color_stop_rgb(1, *darken(STEEL, 0.5))
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgba(*INK, 1)
        ctx.set_line_width(3)
        ctx.stroke()
        # beam body
        L = TIP * A
        rounded_rect(ctx, -L, -BT / 2, 2 * L, BT, 6)
        g = cairo.LinearGradient(0, -BT / 2, 0, BT / 2)
        g.add_color_stop_rgb(0, *lighten(STEEL, 0.35))
        g.add_color_stop_rgb(0.5, *STEEL)
        g.add_color_stop_rgb(1, *darken(STEEL, 0.5))
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgba(*INK, 1)
        ctx.set_line_width(3)
        ctx.stroke()
        for k in (0, 1):                                  # painted halves and end caps
            ctx.rectangle(4 if k else -(L - 30), BT / 2 - 9, L - 34, 6)
            ctx.set_source_rgba(*SIDE[k], 0.9)
            ctx.fill()
            rounded_rect(ctx, (L - 26) if k else -L, -BT / 2 - 3, 26, BT + 6, 6)
            ctx.set_source_rgba(*SIDE[k], 1)
            ctx.fill_preserve()
            ctx.set_source_rgba(*INK, 1)
            ctx.set_line_width(3)
            ctx.stroke()
            symbol(ctx, k, (L - 13) if k else -(L - 13), 0, 6)
            ctx.set_source_rgba(*INK, 1)
            ctx.fill()
        for j in range(-9, 10):                           # graduations: distance from the pivot
            u = j * A / 10
            big = j % 5 == 0
            ctx.move_to(u, -BT / 2 + 2)
            ctx.line_to(u, -BT / 2 + (11 if big else 6))
            ctx.set_source_rgba(*INK, 0.8)
            ctx.set_line_width(3 if big else 2)
            ctx.stroke()
        # weights on the beam
        st = self.fr(f)
        base = self.base[min(f, len(self.base) - 1)]
        for i, x in sorted(st[2]):
            w, h = self.size[i]
            b = base.get(i, 0.0)
            self._weight(ctx, i, x * A, -(BT / 2 + b + h / 2), font=font)
            d = t - self.sim["drops"][i][0]
            if 0 <= d < 0.22:                             # landing puff
                k = d / 0.22
                for sgn in (-1, 1):
                    ctx.move_to(x * A + sgn * (w / 2 + 4 + 18 * k), -(BT / 2 + b) - 2)
                    ctx.line_to(x * A + sgn * (w / 2 + 14 + 30 * k), -(BT / 2 + b) - 10 - 6 * k)
                ctx.set_source_rgba(*CREAM, 0.7 * (1 - k))
                ctx.set_line_width(3)
                ctx.stroke()
        ctx.restore()
        ctx.arc(PX, PY, 11, 0, TAU)                       # pivot pin
        ctx.set_source_rgba(*INK, 1)
        ctx.fill()
        ctx.arc(PX, PY, 6, 0, TAU)
        ctx.set_source_rgba(*lighten(STEEL, 0.4), 1)
        ctx.fill()

    def _pit_falls(self, ctx, t, font):
        """Weights that slid off an end: thrown off the tip, falling into the pit."""
        ctx.save()
        ctx.rectangle(0, 0, W, BENCH_Y)
        for x0, x1 in PIT:
            ctx.rectangle(x0 + 2, BENCH_Y - 1, x1 - x0 - 4, PIT_B - BENCH_Y)
        ctx.clip()
        for i, (_, to, side, v, th) in self.offs.items():
            d = t - to
            if not 0 <= d < 0.9:
                continue
            w, h = self.size[i]
            b = self.off_base[i]
            sgn = 1 if side else -1
            u, vv = sgn * TIP * A, -(BT / 2 + b + h / 2)
            c, s = math.cos(th), math.sin(th)
            x0, y0 = PX + u * c - vv * s, PY + u * s + vv * c
            sp = abs(v) * A
            x = x0 + sgn * min(260.0, sp * c) * d
            y = y0 + sp * abs(s) * d + 0.5 * 2600 * d * d
            lo, hi = PIT[side]
            x = clamp(x, lo + w / 2 + 2, hi - w / 2 - 2) if side else clamp(x, lo + w / 2 + 2, x)
            if side == 0:
                x = max(x, lo + w / 2 + 2)
            if y - h / 2 > PIT_B:
                continue
            dark = ramp(y, BENCH_Y, PIT_B)
            self._weight(ctx, i, x, y, rot=th + self.spin[i] * d, alpha=1 - 0.75 * dark, font=font)
        ctx.restore()

    def _pads(self, ctx, t, font):
        if t < self.tc:
            return
        k = self.loser
        d = t - self.tc
        al = 1 - ramp(t, self.dur - 0.9, self.dur - 0.4)
        if al <= 0:
            return
        x = pad_x(k)
        glow(ctx, x, PAD_TOP + 6, 120, lighten(DANGER, 0.2), 0.55 * al * (0.75 + 0.25 * math.exp(-d / 0.3)))
        rounded_rect(ctx, x - PAD_W / 2, PAD_TOP, PAD_W, PAD_H + 2, 5)
        ctx.set_source_rgba(*lighten(DANGER, 0.25), al)
        ctx.fill()
        symbol(ctx, k, x, PAD_TOP + PAD_H / 2 + 1, 6.5)
        ctx.set_source_rgba(*INK, al)
        ctx.fill()

    def _contact_fx(self, ctx, t, font):
        d = t - self.tc
        if not 0 <= d < 1.2:
            return
        x, y = pad_x(self.loser), PAD_TOP
        ctx.arc(x, y, 18 + 150 * ease_out(min(1, d / 0.45)), 0, TAU)
        ctx.set_source_rgba(*lighten(DANGER, 0.4), max(0, 0.9 * (1 - d / 0.45)))
        ctx.set_line_width(8)
        ctx.stroke()
        for a, v, sz in self.sparks:
            dist = v * (1 - math.exp(-d * 4)) / 4
            px, py = x + math.cos(a) * dist, y + math.sin(a) * dist + 300 * d * d
            ctx.arc(px, py, max(0.1, sz * (1 - d / 1.2)), 0, TAU)
            ctx.set_source_rgba(1.0, 0.85, 0.5, 1 - d / 1.2)
            ctx.fill()
        if d < 1.1:
            s = 58 * ease_out_back(min(1, d / 0.2))
            font.draw(ctx, "CONTACT!", clamp(x, 170, 780), y - 150 - 40 * d, max(1, s), CREAM, anchor="c",
                      alpha=1 - ramp(d, 0.8, 1.1), stroke=INK, stroke_w=9)

    def _gauge(self, ctx, f, t, th, font):
        deg = math.degrees(th)
        a = abs(deg)
        pre = t < self.tc
        if a > 8.0 and pre:                               # zone glow near the threshold (slow, low contrast)
            k = ramp(a, 8.0, 12.0)
            side = 1 if deg > 0 else -1
            x, y = gauge_pt(side * 13.5, GR)
            glow(ctx, x, y, 120, DANGER if a > 12 else AMBER, (0.22 + 0.08 * math.sin(t * 7)) * k)
        x0, y0 = gauge_pt(deg, 112)
        x1, y1 = gauge_pt(deg, GR + 14)
        col = DANGER if a >= 12 else (AMBER if a >= 9.5 else CREAM)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y1)
        ctx.set_source_rgba(*INK, 1)
        ctx.set_line_width(13)
        ctx.stroke_preserve()
        ctx.set_source_rgba(*col, 1)
        ctx.set_line_width(7)
        ctx.stroke()
        ctx.arc(x1, y1, 7, 0, TAU)
        ctx.fill()
        if not pre and t < self.te + 0.2:
            word, wc = "CONTACT", DANGER
        elif a >= 12 and pre:
            word, wc = "SLIDING", DANGER
        elif a >= 9.5 and pre:
            word, wc = "CREEPING", AMBER
        else:
            word, wc = "TILT", MUTED
        font.draw(ctx, word, GX, GY - 66, 30, wc, anchor="c", tracking=2)
        shown = abs(round(deg, 1))
        font.draw(ctx, f"{shown:.1f}°", GX + 8, GY + 8, 72, CREAM, anchor="c", stroke=INK, stroke_w=6)
        if shown >= 0.1:                                  # which side is lower
            k = 1 if deg > 0 else 0
            sx = GX + (88 if k else -88)
            ctx.arc(sx, GY - 18, 15, 0, TAU)
            ctx.set_source_rgba(*SIDE[k], 1)
            ctx.fill()
            symbol(ctx, k, sx, GY - 18, 6)
            ctx.set_source_rgba(*INK, 1)
            ctx.fill()
        font.draw(ctx, "SLIDE AT 12°  •  TOUCH AT 18°", GX, GY + 54, 26, MUTED, anchor="c", tracking=1)

    def _panels(self, ctx, f, t, th, font):
        st = self.fr(f)
        kg = [0, 0]
        drops = self.sim["drops"]
        for i, x in st[2]:
            kg[1 if x > 0 else 0] += drops[i][2]
        deg = math.degrees(th)
        for k in (0, 1):
            x0 = 26.0 if k == 0 else 690.0
            font.draw(ctx, f"{kg[k]} KG", x0 + 26, 1378, 60, CREAM)
            lower = (deg > 1.5) if k else (deg < -1.5)
            if self.tc <= t < self.dur - 0.6:
                word, col = ("TOUCHED", DANGER) if k == self.loser else ("STAYS UP", SIDE[k])
                font.draw(ctx, word, x0 + 26, 1410, 30, col, tracking=1.5,
                          alpha=1 - ramp(t, self.dur - 0.9, self.dur - 0.6))
            elif lower and t < self.tc:
                polygon(ctx, [(x0 + 28, 1388), (x0 + 50, 1388), (x0 + 39, 1405)])
                ctx.set_source_rgba(*SIDE[k], 1)
                ctx.fill()
                font.draw(ctx, "LOWER", x0 + 58, 1410, 30, SIDE[k], tracking=1.5)

    def _result(self, ctx, t, font):
        t1 = max(self.tc + 0.45, self.te - 0.15)        # after the avalanche has cleared the beam
        k1 = ramp(t, t1, t1 + 0.3)
        out = 1 - ramp(t, self.dur - 0.95, self.dur - 0.6)
        if k1 <= 0 or out <= 0:
            return
        w = self.winner
        s = 0.8 + 0.2 * ease_out_back(k1)
        ctx.save()
        ctx.translate(HX, 650)
        ctx.scale(s, s)
        font.draw(ctx, f"{NAMES[w]} STAYS UP", 0, 0, 100, SIDE[w], anchor="c", alpha=k1 * out, stroke=INK,
                  stroke_w=12, shadow=5)
        ctx.restore()
        k2 = ramp(t, t1 + 0.4, t1 + 0.7)
        font.draw(ctx, f"{NAMES[self.loser]} TIP TOUCHED FIRST", HX, 718, 44, CREAM, anchor="c",
                  alpha=k2 * out, stroke=INK, stroke_w=8)
        k3 = ramp(t, t1 + 0.9, t1 + 1.2)
        font.draw(ctx, "DID YOU CALL IT?", HX, 784, 50, AMBER, anchor="c", alpha=k3 * out, stroke=INK, stroke_w=8)

    # ------------------------------------------------------------ sound
    def audio(self):
        sim = self.sim
        rng = np.random.default_rng(self.seed + 555)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        tc, te = self.tc, self.te

        def tt(d):
            return np.arange(int(d * sr)) / sr

        def thunk(m, dur=0.32):
            t_ = tt(dur)
            f0 = 560.0 / (1 + 0.4 * (m - 1))            # heavy is low
            f = f0 * (1 + 0.7 * np.exp(-t_ * 45))
            ph = TAU * np.cumsum(f) / sr
            x = (np.sin(ph) + 0.6 * np.sin(2 * ph) + 0.45 * np.sin(3 * ph) + 0.3 * np.sin(4.2 * ph)) * np.exp(-t_ * (12 - m))
            x = x * np.minimum(1.0, t_ / 0.002)
            return x + 0.7 * au.noise(rng, dur, 1500, 7000) * np.exp(-t_ * 90)

        cache = {}
        for k in range(self.nrel):
            tl, x, m = sim["drops"][k]
            if m not in cache:
                cache[m] = thunk(m)
            pan = clamp(x * 0.7, -0.75, 0.75)
            au.place(fx, cache[m], tl if tl >= 0 else tl + self.dur, 0.42 + 0.06 * m, pan)
            if self.rel[k] >= 0:
                au.place(fx, au.tick(2600, 0.03), self.rel[k], 0.04, pan)      # release click
        au.place(fx, au.tick(2600, 0.03), self.dur + self.rel[0], 0.04, clamp(sim["drops"][0][1] * 0.7, -0.75, 0.75))
        # continuous tilt tone, creak and slide hiss, driven by the recorded motion
        frs = sim["frames"]
        ft = np.arange(len(frs)) / FPS
        ts = np.arange(n) / sr
        th = np.interp(ts, ft, [q[0] for q in frs])
        deg = np.minimum(np.abs(np.degrees(th)), 18.0)
        omg = np.abs(np.gradient(th) * sr)
        omg = np.convolve(omg, np.ones(2400) / 2400, mode="same")
        ph = TAU * np.cumsum(190 + 24 * deg) / sr
        tone = (np.sin(ph) + 0.35 * np.sin(2 * ph)) * (0.010 + 0.05 * (deg / 18) ** 2)
        fc = 85 * (1 + 0.25 * np.sin(TAU * 6.5 * ts))
        saw = 2 * ((np.cumsum(fc) / sr) % 1.0) - 1
        creak = au.bandpass(saw, 300, 2400) * 0.07 * np.minimum(1.0, omg / 1.2) ** 1.5
        lean = np.clip(th * 2, -1, 1)
        slide = np.interp(ts, ft, [q[3] for q in frs])
        hiss = au.noise(rng, n / sr, 900, 5000)[:n] * np.minimum(1.0, slide / 0.25) * 0.20
        bed[:n, 0] += tone + hiss + creak * (0.6 - 0.3 * lean)
        bed[:n, 1] += tone + hiss + creak * (0.6 + 0.3 * lean)
        del th, deg, omg, ph, tone, fc, saw, creak, lean, slide, hiss, ts
        # warning ticks while the tilt is in the creep or slide zone, faster the closer it gets
        tk = 0.0
        while tk < tc:
            d = abs(math.degrees(self.theta(int(tk * FPS))))
            if d >= 9.5:
                au.place(fx, au.tick(1900, 0.05), tk, 0.10 + 0.08 * ramp(d, 9.5, 14))
                tk += 0.24 if d < 12 else 0.15
            else:
                tk += 0.05
        # contact: buzzer and boom
        tb = tt(0.5)
        buzz = sum(np.sin(TAU * 175 * hh * tb) / hh for hh in (1, 3, 5, 7)) * np.minimum(1, tb / 0.005) * \
            np.minimum(1.0, (0.5 - tb) / 0.08)
        pan = -0.5 if self.loser == 0 else 0.5
        au.place(fx, buzz, tc, 0.22, pan)
        au.place(fx, au.boom(rng, 0.9, 120, 45), tc, 0.32, pan)
        # avalanche: a dense run of thunks over a brown-noise rumble
        for i, (_, to, side, v, th_) in sorted(self.offs.items()):
            m = sim["drops"][i][2]
            au.place(fx, thunk(m, 0.25) * (0.8 + 0.4 * rng.random()), to + 0.16 + 0.05 * rng.random(), 0.26,
                     -0.6 if side == 0 else 0.6)
        rl = max(0.5, te - tc + 0.6)
        br = np.cumsum(rng.standard_normal(int(rl * sr)))
        br = au.bandpass(br - np.convolve(br, np.ones(4800) / 4800, mode="same"), 60, 1800)
        br /= max(1e-9, np.abs(br).max())
        tr = tt(rl)
        au.place(fx, br[:len(tr)] * np.minimum(1, tr / 0.05) * np.minimum(1, (rl - tr) / 0.4), tc, 0.5, pan * 0.6)
        # whip back and settle: a swoosh, then one soft bell
        au.place(fx, au.swept_noise(rng, 0.5, 500, 3000) * np.sin(np.pi * tt(0.5) / 0.5), te, 0.3)
        au.place(fx, au.bell(au.hz(79), 2.0), te + 1.25, 0.11)
        au.place(fx, au.bell(au.hz(86), 1.6), te + 1.33, 0.05, 0.2)
        # quiet bed
        au.place(bed, au.pad(au.hz(64), self.dur, 1.0, 0.8), 0.0, 0.014)
        au.place(bed, au.pad(au.hz(71), self.dur, 1.0, 0.8), 0.0, 0.010)
        mix_ = au.reverb(fx, rng, decay=0.6, wet=0.05) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]
        out = au.eq(out, self.EQ)
        pk = np.abs(out).max()
        out = np.tanh(out * (self.DRIVE / max(1e-9, pk)))     # soft-clip the thunk peaks so mastering reaches -14 LUFS
        return out                                             # (rounded tops keep the AAC re-encode under -1 dBTP)

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        title = "Every Drop Tips the Balance… Which Side Stays Up?"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "A steel beam rests on a knife-edge pivot, and a counterweight hanging under the pivot pulls it "
                "back toward level. Every 0.4 seconds the carriage drops a weight of 1 to 5 kg at a random spot "
                "along the beam: its size and the number on it show its mass, and a laser line marks where it will "
                "land before it falls. Each weight turns the beam by its mass times its distance from the pivot, so "
                "a light weight near a tip can outweigh a heavy one near the middle, and every drop swings the beam "
                "one way or the other. The weights hold by friction until the tilt passes 12 degrees, the red line "
                "on the gauge; from 9.5 degrees they start to creep, and past 12 they slide downhill and tip the "
                "beam further. A weight that slides off an end falls into the pit. The first tip to touch its "
                "contact pad on the floor, at 18 degrees, loses: its whole load pours into the pit. The other side "
                "stays up and wins.\n"
                f"Physics, picture and sound are generated by code (seed {self.seed}), with no footage, samples or "
                "stock media.\n"
                "Left or right: which side did you pick?\n"
                "#simulation #satisfying #physics"),
            tags=["physics simulation", "balance beam", "lever", "torque", "tipping point", "prediction",
                  "satisfying", "made with code", "simulation"],
            pinned_comment=("Left or right: which side stays up? Pick before the first weight lands. Mass times "
                            "distance decides it, and past 12 degrees everything slides."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of abstract shapes; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=self.stats,
        )
