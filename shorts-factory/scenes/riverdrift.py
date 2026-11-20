"""RIVER DRIFT: four unsteered floats ride one velocity field down a river. First across the line wins.

The water is a single field: a steady current (faster mid-river, slower at the banks, with a gentle meander),
ten slowly drifting whirlpools and three rocks. Every float is identical and follows the local flow with a
little inertia (relaxation time 0.4 s); floats bump each other, the rocks and the banks. Nothing is steered.
Thousands of tracer streaks drawn from the same field make the current visible.

simulate(seed) is deterministic and needs no rendering, so make.py can score many seeds (drama()) and render
only the best one (Scene). The seed sets the whirlpool and rock layout and which float starts in which lane.
"""
import math
import os
import random

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import TAU, Font, clamp, darken, ease_in_out, ease_out, glow, hex_rgb, lerp, lighten, ramp, rounded_rect

W, H, FPS = 1080, 1920, 30
SUB = 4
DT = 1.0 / (FPS * SUB)

XL, XR = 118.0, 940.0         # river banks (the right 12% of the frame holds nothing essential)
XC, HW = (XL + XR) / 2, (XR - XL) / 2
LEN = 8000.0                  # start line at y = 0, finish line at y = LEN (px)
U0 = 440.0                    # mid-river current (px/s)
PROF = 0.14                   # the current is this much slower at the banks
MEAND = 110.0                 # meander: sideways flow amplitude (px/s)
NV = 10                       # whirlpools
SINK = 0.22                   # inward pull of a whirlpool, as a fraction of its swirl
TAU_F = 0.40                  # float inertia: relaxation time toward the local flow (s)
RF = 24.0                     # float collision radius
T_CAP = 31.0
NAMES = ("ARROW", "LEAF", "KITE", "TWIN")

ANCHOR = 960.0                # the leader sits at this screen y
SLOW = 0.35                   # photo-finish slow motion rate
CREEP = 0.03                  # the "freeze" still creeps at this rate
FREEZE = 1.8
REW = 3.6                     # rewind along the winner's path
PRE = 0.5                     # floats wait on the start line before the loop point


# ====================================================================== the course and the field
def make_course(seed):
    rng = random.Random(seed * 7919 + 11)
    lanes = [0, 1, 2, 3]
    rng.shuffle(lanes)                                    # lanes[k] = float in lane k (rotates between episodes)
    vort = []
    y0, y1 = 620.0, LEN - 700.0
    seg = (y1 - y0) / (NV - 1)
    for i in range(NV):
        cy = y0 + seg * (i + rng.uniform(0.2, 0.8)) if i < NV - 1 else LEN - rng.uniform(360.0, 560.0)
        a = rng.uniform(80.0, 120.0)
        s = rng.uniform(380.0, 620.0) * rng.choice((-1.0, 1.0))
        ax, ay = rng.uniform(50.0, 150.0), rng.uniform(30.0, 90.0)
        cx = rng.uniform(XL + 70 + ax, XR - 70 - ax)
        vort.append((cx, cy, a, s, ax, TAU / rng.uniform(7, 13), rng.uniform(0, TAU),
                     ay, TAU / rng.uniform(9, 15), rng.uniform(0, TAU)))
    rocks = []
    for fr in (0.2, 0.5, 0.8):
        for _ in range(60):
            ry = LEN * fr + rng.uniform(-420.0, 420.0)
            rx = rng.uniform(XL + 110, XR - 110)
            rr = rng.uniform(46.0, 64.0)
            if all(math.hypot(rx - v[0], ry - v[1]) > 340 for v in vort):
                break
        rocks.append((rx, ry, rr))
    return dict(lanes=lanes, vort=vort, rocks=rocks, k=TAU / 1500.0, ph=rng.uniform(0, TAU))


def vpos(v, t):
    return v[0] + v[4] * math.sin(v[5] * t + v[6]), v[1] + v[7] * math.sin(v[8] * t + v[9])


def flow(x, y, t, C):
    """Water velocity at one point (the simulation uses this)."""
    s = (x - XC) / HW
    s2 = s * s
    vy = U0 * (1 - PROF * s2)
    vx = MEAND * math.sin(C["k"] * y + C["ph"]) * (1 - s2)
    for v in C["vort"]:
        a = v[2]
        dy = y - (v[1] + v[7] * math.sin(v[8] * t + v[9]))
        if dy > 3.6 * a or dy < -3.6 * a:
            continue
        dx = x - (v[0] + v[4] * math.sin(v[5] * t + v[6]))
        q = (dx * dx + dy * dy) / (a * a)
        if q > 13.0:
            continue
        g = v[3] / a * math.exp(0.5 * (1 - q))
        ag = abs(g) * SINK
        vx += -dy * g - ag * dx
        vy += dx * g - ag * dy
    for rx, ry, rr in C["rocks"]:
        dy = y - ry
        if dy > 6 * rr or dy < -6 * rr:
            continue
        dx = x - rx
        r2 = dx * dx + dy * dy
        if r2 <= rr * rr:
            return 0.0, 0.0
        w = rr * rr / r2
        vn = (vx * dx + vy * dy) / r2
        vx, vy = vx * (1 + w) - 2 * w * vn * dx, vy * (1 + w) - 2 * w * vn * dy
    return vx, vy


def flow_np(x, y, t, C):
    """The same field for arrays of points (tracer streaks)."""
    s2 = ((x - XC) / HW) ** 2
    vy = U0 * (1 - PROF * s2)
    vx = MEAND * np.sin(C["k"] * y + C["ph"]) * (1 - s2)
    lo, hi = float(y.min()), float(y.max())
    for v in C["vort"]:
        cx, cy = vpos(v, t)
        a = v[2]
        if cy < lo - 3.6 * a or cy > hi + 3.6 * a:
            continue
        dx, dy = x - cx, y - cy
        q = (dx * dx + dy * dy) / (a * a)
        g = v[3] / a * np.exp(0.5 * (1 - np.minimum(q, 40.0)))
        g[q > 13.0] = 0.0
        ag = np.abs(g) * SINK
        vx = vx - dy * g - ag * dx
        vy = vy + dx * g - ag * dy
    for rx, ry, rr in C["rocks"]:
        if ry < lo - 6 * rr or ry > hi + 6 * rr:
            continue
        dx, dy = x - rx, y - ry
        r2 = np.maximum(dx * dx + dy * dy, 1e-6)
        w = np.where(np.abs(dy) > 6 * rr, 0.0, rr * rr / r2)
        vn = (vx * dx + vy * dy) / r2
        vx, vy = vx * (1 + w) - 2 * w * vn * dx, vy * (1 + w) - 2 * w * vn * dy
        ins = r2 <= rr * rr
        vx[ins] = 0.0
        vy[ins] = 0.0
    return vx, vy


def simulate(seed):
    C = make_course(seed)
    x = [0.0] * 4
    for k, i in enumerate(C["lanes"]):
        x[i] = XL + (XR - XL) * (k + 0.5) / 4
    y = [0.0] * 4
    vx = [0.0] * 4
    vy = [0.0] * 4
    fin = [None] * 4
    rec, bumps, hits = [], [], []
    rocks = C["rocks"]
    step = 0
    t_stop = first = None
    kf = DT / TAU_F
    while True:
        t = step * DT
        rec.append((x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]))
        if (t_stop is not None and t >= t_stop) or t > T_CAP or (first is not None and t > first + 6.0):
            break
        for i in range(4):
            ux, uy = flow(x[i], y[i], t, C)
            vx[i] += (ux - vx[i]) * kf
            vy[i] += (uy - vy[i]) * kf
            yp = y[i]
            x[i] += vx[i] * DT
            y[i] += vy[i] * DT
            if x[i] < XL + RF:
                x[i] = XL + RF
                vx[i] = abs(vx[i]) * 0.3
            elif x[i] > XR - RF:
                x[i] = XR - RF
                vx[i] = -abs(vx[i]) * 0.3
            for rx, ry, rr in rocks:
                dx, dy = x[i] - rx, y[i] - ry
                d2 = dx * dx + dy * dy
                m = rr + RF
                if d2 < m * m and d2 > 0:
                    d = math.sqrt(d2)
                    nx, ny = dx / d, dy / d
                    x[i], y[i] = rx + nx * m, ry + ny * m
                    vn = vx[i] * nx + vy[i] * ny
                    if vn < 0:
                        vx[i] -= 1.2 * vn * nx
                        vy[i] -= 1.2 * vn * ny
                        if vn < -40:
                            hits.append((t, i, -vn))
            if fin[i] is None and y[i] >= LEN:
                fin[i] = t + DT * (LEN - yp) / max(1e-9, y[i] - yp)
                if first is None:
                    first = fin[i]
        for i in range(3):
            for j in range(i + 1, 4):
                dx, dy = x[j] - x[i], y[j] - y[i]
                d2 = dx * dx + dy * dy
                if d2 < 4 * RF * RF and d2 > 0:
                    d = math.sqrt(d2)
                    nx, ny = dx / d, dy / d
                    ov = (2 * RF - d) / 2
                    x[i] -= nx * ov
                    y[i] -= ny * ov
                    x[j] += nx * ov
                    y[j] += ny * ov
                    rv = (vx[j] - vx[i]) * nx + (vy[j] - vy[i]) * ny
                    if rv < 0:
                        imp = -0.75 * rv
                        vx[i] -= imp * nx
                        vy[i] -= imp * ny
                        vx[j] += imp * nx
                        vy[j] += imp * ny
                        if rv < -30:
                            bumps.append((t, i, j, -rv))
        if t_stop is None and all(f is not None for f in fin):
            t_stop = t + 0.7
        step += 1
    return dict(seed=seed, course=C, rec=rec, fin=fin, bumps=bumps, hits=hits)


def lead_changes(Y, nf):
    """Leader per frame with a 10 px hysteresis; returns (changes [(t, float)], leader per frame)."""
    lead, out, series = -1, [], []
    for f in range(nf):
        row = Y[f]
        j = int(np.argmax(row))
        if lead < 0:
            if f >= FPS and row[j] - np.partition(row, -2)[-2] >= 10:
                lead = j
        elif j != lead and row[j] - row[lead] >= 10:
            out.append((round(f / FPS, 2), j))
            lead = j
        series.append(lead)
    return out, series


def drama(sim):
    fin = sim["fin"]
    arr = np.array(sim["rec"][::SUB])
    Y = arr[:, 4:]
    done = all(f is not None for f in fin)
    ft = [f if f is not None else 99.0 for f in fin]
    order = sorted(range(4), key=lambda i: ft[i])
    w = order[0]
    T1 = ft[w]
    margin = ft[order[1]] - T1
    nf = min(len(Y), int(T1 * FPS) + 1)
    ch, series = lead_changes(Y, nf)
    f75 = next((f for f in range(nf) if Y[f].max() >= 0.75 * LEN), nf - 1)
    lead75 = series[f75] if series else -1
    traps = []
    for i in range(4):
        n = min(len(Y), int(ft[i] * FPS) + 1)
        yi = Y[:n, i]
        d = (yi[None, :] - yi[:, None]) < 110.0
        idx = np.arange(n)
        span = np.where(d, idx[None, :] - idx[:, None], 0).max() if n else 0
        traps.append(round(float(span) / FPS, 2))
    gates = dict(all_finish=done and ft[order[3]] - T1 <= 5.5,
                 length=17.5 <= T1 <= 23.6,
                 lead_changes=len(ch) >= 5,
                 winner_not_leading_at_75=lead75 != w and lead75 >= 0,
                 margin=0.04 <= margin < 0.4,
                 no_long_trap=max(traps) <= 6.0)
    ok = all(gates.values())
    score = 100.0 if ok else 10.0 * sum(gates.values())
    score += min(len(ch), 9) * 1.0
    score += 2.0 if sum(1 for t, _ in ch if t > 5.0) >= 3 else 0.0
    score += 3.0 if 0.06 <= margin <= 0.25 else 0.0
    score += 2.0 * min(sum(1 for t in traps if t >= 1.3), 3)
    score += 2.0 if ft[order[2]] - T1 < 1.2 else 0.0
    score -= 1.5 * max(0.0, ft[order[3]] - T1 - 3.0)
    score += 2.0 if 19.0 <= T1 <= 22.5 else 0.0
    return dict(
        seed=sim["seed"], score=round(score, 3), valid=ok, gates=gates,
        winner=NAMES[w], winner_index=w, order=[NAMES[i] for i in order],
        finish_times_s=[round(ft[i], 3) for i in order],
        winning_margin_s=round(margin, 3),
        lead_changes=len(ch), lead_change_times_s=[t for t, _ in ch],
        leader_at_75_percent=NAMES[lead75] if lead75 >= 0 else None,
        longest_stall_s={NAMES[i]: traps[i] for i in range(4)},
        lanes_left_to_right=[NAMES[i] for i in sim["course"]["lanes"]],
    )


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
COL = [hex_rgb("#FFC23D"), hex_rgb("#FF5A4E"), hex_rgb("#A6E84A"), hex_rgb("#E86BFF")]
INK = hex_rgb("#05080E")
CREAM = hex_rgb("#F1EEE6")
MUTED = hex_rgb("#8093A8")
MARK = (0.80, 0.84, 1.0)
TR_COL = [(0.17, 0.26, 0.52, 0.60), (0.16, 0.42, 0.70, 0.62), (0.20, 0.62, 0.82, 0.62), (0.45, 0.85, 0.93, 0.66),
          (0.86, 0.97, 1.00, 0.72), (0.15, 0.30, 0.50, 0.30)]
TR_EDGES = (170.0, 300.0, 420.0, 540.0)
NTR = 2800
STREAK = 0.075
VH = 2060.0
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
SLOT_X = [XL + (XR - XL) * (i + 0.5) / 4 for i in range(4)]
MX0, MW, MY0, MH = 32.0, 54.0, 412.0, 1086.0          # course map on the left bank


def map_xy(x, y):
    return MX0 + (x - XL) / (XR - XL) * MW, MY0 + (y + 150.0) / (LEN + 300.0) * MH


def float_path(ctx, k, s):
    """Float outline at the origin, pointing along +x."""
    if k == 0:                                           # ARROW: notched arrowhead
        pts = [(1.15, 0), (-0.85, 0.88), (-0.38, 0), (-0.85, -0.88)]
    elif k == 2:                                         # KITE: diamond
        pts = [(1.2, 0), (-0.05, 0.74), (-1.0, 0), (-0.05, -0.74)]
    elif k == 3:                                         # TWIN: double chevron
        for ox in (0.78, -0.12):
            c = [(0.37, 0), (-0.45, 0.9), (-0.9, 0.9), (-0.1, 0), (-0.9, -0.9), (-0.45, -0.9)]
            ctx.move_to((c[0][0] + ox) * s, c[0][1] * s)
            for px, py in c[1:]:
                ctx.line_to((px + ox) * s, py * s)
            ctx.close_path()
        return
    else:                                                # LEAF: lens
        ctx.move_to(1.2 * s, 0)
        ctx.curve_to(0.5 * s, 0.86 * s, -0.5 * s, 0.86 * s, -1.2 * s, 0)
        ctx.curve_to(-0.5 * s, -0.86 * s, 0.5 * s, -0.86 * s, 1.2 * s, 0)
        ctx.close_path()
        return
    ctx.move_to(pts[0][0] * s, pts[0][1] * s)
    for px, py in pts[1:]:
        ctx.line_to(px * s, py * s)
    ctx.close_path()


def draw_float(ctx, k, x, y, ang, s, alpha=1.0, halo=True, lw=4.0):
    if alpha <= 0.01:
        return
    if halo:
        glow(ctx, x, y, s * 2.6, COL[k], 0.30 * alpha)
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    float_path(ctx, k, s)
    ctx.set_source_rgba(*INK, alpha)
    ctx.set_line_width(lw * 2)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.stroke_preserve()
    ctx.set_source_rgba(*COL[k], alpha)
    ctx.fill()
    ctx.set_source_rgba(*INK, 0.85 * alpha)
    ctx.set_line_width(max(1.5, s * 0.1))
    if k == 1:                                           # leaf midrib
        ctx.move_to(-0.8 * s, 0)
        ctx.line_to(0.75 * s, 0)
        ctx.stroke()
    elif k == 2:                                         # kite cross spar
        ctx.move_to(-0.05 * s, -0.5 * s)
        ctx.line_to(-0.05 * s, 0.5 * s)
        ctx.stroke()
    ctx.restore()


def make_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    s = np.clip(np.abs(xx - XC) / HW, 0, 1)
    img = np.empty((H, W, 3), np.float32)
    img[...] = np.array([0.020, 0.036, 0.066], np.float32)
    img += ((1 - s ** 2) * 0.030)[..., None] * np.array([0.5, 0.9, 1.4], np.float32)
    bank = (xx < XL) | (xx > XR)
    img[bank] = np.array([0.028, 0.034, 0.046], np.float32)
    hatch = bank & (((xx + yy) % 26) < 2)
    img[hatch] += 0.016
    edge = (np.abs(xx - XL) < 2) | (np.abs(xx - XR) < 2)
    img[edge] = np.array([0.30, 0.40, 0.52], np.float32)
    img += np.random.default_rng(12).normal(0, 0.004, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    return surf, buf


def gauss(a, sigma):
    r = int(sigma * 3)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(a, r, mode="edge"), k / k.sum(), mode="valid")


def smoothstep(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((60, -9.0), (160, -5.0), (300, -2.0), (900, 0.0), (3000, 2.5), (8000, 1.5), (16000, -1.0))
    DRIVE = 2.6

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed)
        self.stats = drama(self.sim)
        fin = self.sim["fin"]
        if any(f is None for f in fin):
            raise ValueError(f"seed {seed}: not every float finishes")
        self.C = self.sim["course"]
        self.fin = fin
        self.order = sorted(range(4), key=lambda i: fin[i])
        self.winner = self.order[0]
        self.T1, self.T2 = fin[self.order[0]], fin[self.order[1]]
        self.margin = self.T2 - self.T1
        self.ta = self.T1 - 0.12
        self.sb = self.T2 + 0.06
        self.tb = self.ta + (self.sb - self.ta) / SLOW
        self.tc = self.tb + FREEZE
        self.sc = self.sb + FREEZE * CREEP
        self.dur = math.ceil((self.tc + REW + PRE) * FPS - 1e-6) / FPS
        self.td = self.dur - PRE
        self.nframes = int(round(self.dur * FPS))
        self._res = None
        self._prep()

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    # ------------------------------------------------------------ timeline
    def sim_time(self, t):
        if t < self.ta:
            return t
        if t < self.tb:
            return self.ta + (t - self.ta) * SLOW
        if t < self.tc:
            return self.sb + (t - self.tb) * CREEP
        if t < self.td:
            return lerp(self.sc, -PRE, smoothstep((t - self.tc) / (self.td - self.tc)))
        return t - self.dur

    def disp_time(self, s):
        """Display time of a simulation moment during the race and the slow motion."""
        return s if s < self.ta else self.ta + (s - self.ta) / SLOW

    def _prep(self):
        rec = np.array(self.sim["rec"])
        n = len(rec)
        self.n = n
        self.PX, self.PY = rec[:, :4], rec[:, 4:]
        ts = np.arange(n) * DT
        w = self.winner
        box = np.ones(19) / 19
        ang = np.empty((n, 4))
        for i in range(4):
            vx = np.convolve(np.pad(np.gradient(self.PX[:, i], DT), 9, mode="edge"), box, mode="valid")
            vy = np.convolve(np.pad(np.gradient(self.PY[:, i], DT), 9, mode="edge"), box, mode="valid") + 12.0
            ang[:, i] = np.unwrap(np.arctan2(vy, vx))
        camL = gauss(np.minimum(self.PY.max(axis=1), LEN + 70.0), 40.0) - ANCHOR
        camW = gauss(np.minimum(self.PY[:, w], LEN + 70.0), 40.0) - ANCHOR
        camW = camW + (camL[0] - camW[0]) * (1 - np.clip(ts / 2.0, 0, 1))
        F = self.nframes
        tt = np.arange(F) / FPS
        self.S = np.array([self.sim_time(t) for t in tt])
        idx = np.clip(self.S / DT, 0, n - 1)
        grid = np.arange(n)
        self.FX = np.stack([np.interp(idx, grid, self.PX[:, i]) for i in range(4)], axis=1)
        self.FY = np.stack([np.interp(idx, grid, self.PY[:, i]) for i in range(4)], axis=1)
        self.FA = np.stack([np.interp(idx, grid, ang[:, i]) for i in range(4)], axis=1)
        cl, cw = np.interp(idx, grid, camL), np.interp(idx, grid, camW)
        kd = np.clip((tt - self.tc) / 0.4, 0, 1)
        self.CAM = cl * (1 - kd) + cw * kd
        # whirlpool mark phase: follows simulation time, but never spins fast enough to flicker
        ds = np.diff(self.S, append=self.S[-1] + 1 / FPS)
        self.MPH = np.cumsum(np.clip(ds, -0.05, 0.05))
        # rank per frame (final order once the race is decided), and the eased chip positions
        lanes = self.C["lanes"]
        lane_of = {i: k for k, i in enumerate(lanes)}
        final_rank = {i: r for r, i in enumerate(self.order)}
        self.RANK = np.zeros((F, 4), int)
        for f in range(F):
            t = tt[f]
            if t >= self.td - 0.25:
                r = lane_of
            elif t >= self.tb:
                r = final_rank
            else:
                o = sorted(range(4), key=lambda i: (-round(self.FY[f, i], 1), lane_of[i]))
                r = {i: q for q, i in enumerate(o)}
            for i in range(4):
                self.RANK[f, i] = r[i]
        self.CHIP = np.zeros((F, 4))
        cur = np.array([SLOT_X[lane_of[i]] for i in range(4)])
        for f in range(F):
            cur = cur + (np.array([SLOT_X[q] for q in self.RANK[f]]) - cur) * 0.3
            self.CHIP[f] = cur
        # overtakes during the race (sound): a float whose rank improved, debounced per float
        self.passes = []
        last = [-9.0] * 4
        fa = int(self.ta * FPS)
        for f in range(FPS, min(fa, F)):
            for i in range(4):
                if self.RANK[f, i] < self.RANK[f - 1, i] and tt[f] - last[i] > 0.4:
                    self.passes.append((tt[f], i, int(self.RANK[f, i])))
                    last[i] = tt[f]
        # whirlpool captures: a float inside 1.25 radii for at least half a second
        self.caught = np.full((n, 4), -1, int)
        self.captures = []
        for vi, v in enumerate(self.C["vort"]):
            cx = v[0] + v[4] * np.sin(v[5] * ts + v[6])
            cy = v[1] + v[7] * np.sin(v[8] * ts + v[9])
            for i in range(4):
                ins = np.hypot(self.PX[:, i] - cx, self.PY[:, i] - cy) < 1.25 * v[2]
                ins[int(self.fin[i] / DT):] = False
                edges = np.flatnonzero(np.diff(np.concatenate(([0], ins.astype(int), [0]))))
                for a, b in zip(edges[::2], edges[1::2]):
                    if b - a >= 60:
                        self.caught[a + 36:b, i] = vi
                        self.captures.append((ts[a] + 0.3, i, vi, ts[b - 1]))
        self.captures.sort()
        # leader's distance markers every 500 px (sound), and the winner's path (rewind)
        lead = self.PY.max(axis=1)
        self.marks = [float(ts[np.argmax(lead >= d)]) for d in range(500, int(LEN), 500)]
        k1 = min(n - 1, int(self.T1 / DT) + 8)
        self.path = np.stack([self.PX[:k1:3, w], self.PY[:k1:3, w]], axis=1)
        rnd = random.Random(self.seed * 17 + 3)
        self.rock_shape = [[(rr * rnd.uniform(0.86, 1.04), TAU * (j + rnd.uniform(-0.25, 0.25)) / 9) for j in range(9)]
                           for _, _, rr in self.C["rocks"]]
        self._tracers()

    def _tracers(self):
        C = self.C
        rng = np.random.default_rng(self.seed * 13 + 5)
        F, N = self.nframes, NTR
        x = rng.uniform(XL + 2, XR - 2, N)
        y = self.CAM[0] - 70 + rng.uniform(0, VH, N)
        age = rng.integers(3, 60, N)
        life = rng.integers(10, 130, N)
        self.TR = np.empty((F, N, 4), np.float32)
        self.TB = np.empty((F, N), np.uint8)
        for f in range(F):
            s, cam = self.S[f], self.CAM[f]
            ux, uy = flow_np(x, y, s, C)
            sp = np.hypot(ux, uy)
            o = self.TR[f]
            o[:, 0] = x
            o[:, 1] = y - cam
            o[:, 2] = x - ux * STREAK
            o[:, 3] = y - cam - uy * STREAK
            b = np.digitize(sp, TR_EDGES).astype(np.uint8)
            b[(age < 3) | (life < 3)] = 5
            b[sp < 1.0] = 6
            self.TB[f] = b
            s2 = self.S[f + 1] if f + 1 < F else s + 1 / FPS
            cam2 = self.CAM[f + 1] if f + 1 < F else cam
            dt = s2 - s
            m = max(1, int(math.ceil(abs(dt) * FPS - 1e-6)))
            h = dt / m
            for j in range(m):
                ss = s + j * h
                if j:
                    ux, uy = flow_np(x, y, ss, C)
                ux2, uy2 = flow_np(x + ux * h * 0.5, y + uy * h * 0.5, ss + h * 0.5, C)
                x = x + ux2 * h
                y = y + uy2 * h
            age += 1
            life -= 1
            sy = y - cam2
            up, dn = sy < -70, sy > VH - 70
            y[up] += VH
            y[dn] -= VH
            dead = (life <= 0) | (x < XL + 2) | (x > XR - 2) | (sp < 1.0)
            y[dead] = cam2 - 70 + rng.uniform(0, VH, int(dead.sum()))
            re = dead | up | dn
            k = int(re.sum())
            x[re] = rng.uniform(XL + 2, XR - 2, k)
            age[re] = 0
            life[re] = rng.integers(45, 130, k)

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            bg, keep = make_background()
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep, font=Font(FONT_PATH))
        return self._res

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        self.draw(ctx, f, res)
        ctx.restore()
        res["surf"].flush()
        return bytes(res["surf"].get_data())

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, res):
        f = max(0, min(f, self.nframes - 1))
        t = f / FPS
        font = res["font"]
        s, cam = float(self.S[f]), float(self.CAM[f])
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        self._marks(ctx, cam, font)
        ctx.save()
        ctx.rectangle(XL + 2, 0, XR - XL - 4, H)
        ctx.clip()
        self._streaks(ctx, f)
        self._whirlpools(ctx, f, s, cam, font)
        self._rocks(ctx, cam)
        self._lines(ctx, t, cam, font)
        self._path(ctx, t, cam)
        self._floats(ctx, f, t, s, cam)
        ctx.restore()
        self._photo(ctx, f, t, cam, font)
        self._map(ctx, f, s, font)
        self._strip(ctx, f, t, font)
        self._offscreen(ctx, f, cam)
        self._headline(ctx, t, font)
        self._hook(ctx, t, font)

    def _marks(self, ctx, cam, font):
        d0 = int(math.floor((cam - 40) / 500.0)) * 500
        for d in range(d0, int(cam + H + 500), 500):
            if d < 0 or d > LEN:
                continue
            y = d - cam
            ctx.move_to(XL, y)
            ctx.line_to(XR, y)
            ctx.set_source_rgba(0.5, 0.65, 0.85, 0.10)
            ctx.set_line_width(2)
            ctx.stroke()
            if d % 1000 == 0 and 0 < d < LEN:
                font.draw(ctx, str(d), XR + 14, y + 9, 24, darken(MUTED, 0.25), tracking=1)

    def _streaks(self, ctx, f):
        tr, tb = self.TR[f], self.TB[f]
        ctx.set_line_width(2.4)
        for b in range(6):
            seg = tr[tb == b]
            if not len(seg):
                continue
            for x0, y0, x1, y1 in seg.tolist():
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
            ctx.set_source_rgba(*TR_COL[b])
            ctx.stroke()

    def _whirlpools(self, ctx, f, s, cam, font):
        ph = float(self.MPH[f])
        idx = clamp(s / DT, 0, self.n - 1)
        cg = self.caught[int(idx)] if f / FPS < self.tb else (-1, -1, -1, -1)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for vi, v in enumerate(self.C["vort"]):
            cx, cy = vpos(v, s)
            y = cy - cam
            a = v[2]
            if y < -3 * a or y > H + 3 * a:
                continue
            g = cairo.RadialGradient(cx, y, 0, cx, y, 1.5 * a)
            g.add_color_stop_rgba(0, 0, 0, 0.02, 0.62)
            g.add_color_stop_rgba(0.55, 0, 0, 0.02, 0.30)
            g.add_color_stop_rgba(1, 0, 0, 0.02, 0)
            ctx.set_source(g)
            ctx.arc(cx, y, 1.5 * a, 0, TAU)
            ctx.fill()
            sg = 1.0 if v[3] > 0 else -1.0
            st = abs(v[3]) / 620.0
            for rk, q in ((0.5, 0.25), (0.95, 0.9), (1.4, 1.96)):
                w = abs(v[3]) / a * math.exp(0.5 * (1 - q))
                a0 = sg * w * ph + vi
                for arm in range(3):
                    aa = a0 + arm * TAU / 3
                    if sg > 0:
                        ctx.arc(cx, y, rk * a, aa, aa + 1.25)
                    else:
                        ctx.arc_negative(cx, y, rk * a, aa, aa - 1.25)
                    ctx.new_sub_path()
                ctx.set_source_rgba(*MARK, 0.30 + 0.35 * st)
                ctx.set_line_width(2.0 + 3.0 * st)
                ctx.stroke()
            ctx.arc(cx, y, 6, 0, TAU)
            ctx.set_source_rgba(*MARK, 0.8)
            ctx.fill()
            for i in range(4):
                if cg[i] == vi:
                    ctx.save()
                    ctx.set_dash([18, 12], -ph * 60)
                    ctx.arc(cx, y, 1.75 * a, 0, TAU)
                    ctx.set_source_rgba(*COL[i], 0.9)
                    ctx.set_line_width(5)
                    ctx.stroke()
                    ctx.restore()
                    font.draw(ctx, "CAUGHT", clamp(cx, XL + 70, XR - 70), y - 1.75 * a - 14, 30, COL[i], anchor="c",
                              stroke=INK, stroke_w=7, tracking=1.5)
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)

    def _rocks(self, ctx, cam):
        for (rx, ry, rr), shape in zip(self.C["rocks"], self.rock_shape):
            y = ry - cam
            if y < -120 or y > H + 120:
                continue
            pts = [(rx + r * math.cos(a), y + r * math.sin(a)) for r, a in shape]
            ctx.move_to(*pts[0])
            for p in pts[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.set_source_rgba(0.115, 0.135, 0.165, 1)
            ctx.fill_preserve()
            ctx.set_source_rgba(0.62, 0.69, 0.78, 1)
            ctx.set_line_width(3.5)
            ctx.stroke()
            for k in (-0.35, 0.0, 0.35):
                ctx.move_to(rx - rr * 0.45, y + rr * (k + 0.2))
                ctx.line_to(rx + rr * 0.45, y + rr * (k - 0.2))
            ctx.set_source_rgba(0.62, 0.69, 0.78, 0.35)
            ctx.set_line_width(2.5)
            ctx.stroke()

    def _lines(self, ctx, t, cam, font):
        y = -40.0 - cam                                    # start line, just behind the floats
        if -20 < y < H + 20:
            ctx.move_to(XL, y)
            ctx.line_to(XR, y)
            ctx.set_source_rgba(*CREAM, 0.55)
            ctx.set_line_width(4)
            ctx.stroke()
            font.draw(ctx, "START", XL + 18, y - 12, 28, CREAM, alpha=0.8, stroke=INK, stroke_w=6, tracking=2)
        y = LEN - cam
        if -20 < y < H + 60:
            hot = ramp(t, self.ta - 0.6, self.ta) * (1 - ramp(t, self.tc, self.tc + 0.5))
            if hot > 0:
                g = cairo.LinearGradient(0, y - 46, 0, y + 46)
                g.add_color_stop_rgba(0, 1, 1, 1, 0)
                g.add_color_stop_rgba(0.5, 1, 1, 1, 0.22 * hot)
                g.add_color_stop_rgba(1, 1, 1, 1, 0)
                ctx.rectangle(XL, y - 46, XR - XL, 92)
                ctx.set_source(g)
                ctx.fill()
            ctx.move_to(XL, y)
            ctx.line_to(XR, y)
            ctx.set_source_rgba(*INK, 0.9)
            ctx.set_line_width(12)
            ctx.stroke()
            ctx.save()
            ctx.set_dash([26, 14])
            ctx.move_to(XL, y)
            ctx.line_to(XR, y)
            ctx.set_source_rgba(*CREAM, 1)
            ctx.set_line_width(6)
            ctx.stroke()
            ctx.restore()
            font.draw(ctx, "FINISH", XL + 18, y + 46, 34, CREAM, stroke=INK, stroke_w=7, tracking=2)

    def _path(self, ctx, t, cam):
        al = ramp(t, self.tc - 0.25, self.tc + 0.15) * (1 - ramp(t, self.td - 0.35, self.dur - 0.12))
        if al <= 0:
            return
        p = self.path
        ys = p[:, 1] - cam
        vis = (ys > -80) & (ys < H + 80)
        if not vis.any():
            return
        pen = False
        for (x, _), y, v in zip(p.tolist(), ys.tolist(), vis.tolist()):
            if not v:
                pen = False
            elif pen:
                ctx.line_to(x, y)
            else:
                ctx.move_to(x, y)
                pen = True
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        c = COL[self.winner]
        for lw, a, col in ((30, 0.16, c), (16, 0.38, c), (7, 1.0, lighten(c, 0.55))):
            ctx.set_source_rgba(*col, a * al)
            ctx.set_line_width(lw)
            ctx.stroke_preserve()
        ctx.new_path()
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)

    def _floats(self, ctx, f, t, s, cam):
        rew = ramp(t, self.tc, self.tc + 0.3) * (1 - ramp(t, self.td - 0.3, self.td + 0.1))
        if t < self.tc and s > 0.05:                       # short trails
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            i1 = int(clamp(s / DT, 0, self.n - 1))
            for i in range(4):
                for back, a, lw in ((84, 0.22, 5), (48, 0.36, 7), (20, 0.5, 9)):
                    i0 = max(0, i1 - back)
                    if i1 - i0 < 2:
                        continue
                    xs = self.PX[i0:i1 + 1:4, i].tolist()
                    ys = (self.PY[i0:i1 + 1:4, i] - cam).tolist()
                    ctx.move_to(xs[0], ys[0])
                    for x, y in zip(xs[1:], ys[1:]):
                        ctx.line_to(x, y)
                    ctx.set_source_rgba(*COL[i], a)
                    ctx.set_line_width(lw)
                    ctx.stroke()
            ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        for i in sorted(range(4), key=lambda q: q == self.winner):
            y = self.FY[f, i] - cam
            if y < -80 or y > H + 80:
                continue
            al = 1.0 if i == self.winner else 1 - 0.6 * rew
            draw_float(ctx, i, self.FX[f, i], y, self.FA[f, i], 31.0, alpha=al)

    def _photo(self, ctx, f, t, cam, font):
        """Freeze: the gap between first and second, measured at the line in hundredths of a second."""
        if t >= self.tb and t - self.tb < 0.35:            # one soft flash
            ctx.rectangle(0, 0, W, H)
            ctx.set_source_rgba(1, 1, 1, 0.14 * (1 - (t - self.tb) / 0.35))
            ctx.fill()
        al = ramp(t, self.tb + 0.2, self.tb + 0.45) * (1 - ramp(t, self.tc, self.tc + 0.3))
        if al <= 0:
            return
        a, b = self.order[0], self.order[1]
        xa, ya = self.FX[f, a], self.FY[f, a] - cam
        xb, yb = self.FX[f, b], self.FY[f, b] - cam
        right = max(xa, xb) < 600
        xd = max(xa, xb) + 78 if right else min(xa, xb) - 78
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for col, lw in ((INK, 10), (CREAM, 4)):
            for x0, y0 in ((xa, ya), (xb, yb)):
                ctx.move_to(x0 + (44 if right else -44), y0)
                ctx.line_to(xd, y0)
            ctx.move_to(xd, ya)
            ctx.line_to(xd, yb)
            ctx.set_source_rgba(*col, al)
            ctx.set_line_width(lw)
            ctx.stroke()
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        txt = f"{self.margin:.2f} S"
        ym = max(ya, yb) + 96                              # below both floats, clear of the finish line
        font.draw(ctx, txt, xd + (18 if right else -18), ym, 66, CREAM, anchor="l" if right else "r", alpha=al,
                  stroke=INK, stroke_w=10)

    def _map(self, ctx, f, s, font):
        rounded_rect(ctx, MX0 - 12, MY0 - 14, MW + 24, MH + 28, 14)
        ctx.set_source_rgba(0.012, 0.020, 0.034, 0.92)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.30, 0.40, 0.52, 0.9)
        ctx.set_line_width(2)
        ctx.stroke()
        for yy_, a in ((0.0, 0.5), (LEN, 1.0)):
            _, y = map_xy(XL, yy_)
            ctx.move_to(MX0 - 6, y)
            ctx.line_to(MX0 + MW + 6, y)
            ctx.set_source_rgba(*CREAM, a)
            ctx.set_line_width(3)
            ctx.stroke()
        for v in self.C["vort"]:
            x, y = map_xy(*vpos(v, s))
            ctx.arc(x, y, max(4.0, v[2] * 0.075), 0, TAU)
            ctx.set_source_rgba(*MARK, 0.75)
            ctx.set_line_width(2)
            ctx.stroke()
        for rx, ry, rr in self.C["rocks"]:
            x, y = map_xy(rx, ry)
            ctx.rectangle(x - 4.5, y - 4.5, 9, 9)
            ctx.set_source_rgba(0.62, 0.69, 0.78, 1)
            ctx.fill()
        for i in sorted(range(4), key=lambda q: self.FY[f, q]):
            x, y = map_xy(self.FX[f, i], min(self.FY[f, i], LEN + 110))
            draw_float(ctx, i, x, y, math.pi / 2, 10.5, halo=False, lw=2.0)

    def _chip(self, ctx, font, i, cx, cy, alpha=1.0):
        rounded_rect(ctx, cx - 90, cy - 30, 180, 60, 14)
        ctx.set_source_rgba(0.02, 0.035, 0.06, 0.94 * alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(*COL[i], alpha)
        ctx.set_line_width(4)
        ctx.stroke()
        draw_float(ctx, i, cx - 58, cy, 0.0, 15.0, alpha=alpha, halo=False, lw=2.0)
        font.draw(ctx, NAMES[i], cx - 30, cy + 13, 34, COL[i], alpha=alpha, tracking=1.0)

    def _strip(self, ctx, f, t, font):
        gaps = ramp(t, self.tb + 0.25, self.tb + 0.5) * (1 - ramp(t, self.td - 1.0, self.td - 0.6))
        rounded_rect(ctx, XL + 6, 228, XR - XL - 12, 112 + 40 * gaps, 16)
        ctx.set_source_rgba(0.012, 0.020, 0.034, 0.80)
        ctx.fill()
        for q, lab in enumerate(("1ST", "2ND", "3RD", "4TH")):
            font.draw(ctx, lab, SLOT_X[q], 258, 24, MUTED, anchor="c", tracking=2)
        for i in sorted(range(4), key=lambda q: -self.RANK[f, q]):
            self._chip(ctx, font, i, float(self.CHIP[f, i]), 298)
        if gaps > 0:
            for q, i in enumerate(self.order):
                k = gaps * ramp(t, self.tb + 0.25 + 0.18 * q, self.tb + 0.4 + 0.18 * q)
                txt = "WINNER" if q == 0 else f"+{self.fin[i] - self.T1:.2f} S"
                font.draw(ctx, txt, SLOT_X[q], 366, 32, COL[i] if q == 0 else CREAM, anchor="c", alpha=k,
                          tracking=1.0)

    def _offscreen(self, ctx, f, cam):
        for i in range(4):
            y = self.FY[f, i] - cam
            if y < 392:
                x = float(self.FX[f, i])
                k = ramp(392 - y, 0, 40)
                ctx.move_to(x, 400)
                ctx.line_to(x - 20, 428)
                ctx.line_to(x + 20, 428)
                ctx.close_path()
                ctx.set_source_rgba(*INK, k)
                ctx.set_line_width(7)
                ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                ctx.stroke_preserve()
                ctx.set_source_rgba(*COL[i], k)
                ctx.fill()
                draw_float(ctx, i, x, 456, -math.pi / 2, 15.0, alpha=k, halo=False, lw=2.0)

    def _headline(self, ctx, t, font):
        k1 = ramp(t, self.ta, self.ta + 0.2) * (1 - ramp(t, self.tb + 0.45, self.tb + 0.6))
        if k1 > 0:
            font.draw(ctx, "PHOTO FINISH", XC, 560, 96, CREAM, anchor="c", alpha=k1, stroke=INK, stroke_w=12, shadow=5)
            font.draw(ctx, "SLOW MOTION", XC, 612, 34, MUTED, anchor="c", alpha=k1, stroke=INK, stroke_w=7,
                      tracking=3)
        out = 1 - ramp(t, self.td - 1.0, self.td - 0.6)
        k2 = ramp(t, self.tb + 0.6, self.tb + 0.85) * out
        if k2 > 0:
            w = self.winner
            txt = f"{NAMES[w]} WINS BY {self.margin:.2f} S"
            size = min(88.0, 88.0 * 760.0 / font.width(txt, 88))
            sc = 0.85 + 0.15 * ease_out(ramp(t, self.tb + 0.6, self.tb + 0.9))
            ctx.save()
            ctx.translate(XC, 560)
            ctx.scale(sc, sc)
            font.draw(ctx, txt, 0, 0, size, COL[w], anchor="c", alpha=k2, stroke=INK, stroke_w=12, shadow=5)
            ctx.restore()
        k3 = ramp(t, self.tc + 0.1, self.tc + 0.4) * out
        if k3 > 0:
            font.draw(ctx, "REWIND: THE WINNING PATH", XC, 618, 40, CREAM, anchor="c", alpha=k3, stroke=INK,
                      stroke_w=8, tracking=1.5)

    def _hook(self, ctx, t, font):
        k = max(1 - ramp(t, 2.4, 2.8), ramp(t, self.dur - 0.75, self.dur - 0.3))      # question block
        m = max(1 - ease_in_out(ramp(t, 2.4, 3.1)), ease_in_out(ramp(t, self.dur - 0.95, self.dur - 0.3)))
        if k > 0:
            rounded_rect(ctx, XL + 26, 452, XR - XL - 52, 430, 26)
            ctx.set_source_rgba(0.012, 0.020, 0.034, 0.78 * k)
            ctx.fill()
            for txt, y in (("WHICH FLOAT", 584), ("WINS THE RIVER?", 712)):
                size = min(122.0, 122.0 * 700.0 / font.width(txt, 122))
                font.draw(ctx, txt, XC, y, size, CREAM, anchor="c", alpha=k, stroke=INK, stroke_w=12, shadow=6)
            font.draw(ctx, "ONE CURRENT  •  10 WHIRLPOOLS  •  3 ROCKS", XC, 860, 31, MUTED, anchor="c", alpha=k,
                      tracking=1.0)
        # the rule pill: large under the question, then it parks above the rank strip
        txt = "NO STEERING. FIRST ACROSS WINS"
        size = lerp(34.0, 47.0, m)
        cy = lerp(192.0, 790.0, m)
        tw = font.width(txt, size, 1.2)
        hh = size * 1.5
        rounded_rect(ctx, XC - tw / 2 - 24, cy - hh / 2, tw + 48, hh, 12)
        ctx.set_source_rgba(*CREAM, 1)
        ctx.fill()
        font.draw(ctx, txt, XC, cy + size * 0.37, size, INK, anchor="c", tracking=1.2)

    # ------------------------------------------------------------ sound
    def audio(self):
        rng = np.random.default_rng(self.seed + 912)
        sr = au.SR
        n, tail = int(round(self.dur * sr)), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        F = self.nframes
        ft = np.arange(F) / FPS
        note = [au.hz(m) for m in (69, 72, 76, 79)]        # one pitch per float

        def tt(d):
            return np.arange(int(d * sr)) / sr

        def pan_of(i, s):
            j = int(clamp(s / DT, 0, self.n - 1))
            return clamp((self.PX[j, i] - XC) / HW * 0.7, -0.7, 0.7)

        def drop(f0, dur=0.15):                            # water-drop blip: the pitch flicks upward
            t_ = tt(dur)
            fr = f0 * (0.72 + 0.5 * np.minimum(1.0, t_ / 0.05))
            ph = TAU * np.cumsum(fr) / sr
            return (np.sin(ph) + 0.3 * np.sin(2 * ph)) * np.minimum(1.0, t_ / 0.004) * np.exp(-t_ / 0.055)

        # overtakes: two-note blip in the overtaker's pitch (louder when it takes the lead)
        for t_, i, r in self.passes:
            g = 0.34 if r == 0 else 0.17
            p = pan_of(i, t_)
            au.place(fx, drop(note[i]), t_, g, p)
            au.place(fx, drop(note[i] * (2.0 if r == 0 else 1.5)), t_ + 0.085, g, p)
        # distance markers: a plucked scale that climbs toward the line, and plays back down in the rewind
        scale = (0, 2, 4, 7, 9)
        d_t = ft[(ft >= self.tc) & (ft < self.td)]
        d_s = self.S[(ft >= self.tc) & (ft < self.td)]
        for k, sm in enumerate(self.marks):
            f0 = au.hz(57 + 12 * (k // 5) + scale[k % 5])
            if sm < self.ta:
                au.place(fx, au.pluck(f0, 0.7), sm, 0.13, -0.15)
                au.place(fx, au.tick(1300, 0.04), sm, 0.035, 0.15)
            tr = float(np.interp(sm, d_s[::-1], d_t[::-1]))
            au.place(fx, au.pluck(f0, 0.35), tr, 0.12, 0.2)
        # whirlpool capture: wobbling descending tone; release: a short upward chirp
        for s0, i, vi, s1 in self.captures:
            if s0 >= self.ta:
                continue
            t_ = tt(0.95)
            fr = note[i] * 0.75 * (0.5 ** (t_ / 0.95)) * (1 + 0.07 * np.sin(TAU * 8.5 * t_))
            ph = TAU * np.cumsum(fr) / sr
            x = (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)) * np.minimum(1, t_ / 0.02) * \
                np.minimum(1.0, (0.95 - t_) / 0.25) * (0.75 + 0.25 * np.sin(TAU * 8.5 * t_))
            au.place(fx, x, s0, 0.20, pan_of(i, s0))
            if s1 < self.ta:
                au.place(fx, au.glide(note[i] * 0.5, note[i] * 1.0, 0.16, 0.08), s1, 0.16, pan_of(i, s1))
        # bumps
        last = {}
        for t_, i, j, v in self.sim["bumps"]:
            if t_ < self.ta and t_ - last.get((i, j), -9) > 0.3:
                last[(i, j)] = t_
                au.place(fx, au.tick(820, 0.05), t_, 0.07 + 0.10 * min(1.0, v / 250), pan_of(i, t_))
        lasth = {}
        for t_, i, v in self.sim["hits"]:
            if t_ < self.ta and t_ - lasth.get(i, -9) > 0.4:
                lasth[i] = t_
                au.place(fx, au.glide(190, 95, 0.14, 0.05) + 0.5 * au.noise(rng, 0.14, 300, 1800) * np.exp(-tt(0.14) * 40),
                         t_, 0.14 + 0.14 * min(1.0, v / 300), pan_of(i, t_))
        # photo finish: the water drops away, one bell per float across the line, a shutter, then a short chord
        au.place(fx, au.swept_noise(rng, 0.7, 2400, 260) * np.sin(np.pi * tt(0.7) / 0.7) ** 0.5, self.ta - 0.1, 0.22)
        for q, i in enumerate(self.order):
            if self.fin[i] <= self.sb:
                au.place(fx, au.bell(note[i] * 2, 1.3), self.disp_time(self.fin[i]), 0.30 - 0.08 * q,
                         pan_of(i, self.fin[i]))
        for d in (0.0, 0.07):
            au.place(fx, au.clap(rng, 0.08), self.tb + d, 0.34)
        for q in range(1, 4):
            au.place(fx, au.tick(1750 - 180 * q, 0.06), self.tb + 0.3 + 0.18 * q, 0.14)
        wn = note[self.winner]
        for r, g, d in ((1.0, 0.16, 0.0), (1.26, 0.12, 0.03), (1.5, 0.12, 0.06), (2.0, 0.10, 0.09)):
            au.place(fx, au.bell(wn * r, 1.5), self.tb + 0.62 + d, g)
        # rewind: a rising whoosh, then one soft bell on the start line
        rw = self.td - self.tc
        au.place(fx, au.swept_noise(rng, rw, 350, 2600) * np.sin(np.pi * tt(rw) / rw)[:int(rw * sr)], self.tc, 0.17)
        au.place(fx, au.bell(au.hz(69), 1.2), self.td - 0.05, 0.12)
        # water bed: band-limited noise that gurgles; brighter and louder with the leader's speed
        lead = np.argmax(self.FY, axis=1)
        sp = np.abs(np.gradient(self.FY[np.arange(F), lead], 1 / FPS)) / 520.0
        sp = np.clip(gauss(np.clip(sp, 0, 3), 6.0), 0.05, 1.3)
        ts = np.arange(n) / sr
        spd = np.interp(ts, ft, sp)
        lfo = np.interp(ts, np.arange(0, self.dur + 1, 0.11), np.abs(rng.standard_normal(int((self.dur + 1) / 0.11) + 1)))
        lfo = 0.25 + 0.75 * np.clip(lfo / 1.6, 0, 1)
        lo = au.noise(rng, n / sr + 0.01, 170, 650)[:n]
        hi = au.noise(rng, n / sr + 0.01, 650, 2300)[:n]
        lo /= np.abs(lo).std() * 3
        hi /= np.abs(hi).std() * 3
        water = (lo * (0.55 + 0.25 * spd) + hi * 0.75 * spd ** 1.5) * lfo * 0.042
        bed[:n, 0] += water
        bed[:n, 1] += np.roll(water, 331)
        del lo, hi, water, lfo, spd, ts
        au.place(bed, au.pad(au.hz(50), self.dur, 1.0, 0.8), 0.0, 0.016)
        au.place(bed, au.pad(au.hz(57), self.dur, 1.0, 0.8), 0.0, 0.010)
        mix_ = au.reverb(fx, rng, decay=0.9, wet=0.10) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:n + tail][:n]
        out = au.eq(out, self.EQ)
        return np.tanh(out * (self.DRIVE / max(1e-9, np.abs(out).max())))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        title = "No Steering, Ten Whirlpools… Which Float Wins? 🌊"
        assert len(title) <= 60
        lanes = ", ".join(NAMES[i].title() for i in self.C["lanes"])
        return dict(
            title=title,
            description=(
                "Which float wins the river: Arrow, Leaf, Kite or Twin? Call it before they leave the start line!\n\n"
                "Four identical floats start side by side at the top of a river and nobody steers them. The water is "
                "one velocity field: a steady current that runs faster mid-river and slower along the banks with a "
                "gentle meander, ten whirlpools that drift slowly and pull inward as they spin, and three rocks the "
                "flow has to split around. The thousands of streaks are tracers drawn from that same field, so the "
                "current you see is exactly the current the floats feel. Each float follows the local flow with a "
                "little inertia and can bump the others, the rocks and the banks. A float caught by a whirlpool "
                "circles until it is flung back out. The map on the left shows the whole course from the first "
                "frame, and the strip at the top keeps the running order. The first float across the finish line "
                "wins, and the freeze at the line measures the gaps in hundredths of a second. Then the race "
                "rewinds along the winner's path.\n"
                f"Start lanes, left to right: {lanes}. The lanes rotate between episodes.\n"
                f"Flow, picture and sound are generated by code (seed {self.seed}), with no footage, samples or "
                "stock media.\n"
                "Which float did you pick?\n"
                "#simulation #satisfying #flowfield"),
            tags=["flow field", "river race", "whirlpool", "fluid simulation", "vector field", "prediction",
                  "satisfying", "made with code", "simulation"],
            pinned_comment=("Arrow, Leaf, Kite or Twin: which float wins the river? Pick before they leave the start "
                            "line. Nobody steers, the current and ten drifting whirlpools decide."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of abstract shapes; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=self.stats,
        )
