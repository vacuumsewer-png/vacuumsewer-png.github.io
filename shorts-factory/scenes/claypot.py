"""CLAY POT (CRUSH family, episode 3): a steel wrecking ball swings into one big terracotta flower pot, again and
again, each swing a little higher, then one huge smash. Into how many pieces?

Changes from episodes 1-2 (icecrush: press, chocolatecrush: dropping weight): a new impactor (a swinging wrecking
ball, side impact), a new object (a fired-clay pot with a pot-shaped outline) and a new crack rule (curved shell
fractures that ring the impact point, instead of straight cuts).

Load: the ball hits at times t_k with peak load P_k = fs * p_k, where p_k is the swing height of hit k relative to the
highest regular swing (p_k grows from P0 to 1 and the swings come faster); the last hit falls from SLAM_X times the
highest regular swing (the smash, load SLAM_X * fs). A swing's energy is weight x height, so its load is proportional
to its height. A piece feels the load weakened by its distance from the impact point: w = exp(-DECAY * dist).
Each piece keeps the highest load so far (brittle clay does not heal): it breaks when that load * w passes its strength.

Fracture (same cascade as engine/crush.py): strength = max(1.10 * parent, parent * sqrt(parent area / area) * flaw),
so smaller shards are stronger and each new crack needs a harder hit. A break splits a piece in two along a circular
arc: either a ring around the impact point (a shell fracture) or a gentle arc pointing back at the impact (a radial
crack). The count is the real simulation result; seeds are only filtered by the drama gates.

On screen: one question line and one counter. The answer flashes once after the smash cascade, the why line follows,
and the ball swings back into its frame-0 position for the loop.
"""
import math
import random

import cairo
import numpy as np

from engine import audio as au
from engine import crush
from engine.crush import CrushScene, poly_ca
from engine.gfx import TAU, clamp, ease_out, glow, rounded_rect

ASP = 540.0 / 430.0          # HC / WC: unit v is this much taller than unit u on screen


# ====================================================================== swing schedule (class-level, not seed dependent)
def hits(C):
    """Times and load fractions of the regular hits (the smash is extra, at C.T_SLAM)."""
    n = C.N_HITS
    tk, pk = [C.T_CONTACT], []
    for k in range(n - 1):
        tk.append(tk[-1] + C.D0 + (C.D1 - C.D0) * k / (n - 2))
    for k in range(n):
        pk.append(C.P0 + (1 - C.P0) * (k / (n - 1)) ** C.PEXP)
    return tk, pk


def load_time(s, C, fs, fslam):
    """First time the running-max load reaches s (inf if never)."""
    tk, pk = hits(C)
    if s <= 0:
        return tk[0]
    prev = 0.0
    for t, p in zip(tk, pk):
        P = p * fs
        if s <= P:
            return t + C.RAMP * (s - prev) / max(1e-9, P - prev)
        prev = max(prev, P)
    if s <= fslam:
        return C.T_SLAM + C.RAMP_S * (s - prev) / max(1e-9, fslam - prev)
    return math.inf


def load_at(t, C, fs, fslam):
    tk, pk = hits(C)
    F = prev = 0.0
    for tt, p in zip(tk, pk):
        if t < tt:
            return F
        P = p * fs
        F = prev + (P - prev) * clamp((t - tt) / C.RAMP)
        prev = P
    if t < C.T_SLAM:
        return F
    return prev + (fslam - prev) * clamp((t - C.T_SLAM) / C.RAMP_S)


# ====================================================================== pot outline and curved cuts
def pot_outline():
    """The pot silhouette in unit (u, v) space: rim with an elliptical top, tapered body, slightly round base."""
    pts = []
    for j in range(17):                                         # rim top: upper half of the opening ellipse
        a = math.pi + math.pi * j / 16
        pts.append((0.5 + 0.5 * math.cos(a), 0.085 + 0.085 * math.sin(a)))
    pts += [(0.985, 0.215), (0.93, 0.228)]
    for j in range(9):                                          # base: a shallow arc
        a = math.pi * j / 8
        pts.append((0.5 + 0.31 * math.cos(a), 0.972 + 0.026 * math.sin(a)))
    pts += [(0.07, 0.228), (0.015, 0.215)]
    return pts


def inside(pt, p):
    x, y = pt
    c = False
    n = len(p)
    for i in range(n):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (x1 - x0) * (y - y0) / (y1 - y0):
            c = not c
    return c


def circle_split(p, ccx, ccy, R):
    """Split polygon p (aspect space) by a circle that crosses its boundary exactly twice.

    Returns (a, b, arc) or None; arc runs from the first crossing to the second, inside p."""
    n = len(p)
    cr = []
    for i in range(n):
        ax, ay = p[i]
        bx, by = p[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        fx, fy = ax - ccx, ay - ccy
        A = dx * dx + dy * dy
        if A < 1e-14:
            continue
        B = 2 * (fx * dx + fy * dy)
        Cc = fx * fx + fy * fy - R * R
        disc = B * B - 4 * A * Cc
        if disc <= 0:
            continue
        sq = math.sqrt(disc)
        for s in sorted(((-B - sq) / (2 * A), (-B + sq) / (2 * A))):
            if 0.0 <= s < 1.0:
                cr.append((i, s, (ax + s * dx, ay + s * dy)))
        if len(cr) > 2:
            return None
    if len(cr) != 2 or cr[0][0] == cr[1][0]:
        return None
    (i1, _, q1), (i2, _, q2) = cr
    a1 = math.atan2(q1[1] - ccy, q1[0] - ccx)
    a2 = math.atan2(q2[1] - ccy, q2[0] - ccx)
    dl = (a2 - a1 + math.pi) % TAU - math.pi
    am = a1 + dl / 2
    if not inside((ccx + R * math.cos(am), ccy + R * math.sin(am)), p):
        dl = dl - math.copysign(TAU, dl)
    m = max(2, int(abs(dl) * R / 0.03))
    arc = [(ccx + R * math.cos(a1 + dl * j / m), ccy + R * math.sin(a1 + dl * j / m)) for j in range(m + 1)]
    arc[0], arc[-1] = q1, q2
    a = [q1] + [p[j] for j in range(i1 + 1, i2 + 1)] + arc[::-1][:-1]
    b = [q2] + [p[j % n] for j in range(i2 + 1, i1 + n + 1)] + arc[:-1]
    return a, b, arc


def arc_split(p, rng, C):
    """A curved crack: a ring around the impact point (shell) or a gentle arc pointing at it (radial)."""
    q = [(x, y * ASP) for x, y in p]
    cx, cy, A = poly_ca(q)
    xs, ys = [v[0] for v in q], [v[1] for v in q]
    ext = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    ix, iy = C.IMPACT[0], C.IMPACT[1] * ASP
    dist = math.hypot(cx - ix, cy - iy)
    for attempt in range(24):
        if rng.random() < C.P_RING and dist > 0.25 * ext:
            ccx, ccy = ix + rng.gauss(0, 0.03), iy + rng.gauss(0, 0.04)
            R = math.hypot(cx - ccx, cy - ccy) + rng.gauss(0, 0.14) * ext
        else:
            phi = math.atan2(cy - iy, cx - ix) + rng.gauss(0, 0.30 + 0.05 * attempt)
            off = rng.uniform(-0.18, 0.18) * ext
            px, py = cx - math.sin(phi) * off, cy + math.cos(phi) * off
            R = ext * rng.uniform(1.2, 3.2)
            sg = 1 if rng.random() < 0.5 else -1
            ccx, ccy = px - math.sin(phi) * R * sg, py + math.cos(phi) * R * sg
        if R <= 0.02 * ext:
            continue
        r = circle_split(q, ccx, ccy, R)
        if r is None:
            continue
        a, b, arc = r
        if min(poly_ca(a)[2], poly_ca(b)[2]) >= 0.2 * A:
            un = lambda pts: [(x, y / ASP) for x, y in pts]
            return un(a), un(b), un(arc)
    return None


def weight(C, cu, cv):
    return math.exp(-C.DECAY * math.hypot(cu - C.IMPACT[0], (cv - C.IMPACT[1]) * ASP))


def simulate(seed, C):
    rng = random.Random(seed * 1000003 + 17)
    K, fslam, fs = crush.force_params(seed, C)
    nodes = []

    def add(poly, tb, par, depth, seg=None, arc=None):
        cx, cy, A = poly_ca(poly)
        if par < 0:
            s = K
        else:
            pn = nodes[par]
            s = max(1.10 * pn["s"], pn["s"] * math.sqrt(pn["area"] / A) * math.exp(rng.gauss(0, C.SIGMA)))
        nodes.append(dict(poly=poly, cu=cx, cv=cy, area=A, s=s, w=weight(C, cx, cy), tb=tb, td=math.inf, par=par,
                          depth=depth, cut=seg, arc=arc))
        return len(nodes) - 1

    add(pot_outline(), 0.0, -1, 0)
    i = 0
    while i < len(nodes) and len(nodes) < C.NMAX:
        nd = nodes[i]
        tbk = load_time(nd["s"] / nd["w"], C, fs, fslam)
        if math.isfinite(tbk):
            r = arc_split(nd["poly"], rng, C)
            if r is not None:
                a, b, arc = r
                tbk = max(tbk, nd["tb"] + (C.GEN_DELAY if nd["par"] >= 0 else 0.0))
                nd["td"] = tbk
                seg = (arc[0][0], arc[0][1], arc[-1][0], arc[-1][1])
                add(a, tbk, i, nd["depth"] + 1, seg, arc)
                add(b, tbk, i, nd["depth"] + 1, None)
        i += 1
    return dict(seed=seed, K=K, fslam=fslam, fs=fs, nodes=nodes)


def drama(sim, C):
    st = crush.drama(sim, C)
    if "gates" not in st:
        return st
    early = sum(1 for n in sim["nodes"] if n["td"] <= C.LIVELY_T)
    st["splits_by_2_5s"] = early
    st["gates"]["lively_start"] = bool(early >= C.LIVELY_MIN)
    ok = all(st["gates"].values())
    lost = 100.0 if (st["valid"] and not ok) else 0.0
    st["valid"] = ok
    st["score"] = round(st["score"] - lost + 8.0 * st["gates"]["lively_start"] + 1.5 * min(early, 8), 2)
    return st


# ====================================================================== the scene
class Scene(CrushScene):
    T_TOTAL = 23.8
    T_CONTACT = 0.5                # first hit
    N_HITS = 22
    D0, D1 = 1.25, 0.55            # gap between hits: first, last (the swings come faster)
    P0, PEXP = 0.16, 1.0           # swing height of the first regular hit / the highest; growth curve
    SLAM_PAUSE = 1.75              # last regular hit -> smash (the ball is hauled far back)
    D_PRE = 2.0                    # the swing that is already under way at frame 0
    U_PEAK = 0.55                  # backswing peak, as a fraction of the gap between hits
    RAMP, RAMP_S = 0.04, 0.08
    T_FIRST_MAX = 2.5
    LIVELY_T, LIVELY_MIN = 2.5, 3
    CASCADE_MAX = 0.65
    WHY_DELAY = 0.45
    # model
    K_RANGE = (10, 14)
    SLAM_RANGE = (28, 38)
    SLAM_X = 1.7
    DECAY = 0.9
    IMPACT = (0.12, 0.55)
    P_RING = 0.5
    COUNT_RANGE = (150, 430)
    NMAX = 900
    # layout
    CX, YB, WC, HC = 510.0, 1290.0, 430.0, 540.0
    ANVIL = (250.0, 860.0)
    FOCUS_Y = 900.0
    PUSH = 0.08
    RB = 95.0                      # ball radius
    AMAX = 30.0                    # backswing of the highest regular hit (deg)
    # look
    COOL = (1.00, 0.78, 0.50)
    HOT = (1.00, 0.45, 0.16)
    EDGE_LIGHT = (1.00, 0.80, 0.62)
    EDGE_DARK = (0.18, 0.06, 0.02)
    CHIP_COL = (0.82, 0.44, 0.24)
    QUESTION = ("INTO HOW MANY PIECES?",)
    ROOT = 40
    EQ = ((50, -9.0), (120, -4.5), (300, -1.0), (900, 0.0), (3000, 1.5), (8000, 0.5), (16000, -5.0))
    CEILING_DB = -5.0

    # ---------------------------------------------------------------- model hooks
    @classmethod
    def run_sim(cls, seed):
        return simulate(seed, cls)

    def t_force(self, s):
        return load_time(s, type(self), self.fs, self.fslam)

    def force(self, t):
        return load_at(t, type(self), self.fs, self.fslam)

    def why_lines(self):
        return ("SMALLER SHARDS NEED HARDER HITS",)

    def crack_pts(self, i, seg):
        arc = self.nodes[i].get("arc")
        return arc if arc else super().crack_pts(i, seg)

    def __init__(self, seed):
        super().__init__(seed)
        self.stats = drama(dict(seed=seed, K=self.K, fs=self.fs, fslam=self.fslam, nodes=self.nodes), type(self))
        w = np.array([n["w"] for n in self.nodes])
        self.sa = self.sa / w                                   # effective strength (what the load must reach)
        self.stress = []
        for i, n in enumerate(self.nodes):
            if n["area"] >= 0.010:
                ts = self.t_force(0.86 * n["s"] / n["w"])
                if n["tb"] + 0.12 < ts < min(n["td"] - 0.08, self.T_SLAM):
                    self.stress.append((ts, i))

    # ---------------------------------------------------------------- swing
    def _sched(self):
        if not hasattr(self, "_tk"):
            tk, pk = hits(type(self))
            c = 1 - math.cos(math.radians(self.AMAX))
            self._tk = tk + [self.T_SLAM]
            self._pk = pk + [self.SLAM_X]
            self._ak = [math.acos(max(-1.0, 1 - p * c)) for p in self._pk]      # backswing angle (rad)
        return self._tk, self._pk, self._ak

    def _seg(self, A, u, up):
        """Angle inside one hit-to-hit gap: out to -A (peak at u = up), back in to 0 at u = 1."""
        if u < up:
            return -A * math.sin(0.5 * math.pi * u / up)
        return -A * math.cos(0.5 * math.pi * (u - up) / (1 - up))

    def theta(self, t):
        tk, pk, ak = self._sched()
        up = self.U_PEAK
        tl = self.T_TOTAL - (self.D_PRE - self.T_CONTACT)        # from here on: the frame-0 swing again (loop)
        if t >= tl:
            return self._seg(ak[0], (t - self.T_TOTAL + self.D_PRE - self.T_CONTACT) / self.D_PRE, up)
        if t >= self.T_SLAM:                                     # carries on through, swings back out
            D = tl - self.T_SLAM
            th1 = ak[0] * D / (2 * up * self.D_PRE)
            return th1 * math.sin(math.pi * (t - self.T_SLAM) / D)
        if t < tk[0]:
            return self._seg(ak[0], (t - tk[0] + self.D_PRE) / self.D_PRE, up)
        j = next(k for k, tt in enumerate(tk) if tt > t)
        g = tk[j] - tk[j - 1]
        return self._seg(ak[j], (t - tk[j - 1]) / g, 0.40 if j == len(tk) - 1 else up)

    def ball_xy(self, t):
        th = self.theta(t)
        return self.PIV[0] + self.L * math.sin(th), self.PIV[1] + self.L * math.cos(th)

    @property
    def PIV(self):
        u = 0.07 + 0.12 * (self.IMPACT[1] - 0.228) / 0.744
        x = self.CX + (u - 0.5) * self.WC - self.RB
        return x, 330.0

    @property
    def L(self):
        return (self.YB - (1 - self.IMPACT[1]) * self.HC) - 330.0

    def pulse(self, t):
        tk, pk, _ = self._sched()
        best = 0.0
        for tt, p in zip(tk, pk):
            a = t - tt
            if 0 <= a < 0.14:
                best = max(best, min(1.0, p) * math.sin(math.pi * a / 0.14))
        return best

    def geom(self, t):
        F = self.force(t)
        pp = min(1.0, F / self.fs)
        jolt = self.pulse(t) if t < self.T_SLAM else 0.0
        return F, pp, 1.0 - 0.012 * jolt, 1.0 + 0.012 * jolt, 0.0

    def phys(self, xc, i, tau, spread):
        """After the smash: shards are knocked right (hardest near the impact), pop up, land, slide, fall off."""
        cu, cv = self.cuv[i]
        near = weight(type(self), cu, cv)
        sm = 0.5 + 0.5 * math.sin(7.0 * cu + 2.0) * math.cos(5.0 * cv + 1.0)
        s = (xc - self.CX) / (self.WC / 2)
        v = 70.0 + 520.0 * near * (0.55 + 0.45 * sm) + 110.0 * s
        k, g = 2.2, 2600.0
        h = (1.0 - cv) * self.HC
        rr = 0.35 * math.sqrt(self.area[i] * self.WC * self.HC)
        floor = max(0.0, h - rr)
        vy = 260.0 + 380.0 * near * sm
        dy = min(floor, -vy * tau + 0.5 * g * tau * tau)
        tland = (vy + math.sqrt(vy * vy + 2 * g * floor)) / g
        dx = v * (1 - math.exp(-k * tau)) / k
        rot = (2 * sm - 1) * 5.0 * min(tau, tland)
        edge = self.ANVIL[1] - xc
        if tau > tland and dx > edge:
            te = -math.log(max(1e-6, 1 - edge * k / v)) / k if edge * k < v else math.inf
            if math.isfinite(te):
                dy += 0.5 * g * (tau - max(te, tland)) ** 2
                rot += 2.6 * (tau - max(te, tland))
        return dx, dy, rot

    # ---------------------------------------------------------------- look
    def make_background(self):
        H, W = self.H, self.W
        yy = np.arange(H, dtype=np.float32)
        stops = ((0, (0.045, 0.030, 0.050)), (500, (0.085, 0.055, 0.075)), (1100, (0.140, 0.085, 0.080)),
                 (1300, (0.060, 0.040, 0.045)), (1920, (0.020, 0.014, 0.020)))
        col = np.stack([np.interp(yy, [a for a, _ in stops], [b[c] for _, b in stops]) for c in range(3)], axis=1)
        buf = np.zeros((H, W, 4), np.uint8)
        buf[..., :3] = (np.clip(col[:, None, ::-1], 0, 1) * 255).astype(np.uint8)
        buf[..., 3] = 255
        rng = np.random.default_rng(13)
        buf[..., :3] = np.clip(buf[..., :3].astype(np.int16) + rng.integers(-2, 3, (H, W, 1)), 0, 255).astype(np.uint8)
        surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
        ctx = cairo.Context(surf)
        ctx.set_source_rgba(1.0, 0.75, 0.6, 0.04)              # faint brick courses on the back wall
        ctx.set_line_width(3)
        for r, y in enumerate(range(420, 1300, 62)):
            ctx.move_to(0, y)
            ctx.line_to(W, y)
            for x in range(-60 + 65 * (r % 2), W, 130):
                ctx.move_to(x, y)
                ctx.line_to(x, y + 62)
        ctx.stroke()
        return surf, buf

    def make_texture(self):
        """Fired terracotta: cylinder shading, glossy highlight stripe, rim with a lit top, dark opening, clay grain."""
        w, h = int(self.WC), int(self.HC)
        u = (np.arange(w) + 0.5)[None, :] / w
        v = (np.arange(h) + 0.5)[:, None] / h
        uL = np.where(v < 0.228, 0.0, 0.07 + 0.12 * (v - 0.228) / 0.744)
        s = np.clip((u - uL) / (1 - 2 * uL), 0, 1)
        n = 2 * s - 1
        nz = np.sqrt(np.clip(1 - n * n, 0, 1))
        bright = 0.42 + 0.62 * np.clip(-0.45 * n + 0.85 * nz, 0, 1)
        spec = 0.22 * np.exp(-((s - 0.30) / 0.06) ** 2) + 0.10 * np.exp(-((s - 0.30) / 0.20) ** 2)
        spec += 0.06 * np.exp(-((s - 0.86) / 0.03) ** 2)                      # rim light on the far edge
        base = np.where(v[..., None] < 0.228, np.array([0.80, 0.42, 0.25]), np.array([0.72, 0.35, 0.19]))
        base = base * (1.0 - 0.12 * np.clip((v[..., None] - 0.3) / 0.7, 0, 1))
        rng = np.random.default_rng(5)
        grain = 1.0 + 0.045 * rng.normal(0, 1, (h, w))
        img = base * (bright * grain)[..., None] + spec[..., None] * np.array([1.0, 0.90, 0.80])
        sh = np.clip((v - 0.228) / 0.05, 0, 1)                                 # the rim's shadow on the body
        img *= np.where(v >= 0.228, 0.55 + 0.45 * sh, 1.0)[..., None]
        line = np.exp(-((v - 0.218) / 0.006) ** 2)                             # dark lip under the rim
        img *= (1 - 0.5 * line)[..., None]
        band = np.exp(-((v - 0.30) / 0.008) ** 2) + np.exp(-((v - 0.86) / 0.006) ** 2)   # two thrown grooves
        img *= (1 - 0.22 * band)[..., None]
        ex = (u - 0.5) / 0.5
        eo = ex ** 2 + ((v - 0.085) / 0.085) ** 2                               # outer rim ellipse
        ei = ((u - 0.5) / 0.43) ** 2 + ((v - 0.088) / 0.062) ** 2                # opening
        top = (eo <= 1.0) & (v < 0.17)
        img = np.where(top[..., None], img * 1.12 + 0.05, img)
        inner = ei <= 1.0
        dep = np.clip((v - 0.026) / 0.124, 0, 1)
        dark = np.stack([0.30 - 0.20 * dep, 0.13 - 0.09 * dep, 0.06 - 0.04 * dep], -1)
        img = np.where(inner[..., None], dark * (0.75 + 0.25 * np.clip(1 - np.abs(ex), 0, 1))[..., None], img)
        ring = np.abs(np.sqrt(np.maximum(ei, 0)) - 1.0) < 0.035
        img = np.where((ring & (v > 0.088))[..., None], img * 0.55, img)
        spk = rng.random((h, w)) < 0.004                                        # grog specks in the clay
        img = np.where(spk[..., None], img * 0.6, img)
        img = np.clip(img, 0, 1)
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        st = surf.get_stride()
        a = np.ndarray((h, st // 4, 4), np.uint8, surf.get_data())
        a[:, :w, :3] = (img[..., ::-1] * 255).astype(np.uint8)
        a[:, :w, 3] = 255
        surf.mark_dirty()
        return surf

    def make_rig(self):
        W, H, YB = self.W, self.H, self.YB
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        ctx = cairo.Context(surf)
        a0, a1 = self.ANVIL
        for (x, y, w_, h_, c0, c1) in ((a0 + 70, YB + 60, a1 - a0 - 140, 36, (0.30, 0.28, 0.28), (0.16, 0.15, 0.16)),
                                       (a0, YB, a1 - a0, 72, (0.58, 0.55, 0.52), (0.36, 0.34, 0.33))):
            rounded_rect(ctx, x, y, w_, h_, 8)                    # concrete plinth
            g = cairo.LinearGradient(0, y, 0, y + h_)
            g.add_color_stop_rgb(0, *c0)
            g.add_color_stop_rgb(1, *c1)
            ctx.set_source(g)
            ctx.fill_preserve()
            ctx.set_source_rgba(0.03, 0.02, 0.03, 0.9)
            ctx.set_line_width(4)
            ctx.stroke()
        rng = np.random.default_rng(3)
        for _ in range(140):
            ctx.arc(rng.uniform(a0 + 8, a1 - 8), rng.uniform(YB + 8, YB + 66), rng.uniform(0.8, 2.2), 0, TAU)
            ctx.set_source_rgba(0.2, 0.2, 0.2, 0.35)
            ctx.fill()
        ctx.set_source_rgba(1.0, 0.92, 0.85, 0.55)
        ctx.rectangle(a0 + 6, YB + 1, a1 - a0 - 12, 3)
        ctx.fill()
        px, py = self.PIV                                            # crane jib (I-beam) and pulley
        rounded_rect(ctx, -30, py - 46, px + 120, 34, 4)
        g = cairo.LinearGradient(0, py - 46, 0, py - 12)
        g.add_color_stop_rgb(0, 0.95, 0.70, 0.18)
        g.add_color_stop_rgb(1, 0.62, 0.40, 0.06)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.05, 0.03, 0.02, 0.95)
        ctx.set_line_width(4)
        ctx.stroke()
        ctx.set_source_rgba(0.05, 0.03, 0.02, 0.55)
        for x in range(0, int(px + 80), 46):
            ctx.move_to(x, py - 44)
            ctx.line_to(x + 23, py - 14)
            ctx.line_to(x + 46, py - 44)
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.new_path()
        ctx.arc(px, py, 20, 0, TAU)
        ctx.set_source_rgb(0.25, 0.26, 0.29)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.02, 0.02, 0.03, 0.95)
        ctx.set_line_width(4)
        ctx.stroke()
        surf.flush()
        return surf, []

    def _press(self, ctx, t, hc, pp, heat):
        """The wrecking ball: chain from the jib pulley, steel ball with a moving highlight, ghosts when fast."""
        px, py = self.PIV
        RB = self.RB
        dt = 1.0 / self.FPS
        bx, by = self.ball_xy(t)
        bx0, by0 = self.ball_xy(t - dt)
        sp = math.hypot(bx - bx0, by - by0)
        th = self.theta(t)
        for j in (3, 2, 1):                                       # motion ghosts
            if sp > 18:
                gx, gy = self.ball_xy(t - dt * 0.33 * j)
                ctx.new_path()
                ctx.arc(gx, gy, RB, 0, TAU)
                ctx.set_source_rgba(0.55, 0.58, 0.64, min(0.22, 0.004 * sp) / j)
                ctx.fill()
        cx_, cy_ = bx - math.sin(th) * (RB + 14), by - math.cos(th) * (RB + 14)
        Lc = math.hypot(cx_ - px, cy_ - py)
        ux, uy = (cx_ - px) / Lc, (cy_ - py) / Lc
        nlink = int(Lc / 26)
        for j in range(nlink + 1):                                # chain links, alternating face-on / edge-on
            lx, ly = px + ux * 26 * j, py + uy * 26 * j
            ctx.save()
            ctx.translate(lx, ly)
            ctx.rotate(math.atan2(uy, ux))
            ctx.new_path()
            if j % 2:
                rounded_rect(ctx, -17, -4, 34, 8, 4)
                ctx.set_source_rgb(0.30, 0.31, 0.34)
                ctx.fill()
            else:
                ctx.save()
                ctx.scale(1.0, 0.55)
                ctx.arc(0, 0, 17, 0, TAU)
                ctx.restore()
                ctx.set_source_rgb(0.08, 0.08, 0.10)
                ctx.set_line_width(9)
                ctx.stroke_preserve()
                ctx.set_source_rgb(0.62, 0.64, 0.70)
                ctx.set_line_width(4)
                ctx.stroke()
            ctx.restore()
        ctx.save()                                                # shackle
        ctx.translate(cx_, cy_)
        ctx.rotate(-th)
        rounded_rect(ctx, -16, -6, 32, 26, 6)
        ctx.set_source_rgb(0.22, 0.23, 0.26)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.02, 0.02, 0.03, 0.95)
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.restore()
        ctx.new_path()                                            # the ball
        ctx.arc(bx, by, RB, 0, TAU)
        g = cairo.RadialGradient(bx - 0.38 * RB, by - 0.42 * RB, 4, bx, by, RB)
        g.add_color_stop_rgb(0, 0.92, 0.93, 0.96)
        g.add_color_stop_rgb(0.18, 0.52, 0.54, 0.60)
        g.add_color_stop_rgb(0.7, 0.20, 0.21, 0.25)
        g.add_color_stop_rgb(1, 0.07, 0.07, 0.09)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.01, 0.01, 0.02, 0.95)
        ctx.set_line_width(5)
        ctx.stroke()
        ctx.new_path()
        ctx.arc(bx, by, RB - 7, math.radians(10), math.radians(80))   # warm rim light from the pot side
        ctx.set_source_rgba(1.0, 0.72, 0.45, 0.55)
        ctx.set_line_width(4)
        ctx.stroke()
        tk, pk, _ = self._sched()                                  # impact flash + dust at the contact point
        x0, y0 = bx + RB, by
        for tt, p in zip(tk, pk):
            a = t - tt
            if 0 <= a < 0.35:
                k = min(1.6, p)
                ctx.save()
                ctx.set_operator(cairo.OPERATOR_ADD)
                glow(ctx, x0, y0, 60 + 120 * k, (1.0, 0.80, 0.55), 0.55 * k * math.exp(-a / 0.08))
                ctx.restore()
                for q in range(int(4 + 8 * k)):
                    h1, h2 = crush.hash1(q, tt), crush.hash1(q + 9, tt)
                    ang = math.pi * (h1 - 0.5) * 1.6 + math.pi
                    r = (30 + 160 * k * h2) * ease_out(a / 0.35)
                    ctx.new_path()
                    ctx.arc(x0 + math.cos(ang) * r * 0.6, y0 + math.sin(ang) * r - 40 * a, 6 + 10 * h2, 0, TAU)
                    ctx.set_source_rgba(0.80, 0.58, 0.42, 0.35 * (1 - a / 0.35))
                    ctx.fill()

    # ---------------------------------------------------------------- sound
    def audio(self):
        rng = np.random.default_rng(self.seed + 991)
        sr = au.SR
        TS = self.T_SLAM
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        bed[:n] = au.tension_bed(self.dur, TS, root=self.ROOT, seed=self.seed % 97 + 5, gain=0.50)
        tk, pk, ak = self._sched()
        tsec = np.arange(n) / sr
        duck = np.ones(n)
        for tt in tk:                                            # the bed dips under each incoming swing
            a0 = tt - 0.22
            m = (tsec >= a0) & (tsec < tt + 0.10)
            x = tsec[m]
            dip = np.where(x < tt, np.clip((x - a0) / 0.10, 0, 1), np.clip(1 - (x - tt) / 0.10, 0, 1))
            duck[m] = np.minimum(duck[m], 1.0 - 0.45 * dip)
        bed[:n] *= duck[:, None]
        N = len(tk)
        for k, (tt, p, A) in enumerate(zip(tk, pk, ak)):
            smash = k == N - 1
            kk = k / max(1, N - 2)
            tp = tk[k - 1] if k else -1.0
            wl = min(0.55, 0.5 * (tt - tp), tt - 0.01) if not smash else 0.9   # whoosh as the ball comes in
            wn = au.noise(rng, wl, 300 + 500 * kk, 1800 + 2500 * kk)
            env = np.linspace(0, 1, len(wn)) ** 2.2
            au.place(fx, wn * env, tt - wl, (0.10 + 0.10 * min(p, 1)) if not smash else 0.30, -0.5)
            if k:                                                    # chain rattle at the backswing peak
                tpk = tp + (0.40 if smash else self.U_PEAK) * (tt - tp)
                for q in range(3):
                    au.place(fx, au.tick(2600 + 1400 * crush.hash1(q, tt), 0.04), tpk + 0.035 * q, 0.05 + 0.03 * kk,
                             -0.6)
            if smash:
                continue
            g = 0.22 + 0.30 * p
            au.place(fx, au.boom(rng, 0.40, 150.0 + 30 * kk, 70.0), tt, 0.65 * g)          # thud
            au.place(fx, au.bell(330 * 2 ** (kk * 0.85), 0.45), tt + 0.002, 0.08 + 0.09 * p, -0.2)   # steel clank
            au.place(fx, au.marimba(620 * 2 ** (kk * 0.9), 0.22), tt + 0.003, 0.10 + 0.08 * p)   # clay tonk
            nc = int(0.07 * sr)
            au.place(fx, au.noise(rng, 0.07, 700, 6500)[:nc] * np.exp(-np.arange(nc) / sr / 0.018), tt, 0.22 + 0.40 * p)
        slots = {}                                                # clay cracks: dry crackle + a short ceramic tink
        for nd in self.nodes:
            if math.isfinite(nd["td"]):
                slots.setdefault(round(nd["td"], 2), []).append(nd)
        for te, group in sorted(slots.items()):
            te = min(te, self.dur - 0.3)
            m = len(group)
            k = clamp((te - self.T_CONTACT) / (TS - self.T_CONTACT))
            for q, nd in enumerate(group[:7]):
                tq = te + 0.011 * q
                fb = 1000.0 * max(0.03, nd["area"]) ** -0.3
                nc = int(0.05 * sr)
                cl = (au.noise(rng, 0.05, 1500, 9000)[:nc] * np.minimum(1.0, np.arange(nc) / (0.0008 * sr))
                      * np.exp(-np.arange(nc) / sr / 0.010))
                pan = clamp((nd["cu"] - 0.5) * 1.6, -0.8, 0.8)
                gn = ((0.42 + 0.28 * k) if te < TS else 0.42) / (1.0 + 0.35 * (m - 1)) ** 0.5
                au.place(fx, cl, tq, 0.65 * gn, pan)
                au.place(fx, au.marimba(fb * (1 + 0.18 * crush.hash1(q, te)), 0.18), tq + 0.002, 0.14 * gn, pan)
                if nd["area"] > 0.03:
                    au.place(fx, au.boom(rng, 0.30, 210.0, 90.0), tq, 0.22 * math.sqrt(nd["area"]))
        for ts, i in self.stress:
            au.place(fx, au.tick(2300 + 800 * crush.hash1(i, 2.0), 0.05), ts, 0.07,
                     clamp(crush.hash1(i, 3.0) * 1.6 - 0.8, -0.7, 0.7))
        au.place(fx, au.boom(rng, 1.4, 95.0, 34.0), TS, 0.85)          # the smash
        au.place(fx, au.bell(260.0, 1.0), TS + 0.003, 0.12)
        nr = np.arange(int(1.8 * sr)) / sr
        au.place(fx, au.noise(rng, 1.8, 40, 600) * np.exp(-nr / 0.7) * np.minimum(1.0, nr / 0.01), TS, 0.9)
        ns = np.arange(int(0.9 * sr)) / sr
        au.place(fx, au.noise(rng, 0.9, 2000, 11000) * np.exp(-ns / 0.22) * np.minimum(1.0, ns / 0.004), TS + 0.02, 0.38)
        for q in range(30):                                             # shards landing on concrete
            tq = TS + 0.25 + 1.1 * crush.hash1(q, 3.0) ** 1.4
            au.place(fx, au.marimba(900 + 2200 * crush.hash1(q, 4.0), 0.10), tq, 0.08,
                     clamp(crush.hash1(q, 5.0) * 1.4 - 0.4, -0.8, 0.8))
            au.place(fx, au.tick(3000 + 2000 * crush.hash1(q, 6.0), 0.03), tq, 0.06)
        for j, m_ in enumerate((self.ROOT + 36, self.ROOT + 43, self.ROOT + 48)):
            au.place(fx, au.bell(au.hz(m_), 1.6), self.t_rev + 0.01 * j, 0.16)
        for j, m_ in enumerate((60, 64, 67, 72)):
            au.place(fx, au.marimba(au.hz(m_), 0.4), self.T_TOTAL - 0.62 + 0.13 * j, 0.10)
        mix_ = au.reverb(fx, rng, decay=1.0, wet=0.10) + bed
        out_ = mix_[:n].copy()
        out_[:tail] += mix_[n:]
        out_ = au.eq(out_, self.EQ)
        out_[:int(0.04 * sr)] *= np.linspace(0, 1, int(0.04 * sr))[:, None]
        out_[-int(0.25 * sr):] *= np.linspace(1, 0, int(0.25 * sr))[:, None]
        pk_ = np.abs(out_).max()
        return np.tanh(out_ * (self.DRIVE / max(1e-9, pk_)))

    # ---------------------------------------------------------------- metadata
    def metadata(self):
        return dict(
            title="The Wrecking Ball Hit One Clay Pot... Into How Many Pieces? \U0001F4A5",
            description=(
                "Comment your number \U0001F447 How many pieces did you guess?\n"
                "A steel wrecking ball swings into one big terracotta flower pot again and again. Every swing starts "
                "a little higher and comes a little faster, so every hit lands harder. The last swing starts from "
                "much higher than any before it.\n"
                "A piece breaks when the hardest hit so far beats its strength, and pieces far from the impact feel "
                "less of it. Fired clay cracks along curved shell lines that ring the impact point. Smaller shards "
                "are stronger, so each new crack needs a harder hit.\n"
                "The counter shows the real count from the simulation the whole time.\n"
                "No footage, samples or stock media.\n"
                "#simulation #satisfying #physics"),
            tags=["simulation", "wrecking ball", "clay pot", "terracotta", "flower pot", "how many pieces",
                  "satisfying", "physics", "crush", "shatter", "made with code", "oddly satisfying"],
            pinned_comment="Comment your number \U0001F447 How many pieces did the wrecking ball smash the clay pot into?",
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of a wrecking ball and a clay pot, "
                                              "generated by a toy fracture model; not realistic footage of people, "
                                              "places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / self.FPS, 2),
            answer=int(self.n_final),
            drama=self.stats,
        )


def _t_slam(C):
    tk, _ = hits(C)
    return round(tk[-1] + C.SLAM_PAUSE, 3)


Scene.T_SLAM = _t_slam(Scene)
Scene.T_TOTAL = round(Scene.T_SLAM + 3.0, 1)     # the answer shows for about the last 2.4 s


def score_seed(seed):
    return drama(simulate(seed, Scene), Scene)
