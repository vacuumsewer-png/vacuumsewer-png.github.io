"""CHOCOLATE CRUSH (CRUSH family, episode 2): a steel weight drops on one big chocolate bar, each time from a little
higher. Into how many pieces?

Changes from episode 1 (icecrush, hydraulic press): a different object, a different impactor and a different load.

Impactor / load: a 640 px steel weight runs between two guide rails. It is hoisted and dropped N_DROPS times, the drop
height climbing linearly from H0 to HMAX px; then one last drop from SLAM_X * HMAX (the slam). Impact force is
weight x drop height / stopping distance, so the peak load of a drop is proportional to its height:
P_k = fs * h_k / HMAX, slam = SLAM_X * fs. Each impact ramps the load up over RAMP s (contact time). A piece breaks when
the highest load so far passes its strength (brittle: it does not heal between drops). Load is a staircase, so the
cracks come in bursts on the hits.

Chocolate fracture model (same cascade as engine/crush.py, plus the moulded grooves):
- bar strength K drawn from K_RANGE; slam force from SLAM_RANGE (x10); fs = slam / SLAM_X;
- each piece has a material strength si = parent si * sqrt(parent area / child area) * exp(N(0, SIGMA))
  (small pieces are stronger; the bar's si = K / GROOVE_W);
- a piece that still holds a groove line (one that would split it no worse than 20/80) breaks at
  GROOVE_W = 0.70 of that (the groove is the thin section of the bar); a groove-free piece at its full si;
- breaking strength = max(1.10 * parent's breaking strength, si * (GROOVE_W or 1));
- the cut follows that groove with probability P_ALONG = 0.90 (nearest groove to the centroid, slight jitter),
  otherwise it is a random line as in the engine. So the first cracks snap along the grooves into squares,
  and then the squares (no groove left, full strength) need more and more force.
The count is the real simulation result; seeds are only filtered by the template's drama gates (COUNT_RANGE etc.).

On screen: one question line and one counter. The answer shows only after the slam cascade, then the why line,
then a cross-fade back to frame 0.
"""
import math
import random

import cairo
import numpy as np

from engine import audio as au
from engine import crush
from engine.crush import CrushScene, brushed_metal, clip, poly_ca, split_poly
from engine.gfx import TAU, clamp, ease_in, ease_out, glow, rounded_rect


# ====================================================================== drop schedule (class-level, not seed dependent)
def drops(C):
    """Impact times and peak-load fractions of the regular drops (the slam is extra, at C.T_SLAM)."""
    n = C.N_DROPS
    dt = (C.T_SLAM - C.SLAM_PAUSE - C.T_CONTACT) / (n - 1)
    tk = [C.T_CONTACT + k * dt for k in range(n)]
    hk = [C.H0 + (C.HMAX - C.H0) * k / (n - 1) for k in range(n)]
    return tk, hk, [h / C.HMAX for h in hk]


def load_time(s, C, fs, fslam):
    """First time the running-max load reaches s (inf if never)."""
    tk, _, pk = drops(C)
    if s <= 0:
        return tk[0]
    prev = 0.0
    for t, p in zip(tk, pk):
        P = p * fs
        if s <= P:
            return t + C.RAMP * (s - prev) / max(1e-9, P - prev)
        prev = P
    if s <= fslam:
        return C.T_SLAM + C.RAMP_S * (s - fs) / max(1e-9, fslam - fs)
    return math.inf


def load_at(t, C, fs, fslam):
    """Running-max load (what fractures the bar)."""
    tk, _, pk = drops(C)
    F = 0.0
    prev = 0.0
    for tt, p in zip(tk, pk):
        if t < tt:
            return F
        P = p * fs
        F = prev + (P - prev) * clamp((t - tt) / C.RAMP)
        prev = P
    if t < C.T_SLAM:
        return F
    return fs + (fslam - fs) * clamp((t - C.T_SLAM) / C.RAMP_S)


# ====================================================================== grooves
def groove_cands(p, C):
    """Groove lines (u = i/COLS or v = j/ROWS) that split p with both parts >= 20% of its area."""
    _, _, A = poly_ca(p)
    xs = [q[0] for q in p]
    ys = [q[1] for q in p]
    out = []
    for nx, ny, lines, lo, hi in ((1.0, 0.0, [i / C.COLS for i in range(1, C.COLS)], min(xs), max(xs)),
                                  (0.0, 1.0, [j / C.ROWS for j in range(1, C.ROWS)], min(ys), max(ys))):
        for d in lines:
            if not (lo + 1e-6 < d < hi - 1e-6):
                continue
            a, ha = clip(p, nx, ny, d, True)
            b, _ = clip(p, nx, ny, d, False)
            if len(a) < 3 or len(b) < 3:
                continue
            if min(poly_ca(a)[2], poly_ca(b)[2]) >= 0.2 * A:
                out.append((nx, ny, d, a, b, ha))
    return out


def simulate(seed, C):
    rng = random.Random(seed * 1000003 + 17)
    K, fslam, fs = crush.force_params(seed, C)
    root = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    nodes = []

    def add(poly, tb, par, depth, seg=None):
        cx, cy, A = poly_ca(poly)
        w = C.GROOVE_W if groove_cands(poly, C) else 1.0
        if par < 0:
            si = K / w                     # the bar's own (groove-free) strength; the grooves make it K
        else:
            pn = nodes[par]
            si = pn["si"] * math.sqrt(pn["area"] / A) * math.exp(rng.gauss(0, C.SIGMA))
        s = K if par < 0 else max(1.10 * nodes[par]["s"], si * w)
        nodes.append(dict(poly=poly, cu=cx, cv=cy, area=A, s=s, si=si, tb=tb, td=math.inf, par=par, depth=depth,
                          cut=seg))
        return len(nodes) - 1

    add(root, 0.0, -1, 0)
    i = 0
    while i < len(nodes) and len(nodes) < C.NMAX:
        nd = nodes[i]
        tbk = load_time(nd["s"], C, fs, fslam)
        if math.isfinite(tbk):
            tbk = max(tbk, nd["tb"] + (C.GEN_DELAY if nd["par"] >= 0 else 0.0))
            cands = groove_cands(nd["poly"], C)
            if cands and rng.random() < C.P_ALONG:
                cx, cy = nd["cu"], nd["cv"]
                cands.sort(key=lambda c: abs((cx if c[0] else cy) - c[2]) + 0.04 * rng.random())
                nx, ny, d, a, b, hits = cands[0]
                seg = (hits[0][0], hits[0][1], hits[-1][0], hits[-1][1])
            else:
                a, b, seg = split_poly(nd["poly"], rng)
            nd["td"] = tbk
            add(a, tbk, i, nd["depth"] + 1, seg)
            add(b, tbk, i, nd["depth"] + 1, None)
        i += 1
    return dict(seed=seed, K=K, fslam=fslam, fs=fs, nodes=nodes)


# ====================================================================== the scene
class Scene(CrushScene):
    # drop schedule
    N_DROPS = 22
    H0, HMAX = 28.0, 250.0        # px above the bar top: first and last regular drop
    SLAM_PAUSE = 1.5               # last regular drop -> slam (the weight goes all the way up)
    RAMP, RAMP_S = 0.05, 0.08      # impact contact time (s): regular drops, slam
    G = 2600.0                     # px/s^2 for the falling weight
    WT_H = 140.0
    T_FIRST_MAX = 2.5
    # chocolate
    COLS, ROWS = 5, 4
    GROOVE_W = 0.70
    P_ALONG = 0.90
    K_RANGE = (22, 28)
    SLAM_RANGE = (36, 48)
    SLAM_X = 2.0
    # look
    COOL = (0.98, 0.80, 0.55)
    HOT = (1.00, 0.50, 0.18)
    EDGE_LIGHT = (1.00, 0.86, 0.68)
    EDGE_DARK = (0.12, 0.05, 0.02)
    CHIP_COL = (0.50, 0.29, 0.15)
    QUESTION = ("HOW MANY PIECES", "WILL THIS BAR BECOME?")
    ROOT = 41
    EQ = ((50, -9.0), (120, -4.5), (300, -1.0), (900, 0.0), (3000, 1.5), (8000, 0.5), (16000, -5.0))
    CEILING_DB = -5.0             # crisp snap clicks overshoot after AAC; keep true peak under -1 dBTP

    # ---------------------------------------------------------------- model hooks
    @classmethod
    def run_sim(cls, seed):
        return simulate(seed, cls)

    def t_force(self, s):
        return load_time(s, type(self), self.fs, self.fslam)

    def force(self, t):
        return load_at(t, type(self), self.fs, self.fslam)

    def why_lines(self):
        return ("IT SNAPS ALONG THE GROOVES FIRST,", "THEN SMALL BITS NEED MORE FORCE")

    # ---------------------------------------------------------------- motion
    def _sched(self):
        if not hasattr(self, "_tk"):
            tk, hk, pk = drops(type(self))
            self._tk = tk + [self.T_SLAM]
            self._hk = hk + [self.HMAX * self.SLAM_X]
            self._pk = pk + [self.SLAM_X]
        return self._tk, self._hk, self._pk

    def pulse(self, t):
        tk, _, pk = self._sched()
        best = 0.0
        for tt, p in zip(tk[:-1], pk[:-1]):
            a = t - tt
            if 0 <= a < 0.12:
                best = max(best, p * math.sin(math.pi * a / 0.12))
        return best

    def wh(self, t):
        """Height of the weight's bottom above the bar top (before the slam)."""
        tk, hk, pk = self._sched()
        j = next(k for k, tt in enumerate(tk) if tt > t)
        tp = tk[j - 1] if j > 0 else -0.6
        pprev = pk[j - 1] if j > 0 else 0.0
        hj = hk[j]
        tf = math.sqrt(2 * hj / self.G)
        t_rel = tk[j] - tf
        t_hold = t_rel - 0.05
        ls = tp + 0.14
        if t >= t_rel:
            return max(0.0, hj - 0.5 * self.G * (t - t_rel) ** 2)
        if t >= t_hold:
            return hj
        a = t - tp
        if a < 0.14:
            return 12.0 * min(1.0, pprev) * math.sin(math.pi * a / 0.14)
        u = clamp((t - ls) / max(1e-6, t_hold - ls))
        return hj * u * u * (3 - 2 * u)

    def geom(self, t):
        F = self.force(t)
        pp = min(1.0, F / self.fs)
        d = t - self.T_SLAM
        sl = ease_in(clamp(d / 0.22), 2) if d > 0 else 0.0
        pu = self.pulse(t) if d <= 0 else 0.0
        hc = (1.0 - 0.035 * pu) * (1 - sl) + 0.34 * sl
        spread = 1.0 + 0.015 * pu + 0.16 * sl
        if d > 0.22:
            hc += 0.014 * math.exp(-(d - 0.22) / 0.25) * math.sin(26 * (d - 0.22))
        return F, pp, hc, spread, sl

    def plate_y(self, t, hc):
        top = self.YB - self.HC * hc
        d = t - self.T_SLAM
        if d >= 0:
            y = top
            if d > 0.45:
                y -= 420.0 * ease_out(clamp((d - 0.45) / 0.9), 3)
            return y
        return top - self.wh(t)

    # ---------------------------------------------------------------- look
    def make_texture(self):
        """Milk chocolate: COLS x ROWS raised squares with bevelled edges, dark grooves, a glossy sheen."""
        w, h = int(self.WC), int(self.HC)
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surf)
        ctx.set_source_rgb(0.20, 0.10, 0.05)                       # groove colour
        ctx.paint()
        cw, ch = w / self.COLS, h / self.ROWS
        gap, bev = 6.0, 15.0
        rng = np.random.default_rng(7)
        for i in range(self.COLS):
            for j in range(self.ROWS):
                x0, y0 = i * cw + gap, j * ch + gap
                x1, y1 = (i + 1) * cw - gap, (j + 1) * ch - gap
                # bevel faces: top + left lit, bottom + right shaded
                faces = (((x0, y0), (x1, y0), (x1 - bev, y0 + bev), (x0 + bev, y0 + bev), (0.72, 0.47, 0.29)),
                         ((x0, y0), (x0 + bev, y0 + bev), (x0 + bev, y1 - bev), (x0, y1), (0.60, 0.37, 0.21)),
                         ((x0, y1), (x0 + bev, y1 - bev), (x1 - bev, y1 - bev), (x1, y1), (0.27, 0.13, 0.06)),
                         ((x1, y0), (x1, y1), (x1 - bev, y1 - bev), (x1 - bev, y0 + bev), (0.33, 0.17, 0.08)))
                for *pts, col in faces:
                    ctx.move_to(*pts[0])
                    for q in pts[1:]:
                        ctx.line_to(*q)
                    ctx.close_path()
                    ctx.set_source_rgb(*col)
                    ctx.fill()
                fx0, fy0, fx1, fy1 = x0 + bev, y0 + bev, x1 - bev, y1 - bev   # top face
                g = cairo.LinearGradient(fx0, fy0, fx1, fy1)
                g.add_color_stop_rgb(0, 0.56, 0.33, 0.18)
                g.add_color_stop_rgb(1, 0.42, 0.23, 0.12)
                ctx.rectangle(fx0, fy0, fx1 - fx0, fy1 - fy0)
                ctx.set_source(g)
                ctx.fill()
                g = cairo.RadialGradient(fx0 + 0.3 * (fx1 - fx0), fy0 + 0.25 * (fy1 - fy0), 2,
                                         fx0 + 0.3 * (fx1 - fx0), fy0 + 0.25 * (fy1 - fy0), 0.6 * (fx1 - fx0))
                g.add_color_stop_rgba(0, 1.0, 0.90, 0.78, 0.30)            # soft specular on each square
                g.add_color_stop_rgba(1, 1.0, 0.90, 0.78, 0.0)
                ctx.rectangle(fx0, fy0, fx1 - fx0, fy1 - fy0)
                ctx.set_source(g)
                ctx.fill()
                ctx.set_source_rgba(1.0, 0.88, 0.72, 0.55)                  # crisp lit edge line
                ctx.set_line_width(2.0)
                ctx.move_to(x0 + 1, y1 - 1)
                ctx.line_to(x0 + 1, y0 + 1)
                ctx.line_to(x1 - 1, y0 + 1)
                ctx.stroke()
                for _ in range(10):                                           # tiny surface specks (cocoa bloom)
                    ctx.arc(rng.uniform(fx0, fx1), rng.uniform(fy0, fy1), rng.uniform(0.8, 2.0), 0, TAU)
                    ctx.set_source_rgba(0.30, 0.16, 0.08, 0.35)
                    ctx.fill()
        g = cairo.LinearGradient(0, 0, w, h)                                  # glossy diagonal sheen
        for s, a in ((0.16, 0.0), (0.25, 0.20), (0.31, 0.0), (0.40, 0.0), (0.44, 0.12), (0.48, 0.0)):
            g.add_color_stop_rgba(s, 1, 0.95, 0.88, a)
        ctx.set_source(g)
        ctx.paint()
        for (x0, y0, x1, y1, a) in ((0, h, 0, h - 40, 0.40), (w, 0, w - 30, 0, 0.30)):   # body shade bottom/right
            g = cairo.LinearGradient(x0, y0, x1, y1)
            g.add_color_stop_rgba(0, 0.08, 0.03, 0.01, a)
            g.add_color_stop_rgba(1, 0.08, 0.03, 0.01, 0)
            ctx.set_source(g)
            ctx.paint()
        surf.flush()
        return surf

    def make_rig(self):
        W, H, CX, YB = self.W, self.H, self.CX, self.YB
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        ctx = cairo.Context(surf)
        keep = []

        def metal(x, y, w, h, seed, axis, r=10, base=(0.56, 0.60, 0.66)):
            s, b = brushed_metal(int(w), int(h), seed, base=base, axis=axis)
            keep.append(b)
            rounded_rect(ctx, x, y, w, h, r)
            ctx.save()
            ctx.clip_preserve()
            ctx.set_source_surface(s, x, y)
            ctx.paint()
            ctx.restore()
            ctx.set_source_rgba(0.02, 0.03, 0.06, 0.9)
            ctx.set_line_width(4)
            ctx.stroke()

        for x in (40.0, 830.0):                                            # guide rails, floor to ceiling
            metal(x, -30, 70, YB + 150, 21 + int(x), "x", 8, base=(0.42, 0.46, 0.52))
            ctx.set_source_rgba(0.85, 0.92, 1.0, 0.35)
            ctx.rectangle(x + (64 if x < CX else 3), -30, 3, YB + 150)
            ctx.fill()
        a0, a1 = self.ANVIL                                                  # anvil table + pedestal
        metal(a0 + 120, YB + 60, a1 - a0 - 240, 120, 9, "x", 6, base=(0.34, 0.37, 0.43))
        metal(a0, YB, a1 - a0, 70, 11, "y", 10, base=(0.52, 0.56, 0.63))
        ctx.set_source_rgba(0.85, 0.92, 1.0, 0.75)
        ctx.rectangle(a0 + 6, YB + 1, a1 - a0 - 12, 3)
        ctx.fill()
        surf.flush()
        self._wt = brushed_metal(640, int(self.WT_H), 4, base=(0.50, 0.54, 0.60), axis="y",
                                 bands=((0, 1.35), (0.10, 1.10), (0.55, 0.85), (1, 0.42)))
        keep.append(self._wt[1])
        return surf, keep

    def _press(self, ctx, t, hc, pp, heat):
        """The drop weight: hoist cable, steel block with sleeves riding the rails, hot rim on impact."""
        CX = self.CX
        yb = self.plate_y(t, hc)
        top = yb - self.WT_H
        ctx.new_path()                                                       # cable
        for off, col, lw in ((0, (0.05, 0.06, 0.08), 9), (-1.5, (0.62, 0.66, 0.72), 3)):
            ctx.move_to(CX + off, -10)
            ctx.line_to(CX + off, top - 20)
            ctx.set_source_rgb(*col)
            ctx.set_line_width(lw)
            ctx.stroke()
        rounded_rect(ctx, CX - 34, top - 30, 68, 36, 10)                     # shackle block
        ctx.set_source_rgb(0.30, 0.33, 0.38)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.02, 0.03, 0.06, 0.95)
        ctx.set_line_width(3)
        ctx.stroke()
        for x0 in (110.0, CX + 320):                                         # sleeves on the rails
            rounded_rect(ctx, x0 - 6, top + 34, 48, 70, 6)
            ctx.set_source_rgb(0.24, 0.27, 0.32)
            ctx.fill_preserve()
            ctx.set_source_rgba(0.02, 0.03, 0.06, 0.95)
            ctx.stroke()
        rounded_rect(ctx, CX - 320, top, 640, self.WT_H, 14)                  # the block
        ctx.save()
        ctx.clip_preserve()
        ctx.set_source_surface(self._wt[0], CX - 320, top)
        ctx.paint()
        ctx.rectangle(CX - 290, top + 44, 580, 46)                           # recessed band
        ctx.set_source_rgba(0.05, 0.06, 0.09, 0.35)
        ctx.fill()
        ctx.rectangle(CX - 290, top + 90, 580, 3)
        ctx.set_source_rgba(0.9, 0.95, 1.0, 0.35)
        ctx.fill()
        ctx.restore()
        ctx.set_source_rgba(0.02, 0.03, 0.06, 0.95)
        ctx.set_line_width(4)
        ctx.stroke()
        for x in (CX - 260, CX + 260):                                       # bolts
            ctx.new_path()
            ctx.arc(x, top + 67, 11, 0, TAU)
            g = cairo.RadialGradient(x - 4, top + 63, 1, x, top + 67, 12)
            g.add_color_stop_rgba(0, 0.95, 0.97, 1.0, 1)
            g.add_color_stop_rgba(1, 0.20, 0.23, 0.30, 1)
            ctx.set_source(g)
            ctx.fill()
        ctx.save()                                                           # specular streak
        ctx.set_operator(cairo.OPERATOR_ADD)
        glow(ctx, CX - 150, top + 18, 60, (1.0, 1.0, 1.0), 0.35)
        ctx.restore()
        hit = self.pulse(t) if t < self.T_SLAM else math.exp(-max(0.0, t - self.T_SLAM) / 0.3)
        ctx.rectangle(CX - 312, yb - 7, 624, 5)                              # warm rim, flares on each hit
        c = crush.mix(self.COOL, self.HOT, heat)
        ctx.set_source_rgba(c[0], c[1], c[2], clamp(0.20 + 0.45 * pp + 0.5 * hit))
        ctx.fill()

    # ---------------------------------------------------------------- sound
    def audio(self):
        rng = np.random.default_rng(self.seed + 777)
        sr = au.SR
        TS = self.T_SLAM
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        bed[:n] = au.tension_bed(self.dur, TS, root=self.ROOT, seed=self.seed % 97 + 3, gain=0.50)
        tk, hk, pk = self._sched()
        # the bed holds its breath while the weight falls, then comes back on the hit (keeps every hit a clear onset)
        tsec = np.arange(n) / sr
        duck = np.ones(n)
        for tt, h in zip(tk, hk):
            tf = math.sqrt(2 * h / self.G)
            a0, a1 = tt - tf - 0.05, tt
            m = (tsec >= a0) & (tsec < a1 + 0.12)
            x = tsec[m]
            dip = np.where(x < a1, np.clip((x - a0) / 0.10, 0, 1), np.clip(1 - (x - a1) / 0.12, 0, 1))
            duck[m] = np.minimum(duck[m], 1.0 - DUCK * dip)
        bed[:n] *= duck[:, None]
        # each drop: hoist whirr up, then a steel clank on the bar, louder as the drops get higher
        for k, (tt, h, p) in enumerate(zip(tk, hk, pk)):
            slam = k == len(tk) - 1
            tf = math.sqrt(2 * h / self.G)
            tp = tk[k - 1] if k else 0.0
            lift = max(0.15, tt - tf - 0.05 - (tp + 0.14))
            if k:
                w = au.bandpass(au.glide(110 + 40 * min(p, 1), 190 + 160 * min(p, 1), lift, harm=0.5), 90, 2500)
                env = np.sin(np.pi * np.clip(np.arange(len(w)) / max(1, len(w)), 0, 1)) ** 0.8
                au.place(fx, w * env, tp + 0.14, 0.05 + 0.06 * min(p, 1))
            if slam:
                continue
            g = 0.18 + 0.32 * p
            au.place(fx, au.boom(rng, 0.45, 170.0, 80.0), tt, 0.7 * g)
            au.place(fx, au.bell(520 + 60 * hash01(k), 0.35), tt + 0.002, 0.07 + 0.10 * p, float(rng.uniform(-0.2, 0.2)))
            nc = int(0.06 * sr)
            au.place(fx, au.noise(rng, 0.06, 900, 7000)[:nc] * np.exp(-np.arange(nc) / sr / 0.015), tt, 0.25 + 0.45 * p)
        # chocolate snaps: a dry click per split (brighter for smaller pieces), soft woody knock for big ones
        slots = {}
        for nd in self.nodes:
            if math.isfinite(nd["td"]):
                slots.setdefault(round(nd["td"], 2), []).append(nd)
        for te, group in sorted(slots.items()):
            te = min(te, self.dur - 0.3)
            m = len(group)
            k = clamp((te - self.T_CONTACT) / (TS - self.T_CONTACT))
            for q, nd in enumerate(group[:7]):
                tq = te + 0.012 * q
                fb = 900.0 * max(0.03, nd["area"]) ** -0.3
                nc = int(0.04 * sr)
                click = (au.noise(rng, 0.04, 1800, 10000)[:nc] * np.minimum(1.0, np.arange(nc) / (0.0008 * sr))
                         * np.exp(-np.arange(nc) / sr / 0.008))
                pan = clamp((nd["cu"] - 0.5) * 1.6, -0.8, 0.8)
                gn = ((0.45 + 0.25 * k) if te < TS else 0.45) / (1.0 + 0.35 * (m - 1)) ** 0.5
                au.place(fx, click, tq, CLICKX * gn, pan)
                au.place(fx, au.marimba(fb * (1 + 0.18 * crush.hash1(q, te)), 0.25), tq + 0.002, 0.10 * gn / 0.6, pan)
                if nd["area"] > 0.02:
                    au.place(fx, au.boom(rng, 0.30, 200.0, 90.0), tq, 0.22 * math.sqrt(nd["area"]))
        for ts, i in self.stress:
            au.place(fx, au.tick(2400 + 800 * crush.hash1(i, 2.0), 0.05), ts, 0.08,
                     clamp(crush.hash1(i, 3.0) * 1.6 - 0.8, -0.7, 0.7))
        # the slam: boom, rumble, scatter of crumbs
        au.place(fx, au.boom(rng, 1.4, 90.0, 34.0), TS, 0.85)
        au.place(fx, au.bell(380.0, 0.9), TS + 0.003, 0.10)
        nr = np.arange(int(1.8 * sr)) / sr
        au.place(fx, au.noise(rng, 1.8, 40, 600) * np.exp(-nr / 0.7) * np.minimum(1.0, nr / 0.01), TS, 0.9)
        ns = np.arange(int(0.9 * sr)) / sr
        au.place(fx, au.noise(rng, 0.9, 2500, 12000) * np.exp(-ns / 0.22) * np.minimum(1.0, ns / 0.004), TS + 0.02, 0.35)
        nl = np.arange(int(1.2 * sr)) / sr
        au.place(fx, au.noise(rng, 1.2, 1200, 6000) * np.exp(-nl / 0.45) * np.minimum(1.0, nl / 0.05), TS + 0.15, 0.12)
        for q in range(22):
            au.place(fx, au.tick(1500 + 2500 * crush.hash1(q, 4.0), 0.04), TS + 0.3 + 1.0 * crush.hash1(q, 3.0), 0.10,
                     clamp(crush.hash1(q, 5.0) * 1.6 - 0.8, -0.8, 0.8))
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
        st = self.stats
        return dict(
            title="Each Drop Gets Higher… How Many Chocolate Pieces?",
            description=(
                "Comment your number \U0001F447\n"
                "A steel weight drops on one chocolate bar again and again, a little higher every time, so every hit "
                "lands harder. A piece breaks when a hit beats its strength.\n"
                "Chocolate is thinnest along its moulded grooves, so the first cracks run along them. After that "
                "smaller pieces are stronger, so each new crack needs a harder hit.\n"
                "The last drop falls from twice the height. The counter shows the real count the whole time.\n"
                "No footage, samples or stock media.\n"
                "#simulation #satisfying #physics"),
            tags=["simulation", "chocolate", "chocolate bar", "drop test", "steel weight", "how many pieces",
                  "satisfying", "physics", "crush", "shatter", "made with code", "oddly satisfying"],
            pinned_comment="Comment your number \U0001F447 How many pieces did the chocolate bar break into?",
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of a steel weight and a chocolate bar, "
                                              "generated by a toy fracture model; not realistic footage of people, "
                                              "places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / self.FPS, 2),
            answer=int(self.n_final),
            drama=st,
        )


DUCK = 0.50        # bed dip while the weight falls
CLICKX = 0.6       # snap click level


def hash01(k):
    return crush.hash1(k, 9.7)


def score_seed(seed):
    return crush.drama(simulate(seed, Scene), Scene)
