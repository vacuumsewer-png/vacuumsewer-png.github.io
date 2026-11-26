"""CRUSH family template: a press crushes one familiar object; the counter rolls up to the final piece count.

A scene module subclasses CrushScene, sets the look (texture, colors, question, why line) and the metadata,
and exposes score_seed(seed) for make.py's seed search:

    class Scene(CrushScene):
        def make_texture(self): ...        # the object's material, a WC x HC cairo surface (pieces sample it)
        def metadata(self): ...
    def score_seed(seed): return crush.drama(crush.simulate(seed, Scene), Scene)

Fracture model (from crushcount.py): the press force climbs from contact; a piece breaks in two when the force
passes its strength; strength = parent strength * sqrt(parent area / piece area) * a small random flaw, so small
pieces are stronger; each cut is a random line across the piece. Near the end the press SLAMS to a higher force
and every piece weaker than the slam breaks, generation after generation. The count is a real simulation result.
After the slam the pieces fly and slide with simple physics (friction on the anvil, gravity off its edges).
The last frames cross-fade back to frame 0 for a clean loop.
"""
import bisect
import math
import os
import random

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import TAU, Font, clamp, ease_in, ease_out, ease_out_back, glow, hex_rgb, lerp, mix, ramp, rounded_rect

FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
INK = hex_rgb("#060912")
CREAM = hex_rgb("#F6F0E4")
GOLD = hex_rgb("#FFC23D")


# ====================================================================== geometry
def poly_ca(p):
    a = cx = cy = 0.0
    n = len(p)
    for i in range(n):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % n]
        c = x0 * y1 - x1 * y0
        a += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    if abs(a) < 1e-12:
        return p[0][0], p[0][1], 0.0
    return cx / (3 * a), cy / (3 * a), abs(a) / 2


def clip(p, nx, ny, d, keep_pos):
    """Sutherland-Hodgman against the half-plane n.x - d >= 0 (or <= 0). Returns polygon, intersection points."""
    out, hits = [], []
    sg = 1.0 if keep_pos else -1.0
    n = len(p)
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        sa = sg * (nx * a[0] + ny * a[1] - d)
        sb = sg * (nx * b[0] + ny * b[1] - d)
        if sa >= 0:
            out.append(a)
        if (sa >= 0) != (sb >= 0):
            k = sa / (sa - sb)
            q = (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
            out.append(q)
            hits.append(q)
    return out, hits


def split_poly(p, rng):
    cx, cy, A = poly_ca(p)
    sxx = syy = sxy = 0.0
    for x, y in p:
        dx, dy = x - cx, y - cy
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    th = 0.5 * math.atan2(2 * sxy, sxx - syy)
    mx, my = math.cos(th), math.sin(th)
    proj = [x * mx + y * my for x, y in p]
    ext = max(proj) - min(proj)
    best = None
    for attempt in range(10):
        ang = th + rng.gauss(0, 0.45 + 0.12 * attempt)
        nx, ny = math.cos(ang), math.sin(ang)
        off = rng.uniform(-0.22, 0.22) * ext * (1 + 0.15 * attempt)
        d = nx * (cx + mx * off) + ny * (cy + my * off)
        a, ha = clip(p, nx, ny, d, True)
        b, hb = clip(p, nx, ny, d, False)
        if len(a) < 3 or len(b) < 3:
            continue
        if min(poly_ca(a)[2], poly_ca(b)[2]) >= 0.2 * A:
            best = (a, b, ha)
            break
        if best is None:
            best = (a, b, ha)
    a, b, hits = best
    seg = (hits[0][0], hits[0][1], hits[-1][0], hits[-1][1]) if len(hits) >= 2 else (cx, cy, cx, cy)
    return a, b, seg


def hash1(a, b=0.0):
    v = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - math.floor(v)


# ====================================================================== press schedule
def force_params(seed, C):
    r = np.random.default_rng(seed * 7919 + 13)
    K = float(round(r.uniform(*C.K_RANGE)))
    fslam = float(10 * round(r.uniform(*C.SLAM_RANGE)))
    return K, fslam, fslam / C.SLAM_X


def force_at(t, C, fs, fslam):
    if t <= C.T_CONTACT:
        return 0.0
    if t <= C.T_SLAM:
        return fs * ((t - C.T_CONTACT) / (C.T_SLAM - C.T_CONTACT)) ** C.QEXP
    x = clamp((t - C.T_SLAM) / C.SLAM_D)
    return fs + (fslam - fs) * (1 - (1 - x) ** 2)


def time_for(s, C, fs, fslam):
    """First time the force reaches s (inf if never)."""
    if s <= 0:
        return C.T_CONTACT
    if s <= fs:
        return C.T_CONTACT + (C.T_SLAM - C.T_CONTACT) * (s / fs) ** (1.0 / C.QEXP)
    if s > fslam:
        return math.inf
    x = 1 - math.sqrt(max(0.0, 1 - (s - fs) / (fslam - fs)))
    return C.T_SLAM + x * C.SLAM_D


def simulate(seed, C):
    rng = random.Random(seed * 1000003 + 17)
    K, fslam, fs = force_params(seed, C)
    root = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    nodes = []

    def add(poly, tb, par, depth, seg=None):
        cx, cy, A = poly_ca(poly)
        if par < 0:
            s = K
        else:
            pn = nodes[par]
            s = max(1.10 * pn["s"], pn["s"] * math.sqrt(pn["area"] / A) * math.exp(rng.gauss(0, C.SIGMA)))
        nodes.append(dict(poly=poly, cu=cx, cv=cy, area=A, s=s, tb=tb, td=math.inf, par=par, depth=depth, cut=seg))
        return len(nodes) - 1

    add(root, 0.0, -1, 0)
    i = 0
    while i < len(nodes) and len(nodes) < C.NMAX:
        nd = nodes[i]
        tbk = time_for(nd["s"], C, fs, fslam)
        if math.isfinite(tbk):
            tbk = max(tbk, nd["tb"] + (C.GEN_DELAY if nd["par"] >= 0 else 0.0))
            a, b, seg = split_poly(nd["poly"], rng)
            nd["td"] = tbk
            add(a, tbk, i, nd["depth"] + 1, seg)
            add(b, tbk, i, nd["depth"] + 1, None)
        i += 1
    return dict(seed=seed, K=K, fslam=fslam, fs=fs, nodes=nodes)


def split_times(sim):
    return sorted(n["td"] for n in sim["nodes"] if math.isfinite(n["td"]))


def drama(sim, C):
    """Gates and score for the seed search: an early first crack, a big final count, an escalating crack rate."""
    stats = dict(seed=sim["seed"], valid=False, score=-100.0)
    times = split_times(sim)
    if not times or len(sim["nodes"]) >= C.NMAX:
        stats["note"] = "no cracks or too many pieces"
        return stats
    TS = C.T_SLAM
    n_final = 1 + len(times)
    pre = [t for t in times if t < TS]
    n_pre = 1 + len(pre)
    t1 = times[0]
    seq = [t1] + [t for t in pre if t > t1]
    bigg = max([b - a for a, b in zip(seq[:-1], seq[1:])] + [0.0])
    last = times[-1]
    th = [sum(1 for t in pre if C.T_CONTACT + k * (TS - C.T_CONTACT) / 3 <= t < C.T_CONTACT + (k + 1) * (TS - C.T_CONTACT) / 3)
          for k in range(3)]
    lo, hi = C.COUNT_RANGE
    gates = dict(
        first_crack=bool(t1 <= C.T_FIRST_MAX),
        count_range=bool(lo <= n_final <= hi),
        pre_count=bool(40 <= n_pre <= 0.45 * n_final),
        steady_events=bool(bigg <= 1.3),
        rising=bool(th[2] >= 1.25 * th[0]),
        cascade_done=bool(last <= TS + C.CASCADE_MAX),
    )
    ok = all(gates.values())
    score = (100.0 if ok else 0.0) + 8.0 * sum(gates.values())
    score += 0.04 * min(n_final, hi) - 6.0 * t1 - 6.0 * bigg + 3.0 * min(2.0, th[2] / max(1, th[0]))
    return dict(seed=sim["seed"], score=round(float(score), 2), valid=bool(ok),
                gates={k: bool(v) for k, v in gates.items()},
                answer_pieces=int(n_final), pieces_before_slam=int(n_pre),
                first_crack_s=round(float(t1), 2), longest_gap_s=round(float(bigg), 2),
                block_strength_t=sim["K"], press_max_t=sim["fs"], slam_t=sim["fslam"],
                cascade_done_s=round(float(last - TS), 2), cracks_by_third=th)


# ====================================================================== drawing helpers
def brushed_metal(w, h, seed, base=(0.56, 0.60, 0.66), axis="x", bands=((0, 0.55), (0.18, 1.25), (0.5, 0.9), (1, 0.5))):
    """Brushed-steel surface: a shading profile across `axis` plus fine streaks along the other axis.

    Returns (surface, buffer); keep the buffer alive as long as the surface is used."""
    rng = np.random.default_rng(seed)
    n_across = w if axis == "x" else h
    u = np.linspace(0, 1, n_across)
    prof = np.interp(u, [b[0] for b in bands], [b[1] for b in bands])
    streak = rng.normal(0, 1, (h, w)).astype(np.float32)
    # streaks run along the long axis: smooth the noise along it with a running mean
    k = 41
    ax = 1 if w >= h else 0
    cs = np.cumsum(np.pad(streak, ((k, k), (0, 0)) if ax == 0 else ((0, 0), (k, k)), mode="wrap"), axis=ax)
    if ax == 0:
        sm = (cs[2 * k:] - cs[:-2 * k])[:h] / (2 * k)
    else:
        sm = (cs[:, 2 * k:] - cs[:, :-2 * k])[:, :w] / (2 * k)
    shade = prof[None, :] if axis == "x" else prof[:, None]
    lum = shade * (1.0 + 0.22 * sm + 0.03 * rng.normal(0, 1, (h, w)))
    img = np.clip(lum[..., None] * np.array(base, np.float32)[None, None, :], 0, 1)
    buf = np.zeros((h, w, 4), np.uint8)
    buf[..., :3] = (img[..., ::-1] * 255).astype(np.uint8)
    buf[..., 3] = 255
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_ARGB32, w, h, w * 4)
    return surf, buf


def text_block(font, lines, sizes, cols, width=1080, height=330, cx=470, y0=110, gap=1.08, max_w=860):
    """Pre-rendered centered headline (each line auto-fits max_w)."""
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surf)
    y = y0
    for line, size, col in zip(lines, sizes, cols):
        s = min(size, max_w / max(1e-6, font.width(line, 1.0, 0.0)))
        font.draw(ctx, line, cx, y, s, col, anchor="c", stroke=INK, stroke_w=max(8, s * 0.14), shadow=s * 0.07)
        y += s * gap
    surf.flush()
    return surf


# ====================================================================== the scene template
class CrushScene:
    W, H, FPS = 1080, 1920, 30
    # timing (seconds)
    T_TOTAL = 25.0
    T_CONTACT = 0.5            # the plate is moving from frame 0 and lands on the object here
    T_SLAM = 22.45             # the slam; the cascade it starts finishes within CASCADE_MAX
    SLAM_D = 0.40
    CASCADE_MAX = 0.65
    T_FIRST_MAX = 2.0
    LOOP_FADE = 0.40
    # model
    QEXP = 0.70                # force = fs * x ** QEXP: the crack rate rises through the build
    GEN_DELAY = 0.06
    SIGMA = 0.22
    NMAX = 900
    K_RANGE = (22, 28)         # block strength (tons)
    SLAM_RANGE = (36, 48)      # slam force / 10 (tons)
    SLAM_X = 2.0               # slam = SLAM_X * press max
    COUNT_RANGE = (260, 620)
    # layout (px)
    CX, YB, WC, HC = 470.0, 1280.0, 600.0, 490.0
    ANVIL = (110.0, 830.0)
    GAP0 = 120.0               # plate gap above the object at frame 0
    FOCUS_Y = 840.0
    COUNTER_Y = 1492.0
    PUSH = 0.11
    CTOP = 400.0               # top of the press crossbeam
    HEAD_Y = 0
    # look
    COOL = (0.30, 0.75, 1.00)
    HOT = (1.00, 0.55, 0.22)
    EDGE_LIGHT = (0.90, 0.97, 1.00)
    EDGE_DARK = (0.04, 0.12, 0.26)
    CHIP_COL = (0.85, 0.96, 1.00)
    QUESTION = ("HOW MANY PIECES", "WILL IT BREAK INTO?")
    # sound
    ROOT = 43
    EQ = ((50, -6.0), (120, -2.5), (300, -0.5), (900, 0.0), (3000, 1.5), (8000, 1.0), (16000, -4.0))
    DRIVE = 3.0
    CEILING_DB = -3.4

    # ---------------------------------------------------------------- overridables
    def make_texture(self):
        raise NotImplementedError

    def why_lines(self):
        return ("SMALL PIECES ARE STRONGER,", f"SO THE SLAM STOPS AT {self.n_final}")

    def make_background(self):
        H, W = self.H, self.W
        yy = np.arange(H, dtype=np.float32)
        stops = ((0, (0.020, 0.030, 0.055)), (600, (0.040, 0.065, 0.110)), (1050, (0.060, 0.100, 0.155)),
                 (1300, (0.030, 0.045, 0.075)), (1920, (0.012, 0.018, 0.032)))
        col = np.stack([np.interp(yy, [a for a, _ in stops], [b[c] for _, b in stops]) for c in range(3)], axis=1)
        buf = np.zeros((H, W, 4), np.uint8)
        buf[..., :3] = (np.clip(col[:, None, ::-1], 0, 1) * 255).astype(np.uint8)
        buf[..., 3] = 255
        rng = np.random.default_rng(11)
        buf[..., :3] = np.clip(buf[..., :3].astype(np.int16) + rng.integers(-2, 3, (H, W, 1)), 0, 255).astype(np.uint8)
        surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
        ctx = cairo.Context(surf)
        ctx.set_source_rgba(0.6, 0.8, 1.0, 0.035)          # faint panel seams on the wall
        ctx.set_line_width(3)
        for x in range(-20, W, 135):
            ctx.move_to(x, 380)
            ctx.line_to(x, 1300)
        ctx.stroke()
        return surf, buf

    # ---------------------------------------------------------------- setup
    def __init__(self, seed):
        C = type(self)
        self.seed = seed
        sim = simulate(seed, C)
        self.stats = drama(sim, C)
        nodes = sim["nodes"]
        if not math.isfinite(nodes[0]["td"]):
            raise ValueError(f"seed {seed}: the block never cracks")
        self.K, self.fslam, self.fs = sim["K"], sim["fslam"], sim["fs"]
        self.nodes = nodes
        self.nn = len(nodes)
        self.times = split_times(sim)
        self.n_final = 1 + len(self.times)
        self.t_done = self.times[-1]
        self.t_rev = self.t_done + 0.06                   # the answer locks in (flash) here
        self.nframes = int(round(self.T_TOTAL * self.FPS))
        self.dur = self.nframes / self.FPS
        self.tb = np.array([n["tb"] for n in nodes])
        self.td = np.array([n["td"] if math.isfinite(n["td"]) else 1e9 for n in nodes])
        self.sa = np.array([n["s"] for n in nodes])
        self.area = np.array([n["area"] for n in nodes])
        self.pts = [np.array(n["poly"], float) for n in nodes]
        self.cuv = np.array([[n["cu"], n["cv"]] for n in nodes])
        self.shr = np.array([clamp(1.0 - 3.4 / max(10.0, math.sqrt(n["area"]) * self.WC), 0.55, 0.965) for n in nodes])
        self.kdir = np.zeros((self.nn, 2))
        for i, n in enumerate(nodes):
            if n["par"] >= 0:
                d = self.cuv[i] - self.cuv[n["par"]]
                self.kdir[i] = d / (np.hypot(*d) + 1e-9)
        self.cuts = [(n["tb"], n["cut"], i) for i, n in enumerate(nodes) if n["cut"] is not None]
        # crack energy per frame: drives shake, glow and the counter pulse
        ft = np.arange(self.nframes) / self.FPS
        self.energy = np.zeros(self.nframes)
        for n in nodes:
            if math.isfinite(n["td"]):
                dt = ft - n["td"]
                m = dt >= 0
                self.energy[m] += math.sqrt(n["area"]) * np.exp(-dt[m] / 0.12)
        self.stress = []                                   # a big piece about to break glints
        for i, n in enumerate(nodes):
            if n["area"] >= 0.010:
                ts = time_for(0.86 * n["s"], C, self.fs, self.fslam)
                if n["tb"] + 0.12 < ts < min(n["td"] - 0.08, self.T_SLAM):
                    self.stress.append((ts, i))
        self._make_chips()
        self._res = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, self.W, self.H)
            font = Font(FONT_PATH)
            bg, keep = self.make_background()
            tex = self.make_texture()
            rig, keep2 = self.make_rig()
            q = text_block(font, self.QUESTION, (108, 108), (CREAM, GOLD), height=400, y0=240)
            why = text_block(font, self.why_lines(), (84, 84), (CREAM, GOLD), height=400, y0=250, gap=1.15)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=(keep, keep2), font=font,
                             tex=tex, rig=rig, q=q, why=why)
        return self._res

    # ---------------------------------------------------------------- the press rig (static parts, pre-rendered)
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

        ctop = self.CTOP
        for x in (40.0, 830.0):                                        # pillars
            metal(x, ctop + 40, 70, YB + 120 - ctop - 40, 21 + int(x), "x", 8, base=(0.42, 0.46, 0.52))
        metal(20, ctop, 900, 96, 5, "y", 14, base=(0.50, 0.54, 0.60))  # crossbeam
        metal(CX - 120, ctop + 86, 240, 96, 7, "x", 14)                # cylinder
        for x in (70, 870):                                              # bolts
            for y in (ctop + 30, ctop + 66):
                ctx.new_path()
                ctx.arc(x, y, 9, 0, TAU)
                g = cairo.RadialGradient(x - 3, y - 3, 1, x, y, 10)
                g.add_color_stop_rgba(0, 0.95, 0.97, 1.0, 1)
                g.add_color_stop_rgba(1, 0.20, 0.23, 0.30, 1)
                ctx.set_source(g)
                ctx.fill()
        a0, a1 = self.ANVIL                                              # anvil table + pedestal
        metal(a0 + 120, YB + 60, a1 - a0 - 240, 120, 9, "x", 6, base=(0.34, 0.37, 0.43))
        metal(a0, YB, a1 - a0, 70, 11, "y", 10, base=(0.52, 0.56, 0.63))
        ctx.set_source_rgba(0.85, 0.92, 1.0, 0.75)
        ctx.rectangle(a0 + 6, YB + 1, a1 - a0 - 12, 3)
        ctx.fill()
        surf.flush()
        # moving parts: rod and plate (pre-rendered once)
        self._rod = brushed_metal(84, 900, 3, base=(0.66, 0.70, 0.76), axis="x",
                                  bands=((0, 0.45), (0.3, 1.35), (0.55, 1.0), (1, 0.45)))
        self._plate = brushed_metal(700, 86, 4, base=(0.54, 0.58, 0.64), axis="y",
                                    bands=((0, 1.3), (0.12, 1.05), (0.7, 0.8), (1, 0.45)))
        keep += [self._rod[1], self._plate[1]]
        return surf, keep

    # ---------------------------------------------------------------- chips (small flying bits, ballistic)
    def _make_chips(self):
        rows = []
        for tb, seg, i in self.cuts:
            slam = tb >= self.T_SLAM
            k = clamp((tb - self.T_CONTACT) / (self.T_SLAM - self.T_CONTACT))
            n = 1 if slam else 2 + int(4 * k)
            _, _, hc, spread, _ = self.geom(tb)
            for j in range(n):
                h1, h2, h3 = hash1(i, j + 1.0), hash1(i, j + 7.0), hash1(i, j + 13.0)
                u, v = lerp(seg[0], seg[2], h3), lerp(seg[1], seg[3], h3)
                x0, y0 = self.tx(u, v, hc, spread)
                ang = -math.pi / 2 + (h1 - 0.5) * 2.6
                sp = (260 + 520 * h2) * (1.25 if slam else 0.7 + 0.6 * k)
                rows.append((tb, x0, y0, math.cos(ang) * sp, math.sin(ang) * sp, 3.0 + 6.0 * h3, TAU * h2, 2 * h1 - 1))
        self.chips = np.array(rows) if rows else np.zeros((0, 8))

    def _chip_draw(self, ctx, t):
        if not len(self.chips):
            return
        c = self.chips
        tau = t - c[:, 0]
        m = (tau >= 0) & (tau < 1.6)
        if not m.any():
            return
        c, tau = c[m], tau[m]
        g, YB = 2600.0, self.YB
        x = c[:, 1] + c[:, 3] * tau
        y = c[:, 2] + c[:, 4] * tau + 0.5 * g * tau ** 2
        # landing on the anvil: solve y(tl) = YB, then slide with friction
        disc = c[:, 4] ** 2 + 2 * g * np.maximum(0.0, YB - c[:, 2])
        tl = (-c[:, 4] + np.sqrt(disc)) / g
        xl = c[:, 1] + c[:, 3] * tl
        on = (xl > self.ANVIL[0]) & (xl < self.ANVIL[1]) & (tau > tl)
        fr = 5.0
        x = np.where(on, xl + c[:, 3] * (1 - np.exp(-fr * (tau - tl))) / fr, x)
        y = np.where(on, YB - 2.0, y)
        a = 1.0 - np.clip((tau - 1.1) / 0.5, 0, 1)
        col = self.CHIP_COL
        for k in range(len(c)):
            if y[k] > self.H + 50:
                continue
            s = c[k, 5]
            ctx.save()
            ctx.translate(x[k], y[k])
            ctx.rotate(c[k, 6] + (0.0 if on[k] else 9.0 * c[k, 7] * tau[k]))
            ctx.move_to(-s, s * 0.6)
            ctx.line_to(s * 0.2, -s)
            ctx.line_to(s, s * 0.5)
            ctx.close_path()
            ctx.set_source_rgba(col[0], col[1], col[2], 0.92 * a[k])
            ctx.fill_preserve()
            ctx.set_source_rgba(1, 1, 1, 0.6 * a[k])
            ctx.set_line_width(1.2)
            ctx.stroke()
            ctx.restore()

    # ---------------------------------------------------------------- geometry over time
    def force(self, t):
        return force_at(t, type(self), self.fs, self.fslam)

    def geom(self, t):
        F = self.force(t)
        pp = min(1.0, F / self.fs)
        d = t - self.T_SLAM
        sl = ease_in(clamp(d / 0.22), 2) if d > 0 else 0.0
        hc = (1.0 - 0.06 * pp) * (1 - sl) + 0.34 * sl
        spread = 1.0 + 0.03 * pp + 0.16 * sl
        if d > 0.22:
            hc += 0.014 * math.exp(-(d - 0.22) / 0.25) * math.sin(26 * (d - 0.22))
        return F, pp, hc, spread, sl

    def tx(self, u, v, hc, spread):
        return self.CX + (u - 0.5) * self.WC * spread, self.YB - (1.0 - v) * self.HC * hc

    def plate_y(self, t, hc):
        """Bottom edge of the press plate."""
        if t < self.T_CONTACT:
            return self.YB - self.HC - self.GAP0 * (1.0 - t / self.T_CONTACT)
        y = self.YB - self.HC * hc
        d = t - self.T_SLAM
        if d > 0.45:
            y -= 300.0 * ease_out(clamp((d - 0.45) / 0.9), 3)
        return y

    def phys(self, xc, i, tau, spread):
        """Post-slam motion of a piece whose (pre-physics) centroid is at screen x = xc: slide, hop, fall off."""
        CX, half = self.CX, self.WC / 2
        s = (xc - CX) / half
        sm = 0.5 + 0.5 * math.sin(7.0 * self.cuv[i][0] + 2.0) * math.cos(5.0 * self.cuv[i][1] + 1.0)
        v = 560.0 * s * (0.75 + 0.25 * sm)
        k, g = 2.4, 2600.0
        vy = 420.0 * abs(s) ** 1.6 * (0.6 + 0.4 * sm) * (1.0 - self.cuv[i][1] * 0.5)
        hop = min(0.0, -vy * tau + 0.5 * g * tau * tau)
        edge = self.ANVIL[1] if v >= 0 else self.ANVIL[0]
        D = edge - xc
        if (v >= 0 and D <= 0) or (v < 0 and D >= 0):
            te = 0.0
        else:
            q = D * k / v if abs(v) > 1e-6 else 2.0
            te = -math.log(1 - q) / k if q < 1 else math.inf
        if tau <= te:
            return v * (1 - math.exp(-k * tau)) / k, hop, 0.0
        ve = v * math.exp(-k * te)
        dx = v * (1 - math.exp(-k * te)) / k + ve * (tau - te)
        fall = 0.5 * g * (tau - te) ** 2
        return dx, hop + fall, math.copysign(2.6, v if v else 1.0) * (tau - te)

    # ---------------------------------------------------------------- frames
    def count(self, t):
        return 1 + bisect.bisect_right(self.times, t)

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        t = f / self.FPS
        a = ramp(t, self.T_TOTAL - self.LOOP_FADE, self.T_TOTAL - 1.0 / self.FPS)
        if a <= 0.0:
            self.draw(ctx, f, res)
        else:
            ctx.push_group()
            self.draw(ctx, f, res)
            ctx.pop_group_to_source()
            ctx.paint()
            ctx.push_group()
            self.draw(ctx, 0, res)
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(a)
        ctx.restore()
        res["surf"].flush()
        return bytes(res["surf"].get_data())

    def draw(self, ctx, f, res):
        t = f / self.FPS
        d = t - self.T_SLAM
        CX, FY = self.CX, self.FOCUS_Y
        F, pp, hc, spread, sl = self.geom(t)
        E = float(self.energy[min(f, self.nframes - 1)])
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        # ---- world: slow push-in, shake that grows with the force and each crack
        ctx.save()
        k = clamp(t / self.T_SLAM)
        zoom = 1.0 + self.PUSH * k ** 1.4 + 0.07 * (1.0 - ease_out(t / 0.9, 2)) + (0.04 * ease_out(clamp(d / 0.3)) if d > 0 else 0.0)
        amp = 2.0 + 7.0 * pp ** 3 + 14.0 * min(1.5, E) + (26.0 * math.exp(-d / 0.28) if d > 0 else 0.0)
        sx = amp * (hash1(f, 1.7) - 0.5) * 2
        sy = amp * (hash1(f, 4.1) - 0.5) * 2
        ctx.translate(CX + sx, FY + sy)
        ctx.scale(zoom, zoom)
        ctx.translate(-CX, -FY)
        heat = clamp(pp ** 1.4 + (0.3 if d > 0 else 0.0))
        lcol = mix(self.COOL, self.HOT, heat)
        ctx.save()
        ctx.set_operator(cairo.OPERATOR_ADD)
        glow(ctx, CX, self.YB - self.HC * 0.5, 430 + 140 * pp, lcol, 0.16 + 0.34 * heat + 0.25 * min(1.0, E))
        ctx.restore()
        ctx.set_source_surface(res["rig"], 0, 0)
        ctx.paint()
        self._shadow(ctx, hc, spread)
        self._pieces(ctx, f, t, F, hc, spread, res["tex"], heat)
        self._glints(ctx, t, hc, spread)
        self._chip_draw(ctx, t)
        self._press(ctx, t, hc, pp, heat)
        ctx.restore()
        # ---- flashes
        if d > 0:
            ctx.save()
            ctx.set_operator(cairo.OPERATOR_ADD)
            glow(ctx, CX, 960, 700, (1.0, 0.92, 0.80), 0.45 * math.exp(-d / 0.10))
            ctx.restore()
        r = t - self.t_rev
        if r >= 0:
            ctx.save()
            ctx.set_operator(cairo.OPERATOR_ADD)
            ctx.set_source_rgba(1.0, 0.97, 0.88, 0.55 * math.exp(-r / 0.12))
            ctx.paint()
            ctx.restore()
        # ---- overlay: one question line (the why line takes its place on the reveal) and one counter
        wa = ease_out(clamp(r / 0.25)) if r >= 0 else 0.0
        if wa < 1.0:
            ctx.set_source_surface(res["q"], 0, self.HEAD_Y)
            ctx.paint_with_alpha(1.0 - wa)
        if wa > 0.0:
            ctx.set_source_surface(res["why"], 0, self.HEAD_Y)
            ctx.paint_with_alpha(wa)
        self._counter(ctx, res["font"], t, f, E)

    def _shadow(self, ctx, hc, spread):
        w = self.WC * spread * 0.62
        ctx.save()
        ctx.translate(self.CX, self.YB + 4)
        ctx.scale(1.0, 0.09)
        g = cairo.RadialGradient(0, 0, 0, 0, 0, w)
        g.add_color_stop_rgba(0, 0, 0, 0, 0.55)
        g.add_color_stop_rgba(1, 0, 0, 0, 0)
        ctx.set_source(g)
        ctx.arc(0, 0, w, 0, TAU)
        ctx.fill()
        ctx.restore()

    def _press(self, ctx, t, hc, pp, heat):
        CX = self.CX
        yb = self.plate_y(t, hc)
        top = yb - 86
        rod, plate = self._rod[0], self._plate[0]
        r0 = self.CTOP + 182                                       # rod leaves the cylinder bottom
        ctx.save()
        ctx.rectangle(CX - 42, r0 - 10, 84, max(4.0, top - r0 + 14))
        ctx.clip()
        ctx.set_source_surface(rod, CX - 42, top - 900 + 6)
        ctx.paint()
        ctx.restore()
        ctx.rectangle(CX - 42, r0 - 10, 84, max(4.0, top - r0 + 14))
        ctx.set_source_rgba(0.02, 0.03, 0.06, 0.8)
        ctx.set_line_width(3)
        ctx.stroke()
        rounded_rect(ctx, CX - 350, top, 700, 86, 12)       # plate
        ctx.save()
        ctx.clip_preserve()
        ctx.set_source_surface(plate, CX - 350, top)
        ctx.paint()
        ctx.restore()
        ctx.set_source_rgba(0.02, 0.03, 0.06, 0.95)
        ctx.set_line_width(4)
        ctx.stroke()
        # hot rim on the plate's lower edge grows with the force
        ctx.rectangle(CX - 344, yb - 7, 688, 5)
        c = mix(self.COOL, self.HOT, heat)
        ctx.set_source_rgba(c[0], c[1], c[2], 0.25 + 0.65 * pp)
        ctx.fill()

    def _pieces(self, ctx, f, t, F, hc, spread, tex, heat):
        WC, HC, CX, YB = self.WC, self.HC, self.CX, self.YB
        ex, ey = WC * spread, HC * hc
        d = t - self.T_SLAM
        live = np.nonzero((self.tb <= t) & (self.td > t))[0]
        order = live[np.argsort(self.sa[live])]
        el, ed = self.EDGE_LIGHT, self.EDGE_DARK
        for i in order:
            r = min(1.0, F / self.sa[i])
            age = t - self.tb[i]
            c0, c1 = self.cuv[i]
            s = self.shr[i]
            bx, by = CX + (c0 - 0.5) * ex, YB - (1.0 - c1) * ey
            ox = oy = rot = 0.0
            if age < 0.45 and self.kdir[i].any():
                kk = 7.0 * math.exp(-age / 0.10)
                ox += self.kdir[i][0] * kk
                oy += self.kdir[i][1] * kk * 0.7
            sv = 2.4 * r ** 5
            if sv > 0.05:
                ox += (hash1(i, f * 0.31) - 0.5) * 2 * sv
                oy += (hash1(i, f * 0.57 + 3) - 0.5) * 2 * sv
            if d > 0:
                dx, dy, rot = self.phys(bx, i, d, spread)
                ox += dx
                oy += dy
            px, py = bx + ox, by + oy
            if py > self.H + 120:
                continue
            P = self.pts[i]
            ctx.save()
            ctx.translate(px, py)
            if rot:
                ctx.rotate(rot)
            ctx.scale(s * ex / WC, s * ey / HC)
            ctx.translate(-c0 * WC, -c1 * HC)
            ctx.set_source_surface(tex, 0, 0)
            ctx.new_path()
            ctx.move_to(P[0, 0] * WC, P[0, 1] * HC)
            for j in range(1, len(P)):
                ctx.line_to(P[j, 0] * WC, P[j, 1] * HC)
            ctx.close_path()
            ctx.fill_preserve()                               # textured fill, still in the piece's own space
            ctx.restore()                                     # the path stays (it is kept in device space)
            x0, y0, x1, y1 = ctx.path_extents()
            # facet shading: each piece catches the light a little differently
            hsh = hash1(i, 3.3)
            if hsh > 0.5:
                g = cairo.LinearGradient(x0, y0, x1, y1)
            else:
                g = cairo.LinearGradient(x1, y0, x0, y1)
            g.add_color_stop_rgba(0, 1, 1, 1, 0.06 + 0.18 * hash1(i, 8.1))
            g.add_color_stop_rgba(1, ed[0], ed[1], ed[2], 0.06 + 0.16 * hash1(i, 2.9))
            ctx.set_source(g)
            ctx.fill_preserve()
            hot = 0.45 * r ** 24
            if hot > 0.02:
                ctx.set_source_rgba(self.HOT[0], self.HOT[1], self.HOT[2], hot)
                ctx.fill_preserve()
            ctx.set_source_rgba(ed[0], ed[1], ed[2], 0.75)
            ctx.set_line_width(3.2)
            ctx.stroke_preserve()
            ctx.set_source_rgba(el[0], el[1], el[2], 0.55 + 0.4 * hot)
            ctx.set_line_width(1.3)
            ctx.stroke()
            if self.area[i] > 0.008 and (x1 - x0) > 14 and (y1 - y0) > 10:     # specular glint
                gx, gy = x0 + 0.30 * (x1 - x0), y0 + 0.28 * (y1 - y0)
                rr = 0.16 * min(x1 - x0, y1 - y0)
                ctx.save()
                ctx.set_operator(cairo.OPERATOR_ADD)
                glow(ctx, gx, gy, rr, (1.0, 1.0, 1.0), 0.55)
                ctx.restore()
        # crack flashes: each new cut glows, then fades
        for tb, seg, i in self.cuts:
            age = t - tb
            if 0 <= age < 0.40 and tb < self.T_SLAM + 0.05:
                x0, y0 = self.tx(seg[0], seg[1], hc, spread)
                x1, y1 = self.tx(seg[2], seg[3], hc, spread)
                a = math.exp(-age / 0.11)
                ctx.new_path()
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
                ctx.set_source_rgba(1.0, 0.80, 0.45, 0.55 * a)
                ctx.set_line_width(12)
                ctx.stroke_preserve()
                ctx.set_source_rgba(1.0, 1.0, 1.0, 0.95 * a)
                ctx.set_line_width(4)
                ctx.stroke()

    def _glints(self, ctx, t, hc, spread):
        for ts, i in self.stress:
            age = t - ts
            if 0 <= age < 0.30:
                x, y = self.tx(self.cuv[i][0], self.cuv[i][1], hc, spread)
                u = age / 0.30
                ctx.new_path()
                ctx.arc(x, y, 10 + 34 * ease_out(u), 0, TAU)
                ctx.set_source_rgba(1.0, 0.93, 0.7, 0.85 * (1 - u))
                ctx.set_line_width(4)
                ctx.stroke()

    def _counter(self, ctx, font, t, f, E):
        n = self.count(t)
        r = t - self.t_rev
        txt = str(n)
        sz = 150.0
        lab = "PIECE" if n == 1 else "PIECES"
        w1 = font.width(txt, sz, 1.0)
        w2 = font.width(lab, 64.0, 1.0)
        HX = self.CX
        x0 = HX - (w1 + 24 + w2) / 2
        y = self.COUNTER_Y
        pop = 1.0
        i = bisect.bisect_right(self.times, t)
        if i > 0:
            age = t - self.times[i - 1]
            if age < 0.3:
                pop += 0.10 * math.exp(-age / 0.08)
        col = CREAM
        if r >= 0:
            pop = 1.0 + 0.18 * ease_out_back(clamp(r / 0.3))
            col = GOLD
            ctx.save()
            ctx.set_operator(cairo.OPERATOR_ADD)
            glow(ctx, HX, y - 55, 380, (1.0, 0.75, 0.25), 0.35 * (0.6 + 0.4 * math.exp(-r / 0.3)))
            ctx.restore()
        ctx.save()
        ctx.translate(HX, y - 55)
        ctx.scale(pop, pop)
        ctx.translate(-HX, -(y - 55))
        font.draw(ctx, txt, x0 + w1, y, sz, GOLD if r >= 0 else col, anchor="r", stroke=INK, stroke_w=16, shadow=7, tracking=1.0)
        font.draw(ctx, lab, x0 + w1 + 24, y, 64, col, anchor="l", stroke=INK, stroke_w=10, shadow=4)
        ctx.restore()

    # ---------------------------------------------------------------- sound
    def creak(self, rng, dur, f0):
        """A groan: a slowly falling tone with a rough (stick-slip) amplitude."""
        t = np.arange(int(dur * au.SR)) / au.SR
        x = au.glide(f0, f0 * 0.82, dur, harm=0.6)
        am = 0.55 + 0.45 * np.sign(np.sin(TAU * (28 + 30 * rng.random()) * t + rng.random() * 6))
        env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
        return au.bandpass(x * am * env, 120, 3000)

    def audio(self):
        rng = np.random.default_rng(self.seed + 777)
        sr = au.SR
        TS = self.T_SLAM
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        tsec = np.arange(n) / sr
        bed = np.zeros((n + tail, 2))
        bed[:n] = au.tension_bed(self.dur, TS, root=self.ROOT, seed=self.seed % 97 + 3, gain=0.50)
        # hydraulic hum: pitch and level follow the force
        grid = tsec[::240]
        fu = np.interp(tsec, grid, np.array([self.force(x) / self.fslam for x in grid]))
        f0 = 58.0 + 80.0 * fu
        ph = np.cumsum(f0) / sr
        hum = (np.sin(TAU * ph) + 0.5 * np.sin(TAU * 2 * ph + 0.7) + 0.3 * np.sin(TAU * 3.02 * ph)) * (0.25 + 0.75 * fu) * 0.05
        hum *= np.where(tsec < TS, 1.0, np.exp(-(tsec - TS) / 0.5))
        bed[:n, 0] += hum
        bed[:n, 1] += np.roll(hum, 180)
        # the plate lands on the object
        au.place(fx, au.boom(rng, 0.5, 140.0, 60.0), self.T_CONTACT, 0.35)
        au.place(fx, au.noise(rng, 0.08, 1500, 8000) * np.exp(-np.arange(int(0.08 * sr)) / sr / 0.02), self.T_CONTACT, 0.3)
        # creaks: denser and higher as the force climbs
        t = self.T_CONTACT + 0.3
        while t < TS - 0.3:
            k = clamp((t - self.T_CONTACT) / (TS - self.T_CONTACT))
            au.place(fx, self.creak(rng, 0.35 - 0.12 * k, 90 + 260 * k + 40 * rng.random()), t, 0.10 + 0.10 * k,
                     float(rng.uniform(-0.5, 0.5)))
            t += (1.3 - 1.0 * k) * (0.7 + 0.6 * rng.random())
        # one crack per split: click plus a glassy ping, higher for smaller pieces
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
                fb = 1150.0 * max(0.03, nd["area"]) ** -0.3
                nc = int(0.05 * sr)
                click = (au.noise(rng, 0.05, 2500, 12000)[:nc] * np.minimum(1.0, np.arange(nc) / (0.001 * sr))
                         * np.exp(-np.arange(nc) / sr / 0.012))
                pan = clamp((nd["cu"] - 0.5) * 1.6, -0.8, 0.8)
                gn = ((0.45 + 0.25 * k) if te < TS else 0.45) / (1.0 + 0.35 * (m - 1)) ** 0.5
                au.place(fx, click, tq, gn, pan)
                au.place(fx, au.bell(fb * (1 + 0.18 * hash1(q, te)), 0.5), tq + 0.003, 0.15 * gn / 0.6, pan)
                if nd["area"] > 0.02:                          # a big break also thuds
                    au.place(fx, au.boom(rng, 0.35, 160.0, 70.0), tq, 0.25 * math.sqrt(nd["area"]))
        for ts, i in self.stress:
            au.place(fx, au.tick(2600 + 900 * hash1(i, 2.0), 0.05), ts, 0.10, clamp(hash1(i, 3.0) * 1.6 - 0.8, -0.7, 0.7))
        # the slam: boom, rumble, sliding and falling shards
        au.place(fx, au.boom(rng, 1.4, 90.0, 34.0), TS, 0.85)
        nr = np.arange(int(1.8 * sr)) / sr
        au.place(fx, au.noise(rng, 1.8, 40, 600) * np.exp(-nr / 0.7) * np.minimum(1.0, nr / 0.01), TS, 0.9)
        ns = np.arange(int(0.9 * sr)) / sr
        au.place(fx, au.noise(rng, 0.9, 3000, 14000) * np.exp(-ns / 0.25) * np.minimum(1.0, ns / 0.004), TS + 0.02, 0.35)
        nl = np.arange(int(1.2 * sr)) / sr                       # scrape of the pieces sliding
        au.place(fx, au.noise(rng, 1.2, 1500, 7000) * np.exp(-nl / 0.45) * np.minimum(1.0, nl / 0.05), TS + 0.15, 0.12)
        for q in range(22):
            au.place(fx, au.pluck(1300 + 2600 * hash1(q, 4.0), 0.4), TS + 0.3 + 1.0 * hash1(q, 3.0), 0.09,
                     clamp(hash1(q, 5.0) * 1.6 - 0.8, -0.8, 0.8))
        # the answer locks in: a bright bell chord
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
        pk = np.abs(out_).max()
        return np.tanh(out_ * (self.DRIVE / max(1e-9, pk)))
