"""CRUSH COUNT: a hydraulic press crushes one glass cube. Crushed into how many pieces? (lane 3, destruction count)

A fracture-cascade model, rules all on screen from frame 0:
  * the press force climbs on a gauge (0 up to the "press max"); the cube breaks at its own strength (shown in tons);
  * a piece breaks in two when the force passes its strength; strength = cube strength / sqrt(size), so small
    pieces are stronger: a piece a quarter of the size needs twice the force;
  * each cut falls along a random line, so pieces differ in size and each has its own small flaw;
  * at the end the press SLAMS to twice the press max (shown on the gauge); every piece that is weaker than the slam
    breaks, in generations, and the piece counter rolls up to the answer. Double the force gives about 4x the pieces;
  * pick the band (A, B, C, D) that holds the final piece count; a band is crossed out the moment the count passes its top edge.
Deterministic per seed, so make.py can score many seeds without rendering and render only the best one.
Original object (a plain glass cube), original look; not a copy of any channel's video.
"""
import bisect
import math
import os

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_in, ease_out, ease_out_back, glow, hex_rgb, lerp, lighten, mix,
                        polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
T_TOTAL = 25.0                 # seconds; frame 0 is the clean loop point (intact cube)
T_COL = T_TOTAL - 1.5          # the slam: the answer exists only in the last 1.5 s
HX = 475.0
CX = 475.0
WC, HC_ = 520.0, 400.0         # block size in px
YB = 1100.0                    # cube bottom (anvil top)
QEXP = 0.55                    # force ramp F = Fs * (t / T_COL) ** QEXP
SLAM_D = 0.42                  # seconds the force takes to reach the slam value
GEN_DELAY = 0.06               # a new piece cannot break again before this
EDGES = (260, 330, 400)        # band edges: A < 260, B 260-329, C 330-399, D 400+
BANDS = ("A", "B", "C", "D")
T_FIRST_MAX = 2.0              # first crack must be on screen by this time (rule is 2.5 s)
NMAX = 900
SIGMA = 0.22                   # spread of flaws between sibling pieces


# ====================================================================== fracture model
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
    th = 0.5 * math.atan2(2 * sxy, sxx - syy)           # major axis
    mx, my = math.cos(th), math.sin(th)
    proj = [x * mx + y * my for x, y in p]
    ext = max(proj) - min(proj)
    best = None
    for attempt in range(10):
        ang = th + rng.gauss(0, 0.45 + 0.12 * attempt)  # cut direction is mostly across the major axis
        nx, ny = math.cos(ang), math.sin(ang)           # normal of the cut line
        off = rng.uniform(-0.22, 0.22) * ext * (1 + 0.15 * attempt)
        d = nx * (cx + mx * off) + ny * (cy + my * off)
        a, ha = clip(p, nx, ny, d, True)
        b, hb = clip(p, nx, ny, d, False)
        if len(a) < 3 or len(b) < 3:
            continue
        aa, ab = poly_ca(a)[2], poly_ca(b)[2]
        if min(aa, ab) >= 0.2 * A:
            best = (a, b, ha)
            break
        if best is None:
            best = (a, b, ha)
    a, b, hits = best
    seg = (hits[0][0], hits[0][1], hits[-1][0], hits[-1][1]) if len(hits) >= 2 else (cx, cy, cx, cy)
    return a, b, seg


def force_params(seed):
    r = np.random.default_rng(seed * 7919 + 13)
    K = float(round(r.uniform(22, 28)))
    fslam = float(10 * round(r.uniform(34, 46)))
    return K, fslam, fslam / 2.0


def force_at(t, fs, fslam):
    if t <= T_COL:
        return fs * (max(0.0, t) / T_COL) ** QEXP
    x = clamp((t - T_COL) / SLAM_D)
    return fs + (fslam - fs) * (1 - (1 - x) ** 2)


def time_for(s, fs, fslam):
    """First time the force reaches s (inf if never)."""
    if s <= 0:
        return 0.0
    if s <= fs:
        return T_COL * (s / fs) ** (1.0 / QEXP)
    if s > fslam:
        return math.inf
    x = 1 - math.sqrt(max(0.0, 1 - (s - fs) / (fslam - fs)))
    return T_COL + x * SLAM_D


def simulate(seed):
    import random
    rng = random.Random(seed * 1000003 + 17)
    K, fslam, fs = force_params(seed)
    root = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    nodes = []

    def add(poly, tb, par, depth, seg=None):
        cx, cy, A = poly_ca(poly)
        if par < 0:
            s = K
        else:                                          # a piece inherits its parent's flaw; smaller is stronger
            pn = nodes[par]
            s = max(1.10 * pn["s"], pn["s"] * math.sqrt(pn["area"] / A) * math.exp(rng.gauss(0, SIGMA)))
        nodes.append(dict(poly=poly, cu=cx, cv=cy, area=A, s=s, tb=tb, td=math.inf, par=par, depth=depth, cut=seg))
        return len(nodes) - 1

    add(root, 0.0, -1, 0)
    i = 0
    while i < len(nodes) and len(nodes) < NMAX:
        nd = nodes[i]
        tbk = time_for(nd["s"], fs, fslam)
        if math.isfinite(tbk):
            tbk = max(tbk, nd["tb"] + (GEN_DELAY if nd["par"] >= 0 else 0.0))
            a, b, seg = split_poly(nd["poly"], rng)
            nd["td"] = tbk
            ia = add(a, tbk, i, nd["depth"] + 1, seg)
            add(b, tbk, i, nd["depth"] + 1, None)
        i += 1
    return dict(seed=seed, K=K, fslam=fslam, fs=fs, nodes=nodes)


def split_times(sim):
    return sorted(n["td"] for n in sim["nodes"] if math.isfinite(n["td"]))


def count_at(times, t):
    return 1 + bisect.bisect_right(times, t)


def band_of(n):
    return sum(1 for e in EDGES if n >= e)


def runner_up(n):
    b = band_of(n)
    lo = n - EDGES[b - 1] if b > 0 else 1e9
    hi = EDGES[b] - n if b < len(EDGES) else 1e9
    return BANDS[b - 1] if lo < hi else BANDS[b + 1]


def drama(sim):
    stats = dict(seed=sim["seed"], valid=False, score=-100.0)
    times = split_times(sim)
    if not times or len(sim["nodes"]) >= NMAX:
        stats["note"] = "no cracks or too many pieces"
        return stats
    n_final = 1 + len(times)
    pre = [t for t in times if t < T_COL]
    n_pre = 1 + len(pre)
    t1 = times[0]
    gaps = [b - a for a, b in zip([0.0] + pre[:-1], pre)]
    seq = [t1] + [t for t in pre if t > t1]
    bigg = max([b - a for a, b in zip(seq[:-1], seq[1:])] + [0.0])
    last = times[-1]
    edge_d = min(abs(n_final - e + 0.5) for e in EDGES)
    # events per second in thirds of the build: want rising
    th = [sum(1 for t in pre if k * T_COL / 3 <= t < (k + 1) * T_COL / 3) for k in range(3)]
    gates = dict(
        first_crack=bool(t1 <= T_FIRST_MAX),
        count_range=bool(220 <= n_final <= 540),
        pre_count=bool(45 <= n_pre <= 135),
        slam_multiplier=bool(n_final >= 2.6 * n_pre),
        steady_events=bool(bigg <= 1.2),
        rising=bool(th[2] > th[0]),
        cascade_done=bool(last <= T_COL + 0.65),
        near_edge=bool(edge_d <= 0.04 * n_final),
        not_first_band=bool(band_of(n_final) >= 1),
        no_pre_edge=bool(n_pre < EDGES[0] - 1),
    )
    ok = all(gates.values())
    score = (100.0 if ok else 0.0) + 8.0 * sum(gates.values())
    score += -2.0 * edge_d / max(1.0, 0.01 * n_final) + 4.0 * min(1.0, th[2] / max(1, th[0]) / 2)
    score += -10.0 * bigg
    stats = dict(seed=sim["seed"], score=round(float(score), 2), valid=bool(ok),
                 gates={k: bool(v) for k, v in gates.items()},
                 winner=BANDS[band_of(n_final)], answer_pieces=int(n_final), runner_up=runner_up(n_final),
                 nearest_edge=round(float(edge_d), 1), pieces_before_slam=int(n_pre),
                 first_crack_s=round(float(t1), 2), longest_gap_s=round(float(bigg), 2),
                 block_strength_t=sim["K"], press_max_t=sim["fs"], slam_t=sim["fslam"],
                 cascade_done_s=round(float(last - T_COL), 2), cracks_by_third=th)
    return stats


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
BAND_COL = [hex_rgb("#FF6B57"), hex_rgb("#FFD23F"), hex_rgb("#4CC9FF"), hex_rgb("#B38CFF")]      # circle, square, triangle, diamond
INK = hex_rgb("#070A14")
CREAM = hex_rgb("#F6F0E4")
MUTED = hex_rgb("#8E9BB5")
AMBER = hex_rgb("#FFC23D")
GLASS = ((0.50, 0.90, 0.98), (0.55, 0.74, 1.00), (0.74, 0.70, 1.00))
HOT = (1.0, 0.80, 0.48)
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
Q1, Q2 = "THE PRESS CRUSHES ONE BLOCK", "INTO HOW MANY PIECES?"
RULE = "PICK THE BAND FOR THE FINAL COUNT"
SUBTXT = "SMALL PIECES ARE STRONGER   •   SLAM = 2X THE PRESS MAX"
END_TEXT = "Got it right? +1 streak. Subscribe to keep it."
WHY = "DOUBLE THE FORCE, ABOUT 4X THE PIECES"
BAND_TXT = (f"UNDER {EDGES[0]}", f"{EDGES[0]} TO {EDGES[1] - 1}", f"{EDGES[1]} TO {EDGES[2] - 1}", f"{EDGES[2]} AND UP")
YBANDS = 1384.0


def hash1(a, b=0.0):
    v = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - math.floor(v)


def symbol(ctx, k, x, y, s):
    if k == 0:
        ctx.new_sub_path()
        ctx.arc(x, y, s, 0, TAU)
    elif k == 1:
        ctx.rectangle(x - s * 0.86, y - s * 0.86, s * 1.72, s * 1.72)
    elif k == 2:
        polygon(ctx, [(x, y - s * 1.15), (x + s * 1.1, y + s * 0.8), (x - s * 1.1, y + s * 0.8)])
    else:
        polygon(ctx, [(x, y - s * 1.2), (x + s * 0.95, y), (x, y + s * 1.2), (x - s * 0.95, y)])


def chip(ctx, font, txt, cx, y, size, pad=20, h=None, bg=CREAM, fg=INK):
    tw = font.width(txt, size, 1.2)
    h = h or size * 1.42
    rounded_rect(ctx, cx - tw / 2 - pad, y - h * 0.76, tw + 2 * pad, h, 12)
    ctx.set_source_rgba(*bg, 1)
    ctx.fill()
    font.draw(ctx, txt, cx, y, size, fg, anchor="c", tracking=1.2)


def make_background():
    rng = np.random.default_rng(11)
    yy = np.arange(H, dtype=np.float32)
    stops = ((0, (0.020, 0.040, 0.075)), (560, (0.045, 0.090, 0.150)), (900, (0.075, 0.150, 0.220)),
             (1180, (0.055, 0.105, 0.165)), (1920, (0.015, 0.025, 0.050)))
    img = np.zeros((H, 1, 3), np.float32)
    for c in range(3):
        img[:, 0, c] = np.interp(yy, [a for a, _ in stops], [b[c] for _, b in stops])
    img = np.repeat(img, W, axis=1)
    img += rng.normal(0, 0.004, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    del img
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    g = cairo.RadialGradient(CX, 900, 0, CX, 900, 520)
    g.add_color_stop_rgba(0, 0.20, 0.55, 0.70, 0.30)
    g.add_color_stop_rgba(1, 0.20, 0.55, 0.70, 0.0)
    ctx.set_source(g)
    ctx.rectangle(0, 560, W, 700)
    ctx.fill()
    ctx.set_source_rgba(0.55, 0.85, 1.0, 0.05)                # a faint grid on the test bay wall
    ctx.set_line_width(2)
    for x in range(-40, W, 90):
        ctx.move_to(x, 580)
        ctx.line_to(x, 1180)
    for y in range(600, 1180, 90):
        ctx.move_to(0, y)
        ctx.line_to(W, y)
    ctx.stroke()
    return surf, buf


def make_header(font):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, 460)
    ctx = cairo.Context(surf)
    s1 = min(100.0, 900.0 / font.width(Q2, 1.0), 900.0 / font.width(Q1, 1.0))
    font.draw(ctx, Q1, HX, 124, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    font.draw(ctx, Q2, HX, 124 + s1 * 1.1, s1, AMBER, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    chip(ctx, font, RULE, HX, 124 + s1 * 1.1 + 92, 38, pad=22, bg=AMBER)
    font.draw(ctx, SUBTXT, HX, 124 + s1 * 1.1 + 146, 25, CREAM, anchor="c", stroke=INK, stroke_w=6, tracking=0.6)
    surf.flush()
    return surf


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((50, -6.0), (120, -2.5), (300, -0.5), (900, 0.0), (3000, 1.5), (8000, 1.0), (16000, -4.0))
    DRIVE = 3.0
    CEILING_DB = -3.4
    HEAD_Y = 150

    def __init__(self, seed):
        self.seed = seed
        sim = simulate(seed)
        self.stats = drama(sim)
        if not np.isfinite([n["td"] for n in sim["nodes"]][0]):
            raise ValueError(f"seed {seed}: the block never cracks")
        self.K, self.fslam, self.fs = sim["K"], sim["fslam"], sim["fs"]
        nodes = sim["nodes"]
        self.nn = len(nodes)
        self.nodes = nodes
        self.times = split_times(sim)
        self.n_final = 1 + len(self.times)
        self.band = band_of(self.n_final)
        self.nframes = int(round(T_TOTAL * FPS))
        self.dur = self.nframes / FPS
        self.tb = np.array([n["tb"] for n in nodes])
        self.td = np.array([n["td"] if math.isfinite(n["td"]) else 1e9 for n in nodes])
        self.sa = np.array([n["s"] for n in nodes])
        self.pts = [np.array(n["poly"], float) for n in nodes]
        self.cuv = np.array([[n["cu"], n["cv"]] for n in nodes])
        self.shr = np.array([clamp(1.0 - 3.2 / max(10.0, math.sqrt(n["area"]) * WC), 0.55, 0.96) for n in nodes])
        self.kdir = np.zeros((self.nn, 2))
        for i, n in enumerate(nodes):
            if n["par"] >= 0:
                d = self.cuv[i] - self.cuv[n["par"]]
                self.kdir[i] = d / (np.hypot(*d) + 1e-9)
        self.cuts = [(n["tb"], n["cut"], i) for i, n in enumerate(nodes) if n["cut"] is not None]
        self.edge_t = [None] * 3                          # time the count passes each band's top edge
        for k, e in enumerate(EDGES):
            if self.n_final >= e:
                self.edge_t[k] = self.times[e - 2]
        self.t_done = self.times[-1]
        self.stress = []                                   # a big piece about to break glints (fills the gaps between cracks)
        for i, n in enumerate(nodes):
            if n["area"] >= 0.012:
                ts = time_for(0.86 * n["s"], self.fs, self.fslam)
                if n["tb"] + 0.12 < ts < min(n["td"] - 0.08, T_COL):
                    self.stress.append((ts, i))
        self._res = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            font = Font(FONT_PATH)
            bg, keep = make_background()
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep, font=font,
                             head=make_header(font))
        return self._res

    # ------------------------------------------------------------ geometry over time
    def force(self, t):
        return force_at(t, self.fs, self.fslam)

    def geom(self, t):
        F = self.force(t)
        pp = min(1.0, F / self.fs)
        sl = ease_in(clamp((t - T_COL) / 0.22), 2) if t > T_COL else 0.0
        hc = (1.0 - 0.06 * pp) * (1 - sl) + 0.20 * sl
        spread = 1.0 + 0.03 * pp + 0.22 * sl
        if t > T_COL + 0.22:
            hc += 0.014 * math.exp(-(t - T_COL - 0.22) / 0.25) * math.sin(26 * (t - T_COL - 0.22))
        return F, pp, hc, spread, sl

    def tx(self, u, v, hc, spread):
        return CX + (u - 0.5) * WC * spread, YB - (1.0 - v) * HC_ * hc

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        t = f / FPS
        a = ramp(t - T_COL, 1.1, 1.44)
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

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, res):
        t = f / FPS
        d = t - T_COL
        font = res["font"]
        F, pp, hc, spread, sl = self.geom(t)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        # ---- world, with camera push-in and a shake that grows with the force
        ctx.save()
        ctx.rectangle(0, 572, W, 640)
        ctx.clip()
        k = clamp(t / T_COL) if t < T_COL else 1.0
        zoom = 1.0 + 0.17 * k ** 1.5 + 0.08 * (1.0 - ease_out(t / 0.4, 2)) + (0.035 * ease_out(clamp(d / 0.4)) if d > 0 else 0.0)
        amp = 1.6 + 9.0 * pp ** 3 + (26.0 * math.exp(-d / 0.28) if d > 0 else 0.0)
        sx = amp * (hash1(f, 1.7) - 0.5) * 2
        sy = amp * (hash1(f, 4.1) - 0.5) * 2
        ctx.translate(CX + sx, 900 + sy)
        ctx.scale(zoom, zoom)
        ctx.translate(-CX, -900)
        self._glow(ctx, t, pp, sl)
        self._anvil(ctx)
        self._pieces(ctx, f, t, F, hc, spread)
        self._glints(ctx, t, hc, spread)
        self._chips(ctx, t)
        self._press(ctx, t, F, hc, pp)
        ctx.restore()
        if d > 0:                                              # slam flash
            ctx.save()
            ctx.set_operator(cairo.OPERATOR_ADD)
            glow(ctx, CX, 930, 620, (1.0, 0.95, 0.85), 0.55 * math.exp(-d / 0.10))
            ctx.restore()
        ctx.set_source_surface(res["head"], 0, self.HEAD_Y)
        ctx.paint()
        self._labels(ctx, font, t)
        self._counter(ctx, font, t)
        self._gauge(ctx, font, t, F, d)
        self._bands(ctx, font, t, d)
        self._countdown(ctx, font, t)
        if d >= 0:
            self._result(ctx, font, t, d)
        self._endcard(ctx, font, t, d)

    def _glow(self, ctx, t, pp, sl):
        a = 0.10 + 0.50 * pp ** 2.2 + 0.4 * sl
        ctx.save()
        ctx.set_operator(cairo.OPERATOR_ADD)
        glow(ctx, CX, 900, 360 + 160 * pp, (0.25, 0.75, 1.0), a * 0.55)
        glow(ctx, CX, 900, 260, (1.0, 0.72, 0.30), a * 0.35 * pp)
        ctx.restore()

    def _anvil(self, ctx):
        g = cairo.LinearGradient(0, YB, 0, YB + 70)
        g.add_color_stop_rgba(0, 0.52, 0.58, 0.68, 1)
        g.add_color_stop_rgba(1, 0.16, 0.19, 0.27, 1)
        rounded_rect(ctx, 125, YB, 700, 62, 10)
        ctx.set_source(g)
        ctx.fill()
        ctx.set_source_rgba(0.80, 0.88, 1.0, 0.7)
        ctx.rectangle(130, YB, 690, 4)
        ctx.fill()

    def _press(self, ctx, t, F, hc, pp):
        yb = YB - HC_ * hc - 60.0 * (1.0 - ease_out(t / 0.4, 2))   # the plate lands on the block in the first 0.4 s
        top = yb - 62
        # piston rod from the cylinder
        g = cairo.LinearGradient(CX - 36, 0, CX + 36, 0)
        g.add_color_stop_rgba(0, 0.28, 0.32, 0.42, 1)
        g.add_color_stop_rgba(0.35, 0.85, 0.90, 1.0, 1)
        g.add_color_stop_rgba(1, 0.22, 0.26, 0.36, 1)
        ctx.rectangle(CX - 36, 650, 72, max(4.0, top - 650 + 6))
        ctx.set_source(g)
        ctx.fill()
        # cylinder body, glowing at its seam with the pressure
        g = cairo.LinearGradient(0, 596, 0, 662)
        g.add_color_stop_rgba(0, 0.45, 0.50, 0.62, 1)
        g.add_color_stop_rgba(1, 0.17, 0.20, 0.30, 1)
        rounded_rect(ctx, CX - 100, 596, 200, 66, 12)
        ctx.set_source(g)
        ctx.fill()
        ctx.rectangle(CX - 100, 652, 200, 5)
        ctx.set_source_rgba(1.0, 0.55 + 0.3 * (1 - pp), 0.25, 0.35 + 0.6 * pp)
        ctx.fill()
        # plate
        g = cairo.LinearGradient(0, top, 0, yb)
        g.add_color_stop_rgba(0, 0.70, 0.76, 0.88, 1)
        g.add_color_stop_rgba(1, 0.22, 0.26, 0.36, 1)
        rounded_rect(ctx, 150, top, 650, 62, 12)
        ctx.set_source(g)
        ctx.fill()
        ctx.set_source_rgba(0.95, 0.98, 1.0, 0.8)
        ctx.rectangle(155, top + 2, 640, 3)
        ctx.fill()

    def _pieces(self, ctx, f, t, F, hc, spread):
        live = np.nonzero((self.tb <= t) & (self.td > t))[0]
        order = live[np.argsort(self.sa[live])]
        for i in order:
            sgl = self.sa[i]
            r = min(1.0, F / sgl)
            age = t - self.tb[i]
            c = self.cuv[i]
            p = c + (self.pts[i] - c) * self.shr[i]
            X = CX + (p[:, 0] - 0.5) * WC * spread
            Y = YB - (1.0 - p[:, 1]) * HC_ * hc
            ox = oy = 0.0
            if age < 0.45 and self.kdir[i].any():
                kk = 7.0 * math.exp(-age / 0.10)
                ox += self.kdir[i][0] * kk
                oy += self.kdir[i][1] * kk * 0.7
            sv = 2.4 * r ** 5
            if sv > 0.05:
                ox += (hash1(i, f * 0.31) - 0.5) * 2 * sv
                oy += (hash1(i, f * 0.57 + 3) - 0.5) * 2 * sv
            hop = 0.0
            if t > T_COL:
                q = clamp((t - T_COL) / 0.5)
                hop = -90.0 * hash1(i, 5.0) * 4 * q * (1 - q) * (1.0 if hash1(i, 2.0) > 0.4 else 0.0)
            X = X + ox
            Y = Y + oy + hop
            base = mix(GLASS[int(hash1(i, 9.0) * 3) % 3], GLASS[int(hash1(i, 6.0) * 3) % 3], 0.5)
            heat = 0.80 * r ** 3
            col = mix(base, HOT, heat)
            ctx.new_path()
            ctx.move_to(X[0], Y[0])
            for j in range(1, len(X)):
                ctx.line_to(X[j], Y[j])
            ctx.close_path()
            gx0, gy0, gx1, gy1 = X.min(), Y.min(), X.max(), Y.max()
            g = cairo.LinearGradient(gx0, gy0, gx1 + 1, gy1 + 1)
            g.add_color_stop_rgba(0, *lighten(col, 0.35), 0.92)
            g.add_color_stop_rgba(1, *darken(col, 0.35), 0.74)
            ctx.set_source(g)
            ctx.fill_preserve()
            ctx.set_source_rgba(0.93, 0.99, 1.0, 0.50 + 0.4 * heat)
            ctx.set_line_width(1.8)
            ctx.stroke()
        if t < T_COL:                                          # a light sweep runs over the glass
            hw = WC * spread / 2
            ph = (t * 0.8) % 1.0
            xs = CX - hw - 110 + ph * (2 * hw + 220)
            ctx.save()
            ctx.rectangle(CX - hw, YB - HC_ * hc, 2 * hw, HC_ * hc)
            ctx.clip()
            ctx.set_operator(cairo.OPERATOR_ADD)
            g = cairo.LinearGradient(xs - 60, 0, xs + 60, 0)
            g.add_color_stop_rgba(0, 1, 1, 1, 0)
            g.add_color_stop_rgba(0.5, 0.8, 0.95, 1.0, 0.20)
            g.add_color_stop_rgba(1, 1, 1, 1, 0)
            ctx.set_source(g)
            ctx.rectangle(xs - 60, YB - HC_ * hc, 120, HC_ * hc)
            ctx.fill()
            ctx.restore()
        # crack flashes: the new cut glows white, then fades
        for tb, seg, i in self.cuts:
            age = t - tb
            if 0 <= age < 0.40:
                x0, y0 = self.tx(seg[0], seg[1], hc, spread)
                x1, y1 = self.tx(seg[2], seg[3], hc, spread)
                a = math.exp(-age / 0.11)
                ctx.new_path()
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
                ctx.set_source_rgba(1.0, 0.82, 0.45, 0.55 * a)
                ctx.set_line_width(11)
                ctx.stroke_preserve()
                ctx.set_source_rgba(1.0, 1.0, 1.0, 0.95 * a)
                ctx.set_line_width(3.5)
                ctx.stroke()

    def _glints(self, ctx, t, hc, spread):
        for ts, i in self.stress:
            age = t - ts
            if 0 <= age < 0.30:
                x, y = self.tx(self.cuv[i][0], self.cuv[i][1], hc, spread)
                u = age / 0.30
                ctx.new_path()
                ctx.arc(x, y, 10 + 30 * ease_out(u), 0, TAU)
                ctx.set_source_rgba(1.0, 0.93, 0.7, 0.85 * (1 - u))
                ctx.set_line_width(3.5)
                ctx.stroke()
                ctx.new_path()
                ctx.arc(x, y, 5 * (1 - u) + 1, 0, TAU)
                ctx.set_source_rgba(1.0, 1.0, 1.0, 0.9 * (1 - u))
                ctx.fill()

    def _chips(self, ctx, t):
        for tb, seg, i in self.cuts:
            age = t - tb
            if age < 0 or age > 0.9:
                continue
            slam = tb >= T_COL
            n = 2 if slam else 3
            for j in range(n):
                h1, h2, h3 = hash1(i, j + 1.0), hash1(i, j + 7.0), hash1(i, j + 13.0)
                tt = tb
                _, _, hc, spread, _ = self.geom(tt)
                u = lerp(seg[0], seg[2], h3)
                v = lerp(seg[1], seg[3], h3)
                x0, y0 = self.tx(u, v, hc, spread)
                ang = TAU * h1
                sp = (90 + 330 * h2) * (1.3 if slam else 1.0)
                x = x0 + math.cos(ang) * sp * age
                y = y0 + math.sin(ang) * sp * age - 90 * age + 1300 * age * age * 0.5
                a = 1 - ramp(age, 0.35, 0.9)
                s = 2.6 + 4.0 * h3
                ctx.save()
                ctx.translate(x, y)
                ctx.rotate(TAU * h2 + 6 * age)
                ctx.rectangle(-s, -s * 0.6, 2 * s, 1.2 * s)
                ctx.set_source_rgba(0.85, 0.97, 1.0, 0.9 * a)
                ctx.fill()
                ctx.restore()

    # ------------------------------------------------------------ overlay
    def _labels(self, ctx, font, t):
        # popups on the first few cracks: they make the first seconds readable
        for n, te in enumerate(self.times[:5]):
            dt = t - te
            if 0 <= dt < 0.8:
                seg = self.cuts_for(te)
                if seg is None:
                    continue
                a = 1 - ramp(dt, 0.45, 0.8)
                x = min(790.0, max(160.0, CX + (seg[0] - 0.5) * WC))
                y = 690 - 18 * ease_out(min(1.0, dt / 0.35))
                font.draw(ctx, f"CRACK #{n + 1}", x, y, 42, AMBER, anchor="c", alpha=a, stroke=INK, stroke_w=8)

    def cuts_for(self, te):
        for tb, seg, i in self.cuts:
            if abs(tb - te) < 1e-9:
                return seg
        return None

    def count(self, t):
        return count_at(self.times, t)

    def _counter(self, ctx, font, t):
        n = self.count(t)
        a = ramp(t, 2.8, 3.1)
        if a <= 0.01:
            return
        txt = str(n)
        sz = 112.0
        w1 = font.width(txt, sz, 1.0)
        w2 = font.width("PIECES", 50.0, 1.0)
        x0 = HX - (w1 + 22 + w2) / 2
        y = 1250.0
        pop = 1.0
        last = [te for te in self.times if te <= t]
        if last:
            age = t - last[-1]
            if age < 0.3:
                pop = 1.0 + 0.08 * math.exp(-age / 0.09)
        ctx.save()
        ctx.translate(HX, y - 40)
        ctx.scale(pop, pop)
        ctx.translate(-HX, -(y - 40))
        font.draw(ctx, txt, x0 + w1, y, sz, AMBER, anchor="r", alpha=a, stroke=INK, stroke_w=12, shadow=5)
        font.draw(ctx, "PIECES", x0 + w1 + 22, y, 50, CREAM, anchor="l", alpha=a, stroke=INK, stroke_w=8)
        ctx.restore()

    def _gauge(self, ctx, font, t, F, d):
        a = 1.0 - ramp(d, 0.35, 0.55) if d >= 0 else 1.0
        if a <= 0.01:
            return
        x0, x1, y = 70.0, 880.0, 1298.0
        bw = x1 - x0
        font.draw(ctx, f"PRESS {int(F)} T", x0, y - 12, 38, CREAM, anchor="l", alpha=a, stroke=INK, stroke_w=7, tracking=0.6)
        rounded_rect(ctx, x0, y, bw, 22, 8)
        ctx.set_source_rgba(0.03, 0.05, 0.10, 0.9 * a)
        ctx.fill()
        fr = clamp(F / self.fslam)
        rounded_rect(ctx, x0 + 2, y + 2, max(6.0, (bw - 4) * fr), 18, 6)
        gg = cairo.LinearGradient(x0, 0, x1, 0)
        gg.add_color_stop_rgba(0, 0.25, 0.80, 0.95, a)
        gg.add_color_stop_rgba(0.5, 1.0, 0.80, 0.25, a)
        gg.add_color_stop_rgba(1, 1.0, 0.30, 0.20, a)
        ctx.set_source(gg)
        ctx.fill()
        xm = x0 + bw * 0.5                                     # press max tick
        ctx.rectangle(xm - 2, y - 6, 4, 34)
        ctx.set_source_rgba(*CREAM, a)
        ctx.fill()
        font.draw(ctx, f"PRESS MAX {int(self.fs)} T", xm + 14, y - 12, 26, CREAM, anchor="l", alpha=a, stroke=INK,
                  stroke_w=6, tracking=0.5)
        font.draw(ctx, f"BLOCK BREAKS AT {int(self.K)} T", x0, y + 52, 26, MUTED, anchor="l",
                  alpha=a, stroke=INK, stroke_w=6, tracking=0.5)
        font.draw(ctx, f"SLAM {int(self.fslam)} T", x1, y + 54, 32, AMBER, anchor="r", alpha=a, stroke=INK, stroke_w=6,
                  tracking=0.5)

    def _bands(self, ctx, font, t, d):
        font.draw(ctx, "HOW MANY PIECES AT THE END?", HX, YBANDS - 8, 27, CREAM, anchor="c", stroke=INK, stroke_w=6,
                  tracking=1.5)
        cw, gap, x0 = 207.0, 12.0, 38.0
        yc = YBANDS + 52
        for i in range(4):
            x = x0 + i * (cw + gap)
            te = self.edge_t[i] if i < 3 else None
            out = te is not None and t >= te
            done = t >= self.t_done + 0.05 and d >= 0
            win = done and self.band == i
            lose = done and self.band != i
            col = BAND_COL[i]
            a = 0.38 if (out or lose) else 1.0
            sc, fl = 1.0, 0.0
            if te is not None and 0 <= t - te < 0.5:
                fl = 1 - (t - te) / 0.5
            if win:
                sc = 1.0 + 0.05 * ease_out_back(clamp((t - self.t_done - 0.05) / 0.3))
            ctx.save()
            ctx.translate(x + cw / 2, yc)
            ctx.scale(sc, sc)
            rounded_rect(ctx, -cw / 2, -44, cw, 88, 16)
            ctx.set_source_rgba(0.03, 0.05, 0.11, 0.9)
            ctx.fill_preserve()
            ctx.set_source_rgba(*mix(col, (1, 1, 1), 0.8 * fl + (0.35 if win else 0.0)), a)
            ctx.set_line_width(4.5 + (4 if win else 0) + 3 * fl)
            ctx.stroke()
            ctx.new_path()
            symbol(ctx, i, -48, -9, 16)
            ctx.set_source_rgba(*col, a)
            ctx.fill()
            font.draw(ctx, BANDS[i], 10, 6, 58, col, anchor="c", alpha=a, stroke=INK, stroke_w=7)
            font.draw(ctx, BAND_TXT[i], 0, 35, 28, CREAM, anchor="c", alpha=a, tracking=0.8)
            if out and not win:
                ctx.new_path()
                ctx.move_to(-cw / 2 + 12, 36)
                ctx.line_to(cw / 2 - 12, -36)
                ctx.set_source_rgba(1.0, 0.25, 0.2, 0.95)
                ctx.set_line_width(7)
                ctx.stroke()
            ctx.restore()

    def _countdown(self, ctx, font, t):
        if t >= 3.0:
            return
        a = 1.0 - ramp(t, 2.75, 3.0)
        cur = 3 - int(t)
        u = t - int(t)
        if a <= 0.01:
            return
        y = 1252
        rounded_rect(ctx, HX - 400, y - 126, 800, 150, 24)
        ctx.set_source_rgba(0.03, 0.05, 0.12, 0.90 * a)
        ctx.fill()
        ctx.set_source_rgba(*AMBER, a)
        ctx.set_line_width(4)
        ctx.stroke()
        font.draw(ctx, "PICK NOW", HX - 190, y - 40, 74, CREAM, anchor="c", alpha=a, stroke=INK, stroke_w=8)
        for j, v in enumerate((3, 2, 1)):
            on = v == cur
            sc = 1.0 + (0.28 * (1 - u) ** 2 if on else 0.0)
            ctx.save()
            ctx.translate(HX + 120 + j * 82, y - 40)
            ctx.scale(sc, sc)
            font.draw(ctx, str(v), 0, 0, 84, AMBER if on else MUTED, anchor="c", alpha=a * (1.0 if on else 0.55),
                      stroke=INK, stroke_w=8)
            ctx.restore()
        font.draw(ctx, "COMMENT YOUR PICK", HX, y + 8, 36, AMBER, anchor="c", alpha=a, stroke=INK, stroke_w=7,
                  tracking=1.2)

    def _result(self, ctx, font, t, d):
        t0 = self.t_done - T_COL + 0.02
        k1 = ramp(d, t0, t0 + 0.2)
        a = k1 * (1 - ramp(d, 1.02, 1.1))
        if a <= 0.01:
            return
        k = self.band
        rounded_rect(ctx, HX - 410, 640, 820, 330, 26)
        ctx.set_source_rgba(0.03, 0.05, 0.11, 0.95 * a)
        ctx.fill()
        s = 0.85 + 0.15 * ease_out_back(k1)
        ctx.save()
        ctx.translate(HX, 0)
        ctx.scale(s, s)
        font.draw(ctx, "CRUSHED INTO", 0, 725 / s, 52, CREAM, anchor="c", alpha=a, stroke=INK, stroke_w=8, tracking=1.5)
        font.draw(ctx, f"{self.n_final} PIECES", 0, 850 / s, 138, AMBER, anchor="c", alpha=a, stroke=INK, stroke_w=13,
                  shadow=5)
        ctx.restore()
        tx = f"BAND {BANDS[k]} WINS"
        tw = font.width(tx, 68)
        ctx.new_path()
        symbol(ctx, k, HX - tw / 2 - 40, 920 - 22, 24)
        ctx.set_source_rgba(*BAND_COL[k], a)
        ctx.fill()
        font.draw(ctx, tx, HX + 20, 940, 68, BAND_COL[k], anchor="c", alpha=a, stroke=INK, stroke_w=9)

    def _endcard(self, ctx, font, t, d):
        if d < 0:
            return
        t0 = self.t_done - T_COL + 0.1
        a = ramp(d, t0, t0 + 0.2) * (1 - ramp(d, 1.04, 1.12))
        if a <= 0.01:
            return
        size = min(36.0, 860.0 / font.width(WHY, 1.0))
        chip(ctx, font, WHY, HX, 1322, size, pad=20, h=size * 1.5, bg=AMBER)
        size = min(38.0, 880.0 / font.width(END_TEXT, 1.0))
        chip(ctx, font, END_TEXT, HX, 1524, size, pad=22, h=size * 1.5)

    # ------------------------------------------------------------ sound
    def audio(self):
        rng = np.random.default_rng(self.seed + 777)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        tsec = np.arange(n) / sr

        def tt(dd):
            return np.arange(int(dd * sr)) / sr

        bed = np.zeros((n + tail, 2))
        bed[:n] = au.tension_bed(self.dur, T_COL, root=45, seed=self.seed % 97 + 3, gain=0.55)
        # hydraulic hum: pitch and level follow the press force
        grid = tsec[::240]
        fu = np.interp(tsec, grid, np.array([self.force(x) / self.fslam for x in grid]))
        f0 = 62.0 + 70.0 * fu
        ph = np.cumsum(f0) / sr
        hum = (np.sin(TAU * ph) + 0.5 * np.sin(TAU * 2 * ph + 0.7) + 0.25 * np.sin(TAU * 3.02 * ph)) * (0.25 + 0.75 * fu) * 0.05
        hum *= np.where(tsec < T_COL, 1.0, np.exp(-(tsec - T_COL) / 0.5))
        bed[:n, 0] += hum
        bed[:n, 1] += np.roll(hum, 180)
        # countdown 3-2-1
        for j, m_ in enumerate((60, 64, 72)):
            au.place(fx, au.marimba(au.hz(m_), 0.5), float(j), 0.28)
        # one crack per split: click plus a glassy ping, higher for smaller pieces
        slots = {}
        for nd in self.nodes:
            if np.isfinite(nd["td"]) and nd["td"] < 1e8:
                slots.setdefault(round(nd["td"], 2), []).append(nd)
        for te, group in sorted(slots.items()):
            te = min(te, self.dur - 0.3)
            m = len(group)
            for q, nd in enumerate(group[:7]):
                tq = te + 0.012 * q
                fb = 1250.0 * max(0.03, nd["area"]) ** -0.3
                nc = int(0.05 * sr)
                click = (au.noise(rng, 0.05, 2500, 12000)[:nc] * np.minimum(1.0, np.arange(nc) / (0.001 * sr))
                         * np.exp(-np.arange(nc) / sr / 0.012))
                pan = clamp((nd["cu"] - 0.5) * 1.6, -0.8, 0.8)
                gn = (0.60 if te < T_COL else 0.45) / (1.0 + 0.35 * (m - 1)) ** 0.5
                au.place(fx, click, tq, gn, pan)
                au.place(fx, au.bell(fb * (1 + 0.18 * hash1(q, te)), 0.5), tq + 0.003, 0.15 * gn / 0.6, pan)
        for ts, i in self.stress:                           # a high tink when a big piece glints
            au.place(fx, au.tick(2600 + 900 * hash1(i, 2.0), 0.05), ts, 0.10, clamp(hash1(i, 3.0) * 1.6 - 0.8, -0.7, 0.7))
        # a band is crossed out: a falling two-note chime
        for i, te in enumerate(self.edge_t):
            if te is None:
                continue
            for j, m_ in enumerate((79 - 3 * i, 72 - 3 * i)):
                au.place(fx, au.bell(au.hz(m_), 0.9), te + 0.07 * j, 0.18)
        # the slam: boom, rumble, falling shards
        au.place(fx, au.boom(rng, 1.4, 90.0, 34.0), T_COL, 0.85)
        nr = tt(1.8)
        au.place(fx, au.noise(rng, 1.8, 40, 600) * np.exp(-nr / 0.7) * np.minimum(1.0, nr / 0.01), T_COL, 0.9)
        ns = tt(0.9)
        au.place(fx, au.noise(rng, 0.9, 3000, 14000) * np.exp(-ns / 0.25) * np.minimum(1.0, ns / 0.004), T_COL + 0.02, 0.35)
        for q in range(16):
            au.place(fx, au.pluck(1200 + 2600 * hash1(q, 4.0), 0.4), T_COL + 0.25 + 0.7 * hash1(q, 3.0), 0.10,
                     clamp(hash1(q, 5.0) * 1.6 - 0.8, -0.8, 0.8))
        # the answer: a held tone, then a quick rising arpeggio as the block redraws
        tp = tt(1.0)
        pad = ((np.sin(TAU * au.hz(67) * tp) + 0.4 * np.sin(TAU * au.hz(79) * tp))
               * np.minimum(1.0, tp / 0.05) * np.exp(-tp / 0.7))
        au.place(fx, pad, self.t_done + 0.1, 0.12)
        for j, m_ in enumerate((60, 64, 67, 72)):
            au.place(fx, au.marimba(au.hz(m_), 0.4), T_TOTAL - 0.62 + 0.13 * j, 0.12)
        mix_ = au.reverb(fx, rng, decay=1.0, wet=0.10) + bed
        out_ = mix_[:n].copy()
        out_[:tail] += mix_[n:]
        out_ = au.eq(out_, self.EQ)
        out_[:int(0.04 * sr)] *= np.linspace(0, 1, int(0.04 * sr))[:, None]
        out_[-int(0.25 * sr):] *= np.linspace(1, 0, int(0.25 * sr))[:, None]
        pk = np.abs(out_).max()
        return np.tanh(out_ * (self.DRIVE / max(1e-9, pk)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        st = self.stats
        title = "One Press, One Glass Block. How Many Pieces? \U0001F4A5"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "Comment A, B, C or D \U0001F447 How many pieces did the press crush this glass block into? "
                "Pick before the force starts to climb!\n\n"
                f"A glass block sits under a hydraulic press and the force climbs on a gauge. The whole block breaks at "
                f"{int(self.K)} tons. A piece breaks in two when the force passes its strength, and the strength of a piece "
                "is the block's strength divided by the square root of its size, so a piece a quarter of the size needs twice "
                "the force. Every cut falls along a random line and every piece has its own small flaw. The press climbs to "
                f"{int(self.fs)} tons, then slams to {int(self.fslam)} tons, twice that, and every piece that is weaker than "
                "the slam breaks, generation after generation. The piece counter always shows the real count. A band is "
                f"crossed out the moment the count passes its top edge: A under {EDGES[0]} pieces, B {EDGES[0]} to "
                f"{EDGES[1] - 1}, C {EDGES[1]} to {EDGES[2] - 1}, D {EDGES[2]} and up. The band that holds the final "
                "count wins. Double the force gives about four times the pieces. A simplified 2D toy model, not real "
                "glass physics.\n"
                f"Everything here is generated by code (seed {self.seed}), with no footage, samples or stock media.\n"
                "Got it right? +1 streak. Which band did you pick?\n"
                "#simulation #satisfying #crush"),
            tags=["simulation", "hydraulic press", "glass crush", "press simulation", "how many pieces", "prediction",
                  "satisfying", "made with code", "generative animation", "crush", "shatter", "glass", "physics"],
            pinned_comment=("Comment A, B, C or D \U0001F447 How many pieces does the press crush the block into? "
                            "Pick before the force climbs. A band is crossed out when the count passes it, and the "
                            "slam doubles the force."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of an abstract glass block and a press; a toy "
                                              "model that could not be mistaken for real footage, people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=st,
        )
