"""BRIDGE LOAD: how many tons does it take to break the bridge? (BRIDGE LOAD TEST and BREAKING POINT, merged.)

A deck truss of springs. Rules, all on screen from frame 0:
  * a stack of weights sits on the deck and grows one ton at a time; the load counter is always visible;
  * every beam is a spring; thin beams are weak (snap at 2.5% stretch or squeeze), medium 4.5%, thick 7%;
  * every beam is colored by how close it is to its own limit (cool teal to hot red); a beam past 100% snaps;
  * the load is added slowly, so each state is solved at rest (Newton iterations on the stretched geometry);
  * after a snap the neighbors take the load and the bridge sags; when no state can hold the weight, it falls;
  * pick the load band (A, B, C, D) that holds the collapse; a band is crossed out when the load passes its top edge.
A simplified spring model, NOT engineering advice. The simulation is deterministic per seed, so make.py can score
many seeds without rendering and render only the best one.
"""
import math
import os

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_out, ease_out_back, glow, hex_rgb, lerp, lighten, mix,
                        polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
T_TOTAL = 24.8                  # seconds; frame 0 is the clean loop point
T_COL = 23.3                    # the collapse: the answer exists only in the last 1.5 s
NP_ = 8                         # panels
PW = 110.0                      # panel width in px
BX0 = 40.0                      # x of the left support
YD = 975.0                      # deck height
DL = 0.1                        # load increment (tons)
EA0 = 1400.0                     # force per unit strain per unit of beam size
LIM = (0.0035, 0.012, 0.030, 0.045)   # snap strain: fuse, thin, medium, thick
AREA = (0.5, 1.0, 2.2, 4.0)
EDGES = (45, 60, 75)            # band edges in tons: A < 45, B 45-59, C 60-74, D 75+
BANDS = ("A", "B", "C", "D")
LMIN, LMAX = 30.0, 110.0
T_SNAP_MAX = 2.3                # the first snap must be on screen by this time


# ====================================================================== simulation
def build(seed):
    rng = np.random.default_rng(seed * 6007 + 31)
    hgt = float(rng.uniform(165, 235))
    pts = [(BX0 + i * PW, YD) for i in range(NP_ + 1)]
    pts += [(BX0 + i * PW, YD + hgt) for i in range(1, NP_)]          # nodes 9..15 = b1..b7
    bars = []                                                         # (i, j, class, kind)

    def add(i, j, cls, kind):
        bars.append((i, j, cls, kind))

    T = lambda i: i
    B = lambda i: NP_ + i                                             # b_i for i in 1..7
    kl = int(rng.integers(3, 6))
    for i in range(NP_):
        add(T(i), T(i + 1), 3 if rng.random() < 0.55 else 2, "top")
    for i in range(1, NP_ - 1):
        add(B(i), B(i + 1), 3 if rng.random() < 0.55 else 2, "bot")
    for i in range(1, NP_):
        add(T(i), B(i), 1 if (abs(i - kl) >= 2 and rng.random() < 0.3) else (3 if rng.random() < 0.35 else 2), "vert")
    add(T(0), B(1), 2, "post")
    add(B(NP_ - 1), T(NP_), 2, "post")
    n_x = int(rng.integers(5, 7))
    xp = set(rng.choice(np.arange(1, NP_ - 1), n_x, replace=False).tolist())
    for i in range(1, NP_ - 1):                                       # panel between verticals i and i+1
        main_left = i < NP_ / 2
        a, b = ((T(i), B(i + 1)) if main_left else (B(i), T(i + 1)))
        add(a, b, 2 if rng.random() < 0.65 else 3, "diag")
        if i in xp:
            a2, b2 = ((B(i), T(i + 1)) if main_left else (T(i), B(i + 1)))
            add(a2, b2, 0 if rng.random() < 0.55 else 1, "diag")
    kload = kl
    return dict(seed=seed, hgt=hgt, pts=np.array(pts, float), bars=bars, kload=kload, rng_state=None)


class Truss:
    def __init__(self, g):
        self.g = g
        self.p0 = g["pts"].copy()
        self.nn = len(self.p0)
        b = g["bars"]
        self.bi = np.array([x[0] for x in b])
        self.bj = np.array([x[1] for x in b])
        self.cls = np.array([x[2] for x in b])
        self.kind = [x[3] for x in b]
        d = self.p0[self.bj] - self.p0[self.bi]
        self.L0 = np.hypot(d[:, 0], d[:, 1])
        self.ea = EA0 * np.array(AREA)[self.cls]
        self.lim = np.array(LIM)[self.cls]
        self.free = np.ones(2 * self.nn, bool)
        self.free[0] = self.free[1] = False                           # pinned left support
        self.free[2 * NP_ + 1] = False                                # roller right support (y fixed)
        self.share = np.zeros(2 * self.nn)
        k = g["kload"]
        for off, wgt in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            self.share[2 * (k + off) + 1] = wgt
        self.alive = np.ones(len(b), bool)

    def fint_K(self, u, alive):
        p = self.p0 + u.reshape(-1, 2)
        i, j = self.bi[alive], self.bj[alive]
        d = p[j] - p[i]
        Lc = np.hypot(d[:, 0], d[:, 1])
        n = d / Lc[:, None]
        eps = (Lc - self.L0[alive]) / self.L0[alive]
        N = self.ea[alive] * eps
        f = np.zeros((self.nn, 2))
        np.add.at(f, j, n * N[:, None])
        np.add.at(f, i, -n * N[:, None])
        kk = self.ea[alive] / self.L0[alive] - N / Lc
        gg = N / Lc
        nx, ny = n[:, 0], n[:, 1]
        k11 = kk * nx * nx + gg
        k12 = kk * nx * ny
        k22 = kk * ny * ny + gg
        K = np.zeros((2 * self.nn, 2 * self.nn))
        for a, b_, s in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            for (r, c, v) in ((0, 0, k11), (0, 1, k12), (1, 0, k12), (1, 1, k22)):
                np.add.at(K, (2 * a + r, 2 * b_ + c), s * v)
        return f.ravel(), K, eps

    def solve(self, u0, load, alive):
        """Newton iterations for equilibrium at this load; returns (u, eps_all, ok)."""
        u = u0.copy()
        F = self.share * load
        fr = self.free
        for it in range(40):
            f, K, eps = self.fint_K(u, alive)
            r = (F - f)[fr]
            if np.abs(r).max() < 1e-7 * (1.0 + load):
                full = np.zeros(len(self.bi))
                full[alive] = eps
                return u, full, True
            Kf = K[np.ix_(fr, fr)] + 1e-6 * EA0 / PW * np.eye(int(fr.sum()))
            try:
                du = np.linalg.solve(Kf, r)
            except np.linalg.LinAlgError:
                return u, None, False
            m = np.abs(du).max()
            if m > 40.0:
                du *= 40.0 / m
            u[fr] += du
            if np.abs(u).max() > 500:
                return u, None, False
        return u, None, False


SAG_MAX, SLIDE_MAX = 0.40, 45.0


def too_far(u, g):
    """The bridge counts as fallen when the weight has sunk by 40% of the truss height or the roller has slid in."""
    return bool(abs(u[2 * g["kload"] + 1]) > SAG_MAX * g["hgt"] or abs(u[2 * NP_]) > SLIDE_MAX)


def simulate(seed):
    g = build(seed)
    tr = Truss(g)
    alive = np.ones(len(tr.bi), bool)
    u = np.zeros(2 * tr.nn)
    steps = []            # per increment: load, alive copy, u, eps
    events = []           # (load, bar, step index before, step index after)
    L = 0.0
    collapse = None
    steps.append((0.0, alive.copy(), u.copy(), np.zeros(len(alive))))
    k = 0
    while L < LMAX and collapse is None:
        k += 1
        L = round(k * DL, 6)
        u_new, eps, ok = tr.solve(u, L, alive)
        if not ok or too_far(u_new, g):
            collapse = (L, None, "load", u.copy(), alive.copy())
            break
        u = u_new
        guard = 0
        while True:
            ratio = np.abs(eps) / tr.lim
            ratio[~alive] = 0
            m = int(np.argmax(ratio))
            if ratio[m] < 1.0:
                break
            u_before, eps_before, alive_before = u.copy(), eps.copy(), alive.copy()
            alive[m] = False
            u2, eps2, ok = tr.solve(u, L, alive)
            if not ok or too_far(u2, g):
                alive[m] = True
                collapse = (L, m, "snap", u.copy(), alive.copy())
                break
            u, eps = u2, eps2
            events.append(dict(load=L, bar=m, step=len(steps), before=(u_before, eps_before, alive_before),
                               after=(u.copy(), eps.copy(), alive.copy())))
            guard += 1
            if guard > 12:
                break
        if collapse is not None:
            break
        steps.append((L, alive.copy(), u.copy(), eps.copy()))
    return dict(seed=seed, g=g, tr=tr, steps=steps, events=events, collapse=collapse)


def band_of(L):
    return sum(1 for e in EDGES if L >= e)


def runner_up(lc):
    b = band_of(lc)
    lo = lc - EDGES[b - 1] if b > 0 else 1e9
    hi = EDGES[b] - lc if b < len(EDGES) else 1e9
    return BANDS[b - 1] if lo < hi else BANDS[b + 1]


def warp_p(first_load, lc):
    """Time map t = T_COL * (L / Lc) ** p: the load counter races early and slows toward the collapse."""
    if first_load is None or lc <= 0:
        return 1.0
    u1 = max(1e-3, first_load / lc)
    if u1 >= 0.999:
        return 1.0
    p = math.log(2.0 / T_COL) / math.log(u1)
    return float(min(2.4, max(1.0, p)))


def load_at(t, lc, p):
    if t >= T_COL:
        return lc
    return lc * (max(0.0, t) / T_COL) ** (1.0 / p)


def time_of(L, lc, p):
    return T_COL * (L / lc) ** p


def drama(sim):
    stats = dict(seed=sim["seed"], valid=False, score=-100.0)
    col = sim["collapse"]
    if col is None:
        stats["note"] = "never collapses"
        return stats
    lc = col[0]
    ev = sim["events"]
    if not ev:
        stats["note"] = "no snap before collapse"
        return stats
    first = ev[0]["load"]
    p = warp_p(first, lc)
    t1 = time_of(first, lc, p)
    times = [time_of(e["load"], lc, p) for e in ev]
    lastsnap_t = times[-1]
    edge_d = min(abs(lc - e) for e in EDGES)
    band = band_of(lc)
    # events in the last 10 s and spread
    mid_events = len({round(t, 1) for t in times if 5.0 <= t <= 22.6})
    gates = dict(
        load_range=bool(LMIN <= lc <= LMAX),
        early_snap=bool(t1 <= T_SNAP_MAX),
        enough_snaps=bool(3 <= len(ev) <= 9),
        spread=bool(mid_events >= 2),
        late_snap=bool(any(10.0 <= t <= 22.6 for t in times)),
        near_edge=bool(edge_d <= 2.5),
        not_first_band=bool(band >= 1),
        warp_ok=bool(p <= 2.4),
        holds_after_last=bool(lc - ev[-1]["load"] >= 0.0),
    )
    ok = all(gates.values())
    score = (100.0 if ok else 0.0) + 8.0 * sum(gates.values())
    score += -4.0 * edge_d + 1.5 * min(len(ev), 7) + 2.0 * min(mid_events, 4) + 3.0 * (lc - ev[-1]["load"] >= 3.0)
    stats = dict(seed=sim["seed"], score=round(float(score), 2), valid=bool(ok),
                 gates={k: bool(v) for k, v in gates.items()},
                 winner=BANDS[band], answer_tons=round(float(lc), 1), nearest_edge_tons=round(float(edge_d), 2),
                 runner_up=runner_up(lc),
                 first_snap_s=round(float(t1), 2), snaps=len(ev), snap_times=[round(float(t), 1) for t in times],
                 warp_p=round(p, 2), last_snap_to_collapse_tons=round(float(lc - ev[-1]["load"]), 1))
    return stats


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== dynamics (the look of a snap and of the fall)
GA = 1400.0           # px/s^2: how fast things fall
OWN = 0.4             # tons of beam weight per joint (only felt when the bridge falls)
YW = 1318.0           # water surface
DT = 0.00025
REC = 240             # trajectory samples per second


def dynamics(tr, u0, alive0, load, dur, gravity=False, snaps=False):
    """Explicit mass-spring integration from displacement u0; returns (x[k], alive[k], snaps) sampled at REC Hz."""
    nn, nb = tr.nn, len(tr.bi)
    x = tr.p0 + u0.reshape(-1, 2)
    v = np.zeros_like(x)
    shy = tr.share[1::2]
    wy = load * shy + OWN
    mass = wy / GA
    Fext = np.zeros((nn, 2))
    Fext[:, 1] = load * shy + (OWN if gravity else 0.0)
    fixed = np.zeros((nn, 2), bool)
    fixed[0, :] = True
    fixed[NP_, 1] = True
    Gi = np.zeros((nb, nn))
    Gj = np.zeros((nb, nn))
    Gi[np.arange(nb), tr.bi] = 1
    Gj[np.arange(nb), tr.bj] = 1
    alive = alive0.copy()
    beta, alpha = 0.004, 5.0
    c = beta * tr.ea / tr.L0
    nsteps = int(dur / DT)
    every = int(round(1.0 / (REC * DT)))
    xs, als, evs = [], [], []
    base_y = Fext[:, 1].copy()
    for s in range(nsteps):
        if gravity:                                    # the stack slams down: dynamic overload on the failing deck
            Fext[:, 1] = base_y + load * shy * 1.6 * min(1.0, s * DT / 0.5)
        if s % every == 0:
            xs.append(x.copy())
            als.append(alive.copy())
        d = x[tr.bj] - x[tr.bi]
        Lc = np.hypot(d[:, 0], d[:, 1])
        n = d / Lc[:, None]
        eps = (Lc - tr.L0) / tr.L0
        if snaps:
            br = np.nonzero(alive & (np.abs(eps) > tr.lim))[0]
            for m in br:
                alive[m] = False
                evs.append((s * DT, int(m)))
        vr = ((v[tr.bj] - v[tr.bi]) * n).sum(1)
        Nf = np.where(alive, tr.ea * eps + c * vr, 0.0)
        Fb = n * Nf[:, None]
        F = Fext + Gi.T @ Fb - Gj.T @ Fb - alpha * mass[:, None] * v
        v += DT * F / mass[:, None]
        v[fixed] = 0.0
        sub = x[:, 1] > YW
        if gravity and sub.any():
            v[sub] *= math.exp(-9.0 * DT)
        x += DT * v
    return np.array(xs), np.array(als), evs


def interp_traj(arr, tau):
    k = tau * REC
    i = int(min(max(k, 0), len(arr) - 1))
    j = min(i + 1, len(arr) - 1)
    f = min(max(k - i, 0.0), 1.0)
    return arr[i] * (1 - f) + arr[j] * f


# ====================================================================== look
BAND_COL = [hex_rgb("#FF6B57"), hex_rgb("#FFD23F"), hex_rgb("#4CC9FF"), hex_rgb("#B38CFF")]      # circle, square, triangle, diamond
INK = hex_rgb("#0A0716")
CREAM = hex_rgb("#F6F0E4")
MUTED = hex_rgb("#9C93B5")
AMBER = hex_rgb("#FFC23D")
HEAT = ((0.0, (0.20, 0.78, 0.86)), (0.35, (0.36, 0.86, 0.56)), (0.62, (1.0, 0.86, 0.25)),
        (0.85, (1.0, 0.50, 0.16)), (1.0, (1.0, 0.16, 0.14)))
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
HX = 475.0
Q1, Q2 = "HOW MANY TONS", "BEFORE IT BREAKS?"
RULE = "PICK THE BAND WHERE THE BRIDGE FALLS"
SUBTXT = "DOTTED FUSE SNAPS FIRST   •   THICK BEAMS LAST   •   SPRING MODEL"
END_TEXT = "Got it right? +1 streak. Subscribe to keep it."
BAND_TXT = ("UNDER 45 T", "45 TO 59 T", "60 TO 74 T", "75 T AND UP")
WIDTHS = (3.6, 6.0, 9.0, 13.0)
CRATE_W, SLAB_T = 210.0, 7.0
YBANDS = 1362.0


def heat(r):
    r = clamp(r)
    for (a, ca), (b, cb) in zip(HEAT[:-1], HEAT[1:]):
        if r <= b:
            return mix(ca, cb, (r - a) / (b - a))
    return HEAT[-1][1]


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


def hash1(a, b=0.0):
    v = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - math.floor(v)


def make_background():
    rng = np.random.default_rng(7)
    yy = np.arange(H, dtype=np.float32)
    stops = ((0, (0.045, 0.030, 0.105)), (600, (0.115, 0.060, 0.205)), (1000, (0.33, 0.15, 0.30)),
             (1300, (0.45, 0.21, 0.32)), (1920, (0.05, 0.03, 0.09)))
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
    g = cairo.RadialGradient(780, 1010, 0, 780, 1010, 420)
    g.add_color_stop_rgba(0, 1.0, 0.62, 0.35, 0.28)
    g.add_color_stop_rgba(1, 1.0, 0.62, 0.35, 0.0)
    ctx.set_source(g)
    ctx.rectangle(0, 700, W, 700)
    ctx.fill()
    for lay, (base, amp, col) in enumerate(((1170, 90, (0.20, 0.10, 0.25)), (1235, 70, (0.12, 0.06, 0.17)))):
        ctx.set_source_rgba(*col, 1)
        ctx.move_to(0, 1400)
        x = 0.0
        while x <= W:
            ctx.line_to(x, base - amp * (0.5 + 0.5 * math.sin(x / (170.0 - 40 * lay) + 1.7 * lay))
                        - 24 * math.sin(x / 47.0 + lay))
            x += 18
        ctx.line_to(W, 1400)
        ctx.close_path()
        ctx.fill()
    for x0, x1 in ((-10, BX0 + 4), (BX0 + NP_ * PW - 4, W + 10)):
        ctx.rectangle(x0, YD + 8, x1 - x0, 360)
        g = cairo.LinearGradient(0, YD, 0, YD + 360)
        g.add_color_stop_rgba(0, 0.30, 0.19, 0.27, 1)
        g.add_color_stop_rgba(1, 0.10, 0.06, 0.14, 1)
        ctx.set_source(g)
        ctx.fill()
        ctx.set_source_rgba(0.55, 0.38, 0.45, 1)
        ctx.rectangle(x0, YD + 8, x1 - x0, 6)
        ctx.fill()
        for i in range(9):
            yy_ = YD + 40 + i * 36
            ctx.set_source_rgba(0.06, 0.03, 0.09, 0.45)
            ctx.set_line_width(2)
            ctx.move_to(x0, yy_ + 6 * hash1(i, x0))
            ctx.line_to(x1, yy_ - 5 * hash1(i, x1))
            ctx.stroke()
    return surf, buf


def make_header(font):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, 460)
    ctx = cairo.Context(surf)
    s1 = min(124.0, 850.0 / font.width(Q2, 1.0))
    font.draw(ctx, Q1, HX, 124, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    font.draw(ctx, Q2, HX, 124 + s1 * 1.1, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    chip(ctx, font, RULE, HX, 124 + s1 * 1.1 + 92, 40, pad=22, bg=AMBER)
    font.draw(ctx, SUBTXT, HX, 124 + s1 * 1.1 + 146, 25, CREAM, anchor="c", stroke=INK, stroke_w=6, tracking=0.6)
    surf.flush()
    return surf


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((50, -7.0), (120, -3.0), (300, -0.5), (900, 0.0), (3000, 1.5), (8000, 0.5), (16000, -4.0))
    DRIVE = 3.0
    CEILING_DB = -3.4
    HEAD_Y = 150

    def __init__(self, seed):
        self.seed = seed
        sim = simulate(seed)
        if sim["collapse"] is None or not sim["events"]:
            raise ValueError(f"seed {seed}: no collapse or no snap")
        self.stats = drama(sim)
        self.tr = tr = sim["tr"]
        self.g = sim["g"]
        self.col = col = sim["collapse"]
        self.lc = lc = col[0]
        ev = sim["events"]
        first = ev[0]["load"]
        self.p = warp_p(first, lc)
        self.nframes = int(round(T_TOTAL * FPS))
        self.dur = self.nframes / FPS
        nn, nb = tr.nn, len(tr.bi)
        self.nn, self.nb = nn, nb
        self.kl = self.g["kload"]
        steps = sim["steps"]
        Ls = np.array([x[0] for x in steps])
        groups = []                       # events that happen in the same load increment
        for e in ev:
            if groups and abs(groups[-1][-1]["load"] - e["load"]) < 1e-9:
                groups[-1].append(e)
            else:
                groups.append([e])
        trans = []
        final_group = None
        for gp in groups:
            Le = gp[0]["load"]
            te = time_of(Le, lc, self.p)
            if Le >= lc - 1e-9:
                final_group = gp
                continue
            xs, _, _ = dynamics(tr, gp[0]["before"][0], gp[-1]["after"][2], Le, 1.3)
            ua = tr.p0 + gp[-1]["after"][0].reshape(-1, 2)
            trans.append(dict(t=te, delta=xs - ua[None]))
        self.trans = trans
        u0, al0 = col[3], col[4].copy()
        if final_group:
            u0 = final_group[0]["before"][0]
            al0 = final_group[-1]["after"][2].copy()
        if col[1] is not None:
            al0[col[1]] = False
        xs_c, als_c, evs_c = dynamics(tr, u0, al0, lc, 2.0, gravity=True, snaps=True)
        self.pos = np.zeros((self.nframes, nn, 2))
        self.alive = np.zeros((self.nframes, nb), bool)
        self.Lf = np.zeros(self.nframes)
        for f in range(self.nframes):
            t = f / FPS
            if t < T_COL:
                Lf = load_at(t, lc, self.p)
                i = int(np.searchsorted(Ls, Lf, side="right")) - 1
                i = max(0, min(i, len(steps) - 1))
                st = steps[i]
                u = st[2]
                if i + 1 < len(steps) and np.array_equal(steps[i + 1][1], st[1]):
                    fr = (Lf - st[0]) / max(1e-9, steps[i + 1][0] - st[0])
                    u = st[2] * (1 - fr) + steps[i + 1][2] * fr
                pos = tr.p0 + u.reshape(-1, 2)
                for tr_ in trans:
                    tau = t - tr_["t"]
                    if 0.0 <= tau < 1.25:
                        w = 1.0 if tau < 0.7 else 0.5 + 0.5 * math.cos(math.pi * (tau - 0.7) / 0.55)
                        pos = pos + w * interp_traj(tr_["delta"], tau)
                self.pos[f] = pos
                self.alive[f] = st[1]
                self.Lf[f] = Lf
            else:
                tau = t - T_COL
                self.pos[f] = interp_traj(xs_c, tau)
                self.alive[f] = als_c[int(min(max(tau * REC, 0), len(als_c) - 1))]
                self.Lf[f] = lc
        d = self.pos[:, tr.bj] - self.pos[:, tr.bi]
        Lc_ = np.hypot(d[..., 0], d[..., 1])
        ratio = np.abs((Lc_ - tr.L0) / tr.L0) / tr.lim
        self.ratio = np.where(self.alive, ratio, 0.0)
        self.peak = np.minimum(1.0, self.ratio.max(axis=1))
        self.dead = (~self.alive).sum(axis=1)

        def mid(f, b):
            f = int(min(max(f, 0), self.nframes - 1))
            return tuple(0.5 * (self.pos[f][tr.bi[b]] + self.pos[f][tr.bj[b]]))
        self.snap_ev = []                 # dict(t, bar, xy, label)
        cum = 0
        for gp in groups:
            Le = gp[0]["load"]
            lab = Le < lc - 1e-9
            te = time_of(Le, lc, self.p) if lab else T_COL
            for e in gp:
                cum += 1
                self.snap_ev.append(dict(t=te, bar=e["bar"], xy=mid(int(round(te * FPS)) - 1, e["bar"]), label=lab, n=cum))
        if col[1] is not None:
            self.snap_ev.append(dict(t=T_COL, bar=col[1], xy=mid(int(T_COL * FPS) - 1, col[1]), label=False))
        for (tau, m) in evs_c:
            self.snap_ev.append(dict(t=T_COL + tau, bar=m, xy=mid(int((T_COL + tau) * FPS), m), label=False))
        self.snap_ev.sort(key=lambda e: e["t"])
        self.snap_t = np.full(nb, 1e9)
        for e in self.snap_ev:
            self.snap_t[e["bar"]] = min(self.snap_t[e["bar"]], e["t"])
        self.crit_bar = col[1]
        self.band = band_of(lc)
        self.edge_t = [time_of(e, lc, self.p) if e <= lc else None for e in EDGES]
        self.splash = []
        below = np.zeros(nn, bool)
        for k in range(len(xs_c)):
            y = xs_c[k][:, 1]
            for i in np.nonzero((y > YW - 30) & ~below)[0]:
                below[i] = True
                self.splash.append((T_COL + k / REC, float(xs_c[k][i, 0])))
        if not below[self.kl]:
            raise ValueError(f"seed {seed}: the bridge does not fall into the water")
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

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        t = f / FPS
        a = ramp(t - T_COL, 1.0, 1.45)
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
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self._water(ctx, t)
        self._bridge(ctx, f, t)
        self._sparks(ctx, font, t)
        self._rings(ctx, t)
        ctx.set_source_surface(res["head"], 0, self.HEAD_Y)
        ctx.paint()
        self._counter(ctx, font, f, t)
        self._bands(ctx, font, t)
        self._countdown(ctx, font, t, d)
        if d >= 0:
            self._result(ctx, d, font)
        self._endcard(ctx, font, d)

    def _water(self, ctx, t):
        y0 = YW
        ctx.new_path()
        ctx.move_to(0, y0 + 6)
        for i in range(0, W + 20, 20):
            ctx.line_to(i, y0 + 3 * math.sin(i / 38.0 + 1.6 * t) + 2 * math.sin(i / 17.0 - 2.3 * t))
        ctx.line_to(W, YBANDS - 8)
        ctx.line_to(0, YBANDS - 8)
        ctx.close_path()
        g = cairo.LinearGradient(0, y0, 0, YBANDS)
        g.add_color_stop_rgba(0, 0.16, 0.42, 0.52, 0.95)
        g.add_color_stop_rgba(1, 0.04, 0.10, 0.20, 1)
        ctx.set_source(g)
        ctx.fill()
        ctx.set_source_rgba(0.75, 0.95, 1.0, 0.35)
        ctx.set_line_width(2)
        for k in range(5):
            yy = y0 + 12 + 8 * k
            ph = 90 * k + 130 * t * (1 + 0.2 * k)
            for x in np.arange(-60 + (ph % 150), W, 150):
                ctx.move_to(x, yy)
                ctx.line_to(x + 40, yy)
        ctx.stroke()

    def _bridge(self, ctx, f, t):
        tr = self.tr
        P = self.pos[f]
        alive = self.alive[f]
        ratio = self.ratio[f]
        x0, y0 = P[0]
        ctx.new_path()
        polygon(ctx, [(x0, y0 + 6), (x0 - 28, y0 + 48), (x0 + 28, y0 + 48)])
        ctx.set_source_rgba(0.82, 0.76, 0.72, 1)
        ctx.fill()
        ctx.new_path()
        ctx.arc(BX0 + NP_ * PW, YD + 24, 16, 0, TAU)
        ctx.set_source_rgba(0.82, 0.76, 0.72, 1)
        ctx.fill()
        for b in range(len(tr.bi)):                      # deck strip over the top chord, gone where that beam is gone
            if tr.kind[b] == "top" and alive[b]:
                a, c = P[tr.bi[b]], P[tr.bj[b]]
                nx, ny = -(c[1] - a[1]), c[0] - a[0]
                nl = math.hypot(nx, ny) or 1
                nx, ny = nx / nl * 10, ny / nl * 10
                ctx.new_path()
                ctx.move_to(a[0] + nx, a[1] + ny)
                ctx.line_to(c[0] + nx, c[1] + ny)
                ctx.set_source_rgba(0.30, 0.22, 0.34, 1)
                ctx.set_line_width(9)
                ctx.stroke()
                ctx.new_path()
                ctx.move_to(a[0] + 1.35 * nx, a[1] + 1.35 * ny)
                ctx.line_to(c[0] + 1.35 * nx, c[1] + 1.35 * ny)
                ctx.set_source_rgba(0.80, 0.70, 0.55, 0.9)
                ctx.set_line_width(2)
                ctx.stroke()
        for b in np.argsort(ratio):                      # coolest first, hottest on top
            a, c = P[tr.bi[b]], P[tr.bj[b]]
            cls = int(tr.cls[b])
            if alive[b]:
                r = float(ratio[b])
                col = heat(r)
                if r > 0.72:
                    ctx.new_path()
                    ctx.move_to(*a)
                    ctx.line_to(*c)
                    ctx.set_source_rgba(*col, 0.30 * clamp((r - 0.72) / 0.28))
                    ctx.set_line_width(WIDTHS[cls] + 11)
                    ctx.stroke()
                ctx.save()
                ctx.new_path()
                ctx.move_to(*a)
                ctx.line_to(*c)
                ctx.set_source_rgba(*darken(col, 0.55), 1)
                ctx.set_line_width(WIDTHS[cls] + 3)
                if cls == 0:
                    ctx.set_dash([9, 7])
                ctx.stroke_preserve()
                ctx.set_source_rgba(*col, 1)
                ctx.set_line_width(WIDTHS[cls])
                ctx.stroke()
                ctx.restore()
            else:
                since = t - self.snap_t[b]
                fl = math.exp(-max(0.0, since) / 0.25) if since >= 0 else 0.0
                for s0, s1 in ((0.0, 0.28), (1.0, 0.72)):
                    ctx.new_path()
                    ctx.move_to(a[0] + (c[0] - a[0]) * s0, a[1] + (c[1] - a[1]) * s0)
                    ctx.line_to(a[0] + (c[0] - a[0]) * s1, a[1] + (c[1] - a[1]) * s1)
                    ctx.set_source_rgba(*mix((0.45, 0.30, 0.34), (1.0, 0.8, 0.5), fl), 0.85)
                    ctx.set_line_width(max(3.0, WIDTHS[cls] * 0.7))
                    ctx.stroke()
        for i in range(self.nn):
            ctx.new_path()
            ctx.arc(P[i][0], P[i][1], 6.5, 0, TAU)
            ctx.set_source_rgba(0.09, 0.06, 0.14, 1)
            ctx.fill_preserve()
            ctx.set_source_rgba(0.86, 0.82, 0.92, 0.9)
            ctx.set_line_width(2)
            ctx.stroke()
        if alive.any() and t < T_COL:                    # ring on the hottest beam
            b = int(np.argmax(ratio))
            r = float(ratio[b])
            if r > 0.66:
                a, c = P[tr.bi[b]], P[tr.bj[b]]
                ctx.new_path()
                ctx.arc(0.5 * (a[0] + c[0]), 0.5 * (a[1] + c[1]), 26 + 4 * math.sin(f * 0.5), 0, TAU)
                ctx.set_source_rgba(*heat(r), 0.9)
                ctx.set_line_width(3.5)
                ctx.stroke()
        self._crate(ctx, P, self.Lf[f])

    def _crate(self, ctx, P, L):
        k = self.kl
        a, c = P[k - 1], P[k + 1]
        m = 0.5 * (0.5 * (a + c) + P[k])
        ang = math.atan2(c[1] - a[1], c[0] - a[0])
        n = int(math.ceil(L / 5.0 - 1e-9))
        ctx.save()
        ctx.translate(m[0], m[1] - 20)
        ctx.rotate(ang)
        for i in range(n):
            col = AMBER if i == n - 1 else mix((0.52, 0.56, 0.68), (0.36, 0.40, 0.54), i % 2)
            ctx.rectangle(-CRATE_W / 2, -(i + 1) * SLAB_T, CRATE_W, SLAB_T - 1.4)
            ctx.set_source_rgba(*col, 1)
            ctx.fill()
        if n:
            ctx.rectangle(-CRATE_W / 2, -n * SLAB_T, CRATE_W, n * SLAB_T)
            ctx.set_source_rgba(0.05, 0.03, 0.10, 0.75)
            ctx.set_line_width(2.5)
            ctx.stroke()
        ctx.restore()

    def _sparks(self, ctx, font, t):
        for e in self.snap_ev:
            dt = t - e["t"]
            if dt < 0 or dt > 0.75:
                continue
            x, y = e["xy"]
            k = e["bar"]
            u = dt / 0.5
            if u < 1:
                ctx.new_path()
                ctx.arc(x, y, 14 + 44 * ease_out(u), 0, TAU)
                ctx.set_source_rgba(1.0, 0.86, 0.5, 0.9 * (1 - u))
                ctx.set_line_width(4)
                ctx.stroke()
                for j in range(12):
                    ang = TAU * (j + hash1(k, j)) / 12
                    sp = 150 + 260 * hash1(k, j + 40)
                    px = x + math.cos(ang) * sp * dt
                    py = y + math.sin(ang) * sp * dt + 520 * dt * dt
                    ctx.new_path()
                    ctx.move_to(px, py)
                    ctx.line_to(px - math.cos(ang) * 14, py - math.sin(ang) * 14)
                    ctx.set_source_rgba(1.0, 0.8 - 0.4 * hash1(k, j + 9), 0.3, 1 - u)
                    ctx.set_line_width(3.5)
                    ctx.stroke()
            if e["label"]:
                a = 1 - ramp(dt, 0.35, 0.7)
                font.draw(ctx, f"SNAP #{e['n']}", x, y - 52 - 26 * ease_out(min(1.0, dt / 0.4)), 40, AMBER, anchor="c", alpha=a,
                          stroke=INK, stroke_w=8)

    def _rings(self, ctx, t):
        for ts, x in self.splash:
            u = t - ts
            if 0 <= u < 1:
                for q in range(2):
                    uu = u - 0.18 * q
                    if uu <= 0:
                        continue
                    ctx.save()
                    ctx.translate(x, YW + 4)
                    ctx.scale(1.0, 0.22)
                    ctx.new_path()
                    ctx.arc(0, 0, 14 + 120 * ease_out(uu), 0, TAU)
                    ctx.restore()
                    ctx.set_source_rgba(0.85, 0.97, 1.0, 0.8 * (1 - uu))
                    ctx.set_line_width(3)
                    ctx.stroke()

    def _counter(self, ctx, font, f, t):
        L = float(self.Lf[f])
        txt = f"{L:.1f}"
        sz = 126.0
        w1 = font.width(txt, sz, 1.0)
        w2 = font.width("TONS", 52.0, 1.0)
        x0 = HX - (w1 + 22 + w2) / 2
        y = 838.0
        pop = 1.0
        k10 = int(L // 10)
        if k10 >= 1 and t < T_COL:
            age = t - time_of(10.0 * k10, self.lc, self.p)
            if 0 <= age < 0.35:
                pop = 1.0 + 0.10 * math.exp(-age / 0.11)
        ctx.save()
        ctx.translate(HX, y - 45)
        ctx.scale(pop, pop)
        ctx.translate(-HX, -(y - 45))
        font.draw(ctx, txt, x0 + w1, y, sz, AMBER, anchor="r", stroke=INK, stroke_w=12, shadow=5)
        font.draw(ctx, "TONS", x0 + w1 + 22, y, 52, CREAM, anchor="l", stroke=INK, stroke_w=8)
        ctx.restore()
        a = ramp(t, 2.8, 3.1) * (1.0 if t < T_COL else 0.0)
        if a > 0.01:
            pk = float(self.peak[f])
            gy = 712.0
            font.draw(ctx, "HOTTEST BEAM", 70, gy, 34, CREAM, anchor="l", alpha=a, stroke=INK, stroke_w=6, tracking=0.8)
            bw = 270.0
            rounded_rect(ctx, 340, gy - 26, bw, 30, 8)
            ctx.set_source_rgba(0.04, 0.02, 0.09, 0.85 * a)
            ctx.fill()
            rounded_rect(ctx, 342, gy - 24, max(8.0, (bw - 4) * pk), 26, 6)
            ctx.set_source_rgba(*heat(pk), a)
            ctx.fill()
            font.draw(ctx, f"{int(pk * 100)}%", 626, gy, 38, heat(pk), anchor="l", alpha=a, stroke=INK, stroke_w=7,
                      tracking=0.5)

    def _bands(self, ctx, font, t):
        font.draw(ctx, "WHERE DOES IT FALL?", HX, YBANDS - 22, 27, CREAM, anchor="c", stroke=INK, stroke_w=6, tracking=1.5)
        cw, gap, x0 = 207.0, 12.0, 38.0
        for i in range(4):
            x = x0 + i * (cw + gap)
            te = self.edge_t[i] if i < 3 else None
            out = te is not None and t >= te
            win = t >= T_COL + 0.05 and self.band == i
            lose = t >= T_COL + 0.05 and self.band != i
            col = BAND_COL[i]
            a = 0.38 if (out or lose) else 1.0
            sc = 1.0
            fl = 0.0
            if te is not None and 0 <= t - te < 0.5:
                fl = 1 - (t - te) / 0.5
            if win:
                sc = 1.0 + 0.05 * ease_out_back(clamp((t - T_COL - 0.05) / 0.3))
            ctx.save()
            ctx.translate(x + cw / 2, YBANDS + 48)
            ctx.scale(sc, sc)
            rounded_rect(ctx, -cw / 2, -48, cw, 96, 16)
            ctx.set_source_rgba(0.05, 0.03, 0.11, 0.9)
            ctx.fill_preserve()
            ctx.set_source_rgba(*mix(col, (1, 1, 1), 0.8 * fl + (0.35 if win else 0.0)), a)
            ctx.set_line_width(4.5 + (4 if win else 0) + 3 * fl)
            ctx.stroke()
            ctx.new_path()
            symbol(ctx, i, -48, -10, 17)
            ctx.set_source_rgba(*col, a)
            ctx.fill()
            font.draw(ctx, BANDS[i], 10, 6, 60, col, anchor="c", alpha=a, stroke=INK, stroke_w=7)
            font.draw(ctx, BAND_TXT[i], 0, 38, 29, CREAM, anchor="c", alpha=a, tracking=0.8)
            if out and not win:
                ctx.new_path()
                ctx.move_to(-cw / 2 + 12, 40)
                ctx.line_to(cw / 2 - 12, -40)
                ctx.set_source_rgba(1.0, 0.25, 0.2, 0.95)
                ctx.set_line_width(7)
                ctx.stroke()
            ctx.restore()

    def _countdown(self, ctx, font, t, d):
        if t >= 3.0:
            return
        a = 1.0 - ramp(t, 2.75, 3.0)
        cur = 3 - int(t)
        u = t - int(t)
        if a <= 0.01:
            return
        y = 705
        rounded_rect(ctx, HX - 400, y - 126, 800, 150, 24)
        ctx.set_source_rgba(0.04, 0.02, 0.10, 0.80 * a)
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

    def _result(self, ctx, d, font):
        k1 = ramp(d, 0.20, 0.42)
        a = k1 * (1 - ramp(d, 0.72, 0.98))
        if a <= 0.01:
            return
        k = self.band
        rounded_rect(ctx, HX - 410, 640, 820, 330, 26)
        ctx.set_source_rgba(0.04, 0.02, 0.10, 0.95 * a)
        ctx.fill()
        s = 0.85 + 0.15 * ease_out_back(k1)
        ctx.save()
        ctx.translate(HX, 0)
        ctx.scale(s, s)
        font.draw(ctx, "BRIDGE FELL AT", 0, 725 / s, 52, CREAM, anchor="c", alpha=a, stroke=INK, stroke_w=8, tracking=1.5)
        font.draw(ctx, f"{self.lc:.1f} TONS", 0, 850 / s, 138, AMBER, anchor="c", alpha=a, stroke=INK, stroke_w=13,
                  shadow=5)
        ctx.restore()
        tx = f"BAND {BANDS[k]} WINS"
        tw = font.width(tx, 68)
        ctx.new_path()
        symbol(ctx, k, HX - tw / 2 - 40, 920 - 22, 24)
        ctx.set_source_rgba(*BAND_COL[k], a)
        ctx.fill()
        font.draw(ctx, tx, HX + 20, 940, 68, BAND_COL[k], anchor="c", alpha=a, stroke=INK, stroke_w=9)

    def _endcard(self, ctx, font, d):
        a = ramp(d, 0.30, 0.55) * (1 - ramp(d, 0.78, 1.02))
        if a <= 0.01:
            return
        size = min(38.0, 880.0 / font.width(END_TEXT, 1.0))
        chip(ctx, font, END_TEXT, HX, 1508, size, pad=22, h=size * 1.5)

    # ------------------------------------------------------------ sound
    def audio(self):
        rng = np.random.default_rng(self.seed + 777)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        tsec = np.arange(n) / sr

        def tt(dd):
            return np.arange(int(dd * sr)) / sr

        lc, p = self.lc, self.p
        # tension drone: two detuned sines whose pitch follows the load
        grid = tsec[::240]
        u = np.interp(tsec, grid, np.array([load_at(x, lc, p) / lc for x in grid]))
        f0 = 52.0 + 58.0 * u ** 1.4
        ph1 = np.cumsum(f0) / sr
        ph2 = np.cumsum(f0 * 1.006) / sr
        drone = (np.sin(TAU * ph1) + np.sin(TAU * ph2) + 0.35 * np.sin(TAU * 2.01 * ph1)) * (0.5 + 0.5 * u) * 0.055
        drone *= np.where(tsec < T_COL, 1.0, np.exp(-(tsec - T_COL) / 0.35))
        bed[:n, 0] += drone
        bed[:n, 1] += np.roll(drone, 240)
        # countdown 3-2-1
        for j, m_ in enumerate((60, 64, 72)):
            au.place(fx, au.marimba(au.hz(m_), 0.5), float(j), 0.30)
        # a low knock that speeds up toward the collapse, so there is always a beat; a ping at every 10 tons
        tk = 3.3
        while tk < T_COL - 0.3:
            uu = clamp((tk - 3.0) / (T_COL - 3.0))
            au.place(fx, au.marimba(au.hz(57 + 7 * uu), 0.35), tk, 0.36 + 0.12 * uu)
            tk += lerp(1.05, 0.36, uu ** 1.2)
        for k in range(10, int(lc) + 1, 10):
            tk = time_of(float(k), lc, p)
            if 2.9 < tk < T_COL - 0.1:
                au.place(fx, au.marimba(au.hz(76 + (k // 10) % 3), 0.4), tk, 0.16)
        # one soft tick per ton as the counter runs
        for k in range(1, int(lc) + 1):
            tk = time_of(float(k), lc, p)
            if tk > T_COL - 0.05:
                break
            au.place(fx, au.tick(1500 + 9 * k, 0.04), tk, 0.05 + 0.04 * (k / lc),
                     clamp(((k * 37) % 100) / 60 - 0.8, -0.7, 0.7))
        # creaks: rate follows the hottest beam
        t = 0.4
        while t < T_COL - 0.1:
            pk = float(self.peak[min(self.nframes - 1, int(t * FPS))])
            gap = 1.0 / (0.7 + 9.0 * pk ** 2)
            dur = 0.10 + 0.08 * hash1(t, 5.0)
            fa = 420 + 520 * hash1(t, 7.0)
            x = au.glide(fa, fa * 0.62, dur, decay=7.0, harm=0.6)
            au.place(fx, x * (0.35 + 0.65 * pk), t, 0.09 + 0.10 * pk, clamp(hash1(t, 9.0) * 1.6 - 0.8, -0.8, 0.8))
            t += gap * (0.6 + 0.8 * hash1(t, 11.0))
        # beam snaps: a click plus a plucked-string twang pitched by the beam length
        for e in self.snap_ev:
            te = e["t"]
            fb = 520.0 * (110.0 / max(40.0, self.tr.L0[e["bar"]])) ** 0.8
            nc = int(0.04 * sr)
            click = (au.noise(rng, 0.04, 2500, 11000)[:nc] * np.minimum(1.0, np.arange(nc) / (0.0012 * sr))
                     * np.exp(-np.arange(nc) / sr / 0.01))
            pan = clamp((e["xy"][0] - 475) / 450, -0.8, 0.8)
            big = 1.0 if e["label"] else 0.55
            au.place(fx, click, te, 0.65 * big, pan)
            au.place(fx, au.pluck(fb, 0.8), te + 0.004, 0.40 * big, pan)
        # a band is crossed out: a falling two-note chime
        for i, te in enumerate(self.edge_t):
            if te is None:
                continue
            for j, m_ in enumerate((79 - 3 * i, 72 - 3 * i)):
                au.place(fx, au.bell(au.hz(m_), 0.9), te + 0.07 * j, 0.20)
        # collapse: boom, rumble, scattered pings, water
        au.place(fx, au.boom(rng, 1.4, 90.0, 34.0), T_COL, 0.85)
        nr = tt(2.0)
        au.place(fx, au.noise(rng, 2.0, 40, 420) * np.exp(-nr / 0.8) * np.minimum(1.0, nr / 0.01), T_COL, 0.9)
        for q in range(14):
            au.place(fx, au.pluck(700 + 1900 * hash1(q, 4.0), 0.5), T_COL + 0.05 + 0.55 * hash1(q, 3.0), 0.16,
                     clamp(hash1(q, 5.0) * 1.6 - 0.8, -0.8, 0.8))
        for ts_, xx in self.splash[:6]:
            ns = tt(0.9)
            sp = au.noise(rng, 0.9, 500, 7000) * np.exp(-ns / 0.22) * np.minimum(1.0, ns / 0.01)
            au.place(fx, sp, ts_, 0.22, clamp((xx - 475) / 450, -0.8, 0.8))
        # the answer: one clean held tone, then a quick rising arpeggio as the bridge redraws
        tp = tt(1.0)
        pad = ((np.sin(TAU * au.hz(67) * tp) + 0.4 * np.sin(TAU * au.hz(79) * tp))
               * np.minimum(1.0, tp / 0.05) * np.exp(-tp / 0.7))
        au.place(fx, pad, T_COL + 0.30, 0.16)
        for j, m_ in enumerate((60, 64, 67, 72)):
            au.place(fx, au.marimba(au.hz(m_), 0.4), T_TOTAL - 0.62 + 0.13 * j, 0.14)
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
        title = "How Many Tons Before the Bridge Breaks? \U0001F309"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "How many tons before this bridge breaks: band A, B, C or D? Pick one before the load starts to climb!\n\n"
                "A truss bridge is built from springs, and a stack of weights grows on its deck, 5 tons per slab. "
                "Every beam is colored by how close it is to its own snap limit, cool teal to hot red. "
                "Beam sizes are the rules, drawn on screen: a dotted fuse beam snaps at 0.35% stretch or squeeze, a thin beam at "
                "1.2%, a medium beam at 3% and a thick beam at 4.5%. A beam past 100% of its limit snaps, and the rest take "
                "its load and the deck sags. The load is added slowly, so every state is solved at rest. The bridge counts as "
                "fallen when the weight has sunk by 40% of the truss height or the roller end has slid in, and then it drops. "
                "The load counter races at the start and slows near the end, but it always shows the real tons. "
                "A band is crossed out the moment the load passes its top edge: A under 45 tons, B 45 to 59, C 60 to 74, "
                "D 75 and up. The band that holds the collapse wins. It is a simplified spring model, not engineering advice.\n"
                f"Everything here is generated by code (seed {self.seed}), with no footage, samples or stock media.\n"
                "Which band did you pick?\n"
                "#simulation #satisfying #bridge"),
            tags=["simulation", "bridge simulation", "bridge collapse", "spring model", "which one wins", "prediction",
                  "satisfying", "made with code", "generative animation", "bridge", "load test", "physics"],
            pinned_comment=("A, B, C or D: how many tons before the bridge falls? Pick before the weights stack up. "
                            "A band is crossed out the moment the load passes its top edge, and the dotted fuse beams snap first."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of an abstract spring truss; a toy model that "
                                              "could not be mistaken for real footage, people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=st,
        )
