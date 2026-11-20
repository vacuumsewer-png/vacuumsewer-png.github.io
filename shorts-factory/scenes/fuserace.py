"""FUSE RACE: one spark lights a hub; five tangled fuses lead to five rockets. First flame to its rocket wins.

Rules: every fuse is a smooth random spline from the hub to its own rocket, and the five lengths are within
12% of each other. Flame burns at one constant speed along arc length on every fuse, for the whole run.
Where two fuses cross (a marked junction) the flame jumps across, in both directions, and then burns both
ways along the fuse it landed on. A rocket launches the moment any flame reaches its end of the fuse; the
rest follow in arrival order. The burn is an exact shortest-path computation (Dijkstra over the fuse
network), so simulate(seed) needs no rendering: make.py scores hundreds of seeds for drama (drama()) and
renders only the best one (Scene). Near-ties come from seed selection only; nothing is edited.
"""
import heapq
import math
import os
import random

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_in, ease_in_out, ease_out, glow, hex_rgb, lerp, lighten,
                        mix, polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
DUR = 25.0
NF = int(DUR * FPS)
NFUSE = 5
NAMES = ("ORANGE", "CYAN", "MAGENTA", "GOLD", "GREEN")       # symbols: triangle, ring, cross, square, diamond
HUB = (470.0, 1215.0)
ROCKET_X = (110.0, 285.0, 460.0, 635.0, 810.0)
PAD_Y = 530.0                    # rocket base; the fuse ends just under it
END_Y = 545.0
BOX = (60.0, 900.0, 640.0, 1150.0)   # x0, x1, y0, y1 for the random waypoints
DS = 2.0                         # arc-length sample step (px)
LMIN, LMAX = 950.0, 1080.0
LEN_SPREAD = 1.12                # longest / shortest fuse
T0 = 0.30                        # burn seconds already elapsed on frame 0 (the spark is travelling)
T_WIN = 19.2                     # video time the first rocket launches (sets the one constant burn speed)
RISE = 1.0                       # flight time of the winner to its burst
BURST_GAP = 0.65                 # the bursts then follow one by one in launch order (show timing, not the result)
T_CONV = 23.3                    # embers start converging on the hub
T_RESET = 23.9                   # fresh fuses and rockets fade back in
T_BACK = 24.7
BURST_Y = (330.0, 280.0, 400.0, 300.0, 380.0)   # burst height by launch place; x drifts a little to the center


# ====================================================================== geometry
def catmull(P, per=40):
    P = np.asarray(P, float)
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    u = np.linspace(0, 1, per, endpoint=False)[:, None]
    for j in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[j - 1], Q[j], Q[j + 1], Q[j + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(P[-1][None, :])
    return np.vstack(out)


def arclen(pts):
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def resample(pts, ds=DS):
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    s = np.concatenate([np.arange(0.0, L, ds), [L]])
    return s, np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])


def seg_cross(A, B, skip_adjacent=False):
    """All proper crossings between polylines A (n x 2) and B (m x 2): list of (i, ta, j, tb, angle)."""
    a0, a1, b0, b1 = A[:-1], A[1:], B[:-1], B[1:]
    r = (a1 - a0)[:, None, :]
    s = (b1 - b0)[None, :, :]
    qp = b0[None, :, :] - a0[:, None, :]
    den = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / den
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / den
    ok = (np.abs(den) > 1e-9) & (t >= 0) & (t < 1) & (u >= 0) & (u < 1)
    if skip_adjacent:
        ii, jj = np.indices(ok.shape)
        ok &= np.abs(ii - jj) > 2
    out = []
    for i, j in zip(*np.nonzero(ok)):
        ra, sb = r[i, 0], s[0, j]
        cosang = abs(ra @ sb) / (np.hypot(*ra) * np.hypot(*sb) + 1e-9)
        out.append((int(i), float(t[i, j]), int(j), float(u[i, j]), math.degrees(math.acos(min(1.0, cosang)))))
    return out


def build_fuse(rng, k, ang, target):
    """Spline from the hub to rocket k, leaving the hub at angle ang; wiggle tuned so the length hits target."""
    hx, hy = HUB
    p1 = np.array([hx + 85 * math.cos(ang), hy + 85 * math.sin(ang)])
    q = np.array([ROCKET_X[k], END_Y + 75.0])
    e = np.array([ROCKET_X[k], END_Y])
    K = 4
    x0, x1, y0, y1 = BOX
    amp, side = rng.uniform(140, 330), rng.choice((-1.0, 1.0))
    rnd = []
    for j in range(K):
        fy = (j + rng.uniform(0.25, 0.75)) / K
        sx = lerp(p1[0], q[0], (j + 1) / (K + 1))
        rnd.append((clamp(sx + side * (-1) ** j * amp * rng.uniform(0.5, 1.0), x0, x1), lerp(y1 - 30, y0 + 30, fy)))
    rnd = np.array(rnd)
    straight = np.array([p1 + (q - p1) * (j + 1) / (K + 1) for j in range(K)])

    def curve(a):
        return catmull([HUB, p1, *(straight + a * (rnd - straight)), q, e])

    if arclen(curve(1.0)) < target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if arclen(curve(mid)) < target:
            lo = mid
        else:
            hi = mid
    return curve(0.5 * (lo + hi))


def make_layout(rng):
    for _ in range(60):
        L0 = rng.uniform(LMIN, LMAX)
        f = rng.uniform(1.0, LEN_SPREAD - 0.012, NFUSE)
        f = 1.0 + (f - f.min()) * ((LEN_SPREAD - 0.012 - 1.0) / max(1e-9, f.max() - f.min())) * rng.uniform(0.55, 1.0)
        angs = np.linspace(math.radians(-165), math.radians(-15), NFUSE)
        perm = list(range(NFUSE))
        for _s in range(int(rng.integers(1, 3))):
            q_ = int(rng.integers(0, NFUSE - 1))
            perm[q_], perm[q_ + 1] = perm[q_ + 1], perm[q_]
        fuses, ok = [], True
        for k in range(NFUSE):
            c = build_fuse(rng, k, angs[perm[k]], L0 * f[k])
            if c is None:
                ok = False
                break
            s, x, y = resample(c)
            if x.min() < 35 or x.max() > 925 or y[s > 120].max() > 1185 or y.min() < END_Y - 1:
                ok = False
                break
            fuses.append(dict(s=s, x=x, y=y, L=float(s[-1])))
        if not ok:
            continue
        coarse = []
        for fu in fuses:
            idx = np.arange(30, len(fu["s"]), 3)     # skip the shared hub point
            if idx[-1] != len(fu["s"]) - 1:
                idx = np.append(idx, len(fu["s"]) - 1)
            coarse.append((np.stack([fu["x"][idx], fu["y"][idx]], 1), fu["s"][idx]))
        if any(seg_cross(c, c, True) for c, _ in coarse):
            continue
        juncs = []
        for a in range(NFUSE):
            for b in range(a + 1, NFUSE):
                for i, ta, j, tb, angle in seg_cross(coarse[a][0], coarse[b][0]):
                    sa = lerp(coarse[a][1][i], coarse[a][1][i + 1], ta)
                    sb = lerp(coarse[b][1][j], coarse[b][1][j + 1], tb)
                    P = coarse[a][0][i] + ta * (coarse[a][0][i + 1] - coarse[a][0][i])
                    juncs.append(dict(a=a, b=b, sa=float(sa), sb=float(sb), x=float(P[0]), y=float(P[1]),
                                      angle=angle))
        bad = any(min(j["sa"], j["sb"]) < 110 or j["sa"] > fuses[j["a"]]["L"] - 90
                  or j["sb"] > fuses[j["b"]]["L"] - 90 or j["angle"] < 28 for j in juncs)
        if bad or not (3 <= len(juncs) <= 10):
            continue
        if any(math.hypot(p["x"] - q["x"], p["y"] - q["y"]) < 48 for n, p in enumerate(juncs) for q in juncs[n + 1:]):
            continue
        return fuses, juncs
    return None, None


def burn(fuses, juncs):
    """Exact arrival distance of the flame at every sample of every fuse (Dijkstra; junction hops cost 0)."""
    off = np.cumsum([0] + [len(f["s"]) for f in fuses])
    N = off[-1]
    nb = [[] for _ in range(N)]
    for i, fu in enumerate(fuses):
        s = fu["s"]
        for k in range(len(s) - 1):
            w = s[k + 1] - s[k]
            nb[off[i] + k].append((off[i] + k + 1, w))
            nb[off[i] + k + 1].append((off[i] + k, w))
    for j in juncs:
        ka = off[j["a"]] + int(round(j["sa"] / DS))
        kb = off[j["b"]] + int(round(j["sb"] / DS))
        j["ka"], j["kb"] = ka - off[j["a"]], kb - off[j["b"]]
        nb[ka].append((kb, 0.0))
        nb[kb].append((ka, 0.0))
    d = np.full(N, np.inf)
    heap = []
    for i in range(NFUSE):
        d[off[i]] = 0.0
        heap.append((0.0, int(off[i])))
    heapq.heapify(heap)
    while heap:
        dv, u = heapq.heappop(heap)
        if dv > d[u]:
            continue
        for v, w in nb[u]:
            nd = dv + w
            if nd < d[v] - 1e-9:
                d[v] = nd
                heapq.heappush(heap, (nd, v))
    return [d[off[i]:off[i + 1]].copy() for i in range(NFUSE)]


def fronts(fu, d, D):
    """Flame fronts on one fuse at burn distance D: list of (x, y, s, direction)."""
    b = d <= D
    ch = np.flatnonzero(b[:-1] != b[1:])
    out = []
    s, x, y = fu["s"], fu["x"], fu["y"]
    for k in ch:
        dd = d[k + 1] - d[k]
        u = 0.5 if abs(dd) < 1e-9 else clamp((D - d[k]) / dd)
        out.append((lerp(x[k], x[k + 1], u), lerp(y[k], y[k + 1], u), lerp(s[k], s[k + 1], u), 1 if b[k] else -1))
    return out


def remaining(fu, d, D):
    """Fuse still unburnt between the furthest-along burnt point and the rocket (0 once the rocket is lit)."""
    b = d <= D
    if b[-1]:
        return 0.0
    k = int(np.flatnonzero(b).max()) if b.any() else 0
    dd = d[k + 1] - d[k]
    u = 0.5 if abs(dd) < 1e-9 else clamp((D - d[k]) / dd)
    return fu["L"] - lerp(fu["s"][k], fu["s"][k + 1], u)


# ====================================================================== simulation
def simulate(seed, record=False):
    rng = np.random.default_rng(seed * 7919 + 13)
    fuses, juncs = make_layout(rng)
    if fuses is None:
        return dict(valid=False, reason="no layout", seed=seed)
    dist = burn(fuses, juncs)
    arr = np.array([d[-1] for d in dist])
    order = [int(i) for i in np.argsort(arr)]
    v = arr[order[0]] / (T_WIN + T0)                 # one constant burn speed, px per second
    for j in juncs:
        j["t_ign"] = min(dist[j["a"]][j["ka"]], dist[j["b"]][j["kb"]]) / v - T0
        # a real jump: the flame reached this point on one fuse first and landed on the other one here
        # (the landing side is a local minimum of arrival distance: two new fronts start from it)
        def landed(fi, k):
            d = dist[fi]
            return 0 < k < len(d) - 1 and d[k] < min(d[k - 1], d[k + 1]) - 1e-6
        j["jump"] = bool(landed(j["a"], j["ka"]) or landed(j["b"], j["kb"]))
    return dict(valid=True, seed=seed, fuses=fuses, juncs=juncs, dist=dist, arr=arr, order=order, v=v,
                t_launch=[float(a / v - T0) for a in arr])


def rem_series(sim, times):
    v = sim["v"]
    return np.array([[remaining(sim["fuses"][i], sim["dist"][i], v * (t + T0)) / v for i in range(NFUSE)]
                     for t in times])


def drama(sim):
    if not sim["valid"]:
        return dict(seed=sim["seed"], score=-1e9, valid=False)
    v, order, fuses = sim["v"], sim["order"], sim["fuses"]
    tl = sim["t_launch"]
    w, r2 = order[0], order[1]
    margin = tl[r2] - tl[w]
    spread = tl[order[-1]] - tl[w]
    times = np.arange(0.0, T_WIN, 1 / 15)
    R = rem_series(sim, times)
    lead = np.argmin(R, axis=1)
    changes, cur, since, last_change = 0, int(lead[0]), 0.0, 0.0
    cand, cand_t = cur, 0.0
    for t, l in zip(times, lead):
        if l != cand:
            cand, cand_t = int(l), t
        if cand != cur and t - cand_t >= 0.4:
            cur, changes, last_change = cand, changes + 1, float(cand_t)
    L = [f["L"] for f in fuses]
    shortest = int(np.argmin(L))
    jumps = [j for j in sim["juncs"] if j["jump"] and j["t_ign"] < T_WIN]
    k3 = int((T_WIN - 3.0) * 15)
    rem3 = sorted(R[k3])[:2]
    shortcut = sum(1 for i in range(NFUSE) for k in range(1, len(times)) if R[k - 1, i] - R[k, i] > 1.0)
    valid = 0.42 <= margin <= 1.1 and spread <= 3.6 and changes >= 1
    score = (10 + 4 * min(changes, 4) + 5 * (w != shortest) + 4 * (last_change > 9.0) + 3 * (last_change > 14.0)
             + 2 * min(shortcut, 4) - 6 * abs(margin - 0.6) - 1.0 * max(0.0, spread - 2.8)
             + 2 * (rem3[1] < 4.3) + min(len(jumps), 6) * 0.5)
    return dict(seed=sim["seed"], score=round(float(score if valid else score - 100), 3), valid=bool(valid),
                winner=NAMES[w], winner_index=w, launch_order=[NAMES[i] for i in order],
                launch_times_s=[round(tl[i], 2) for i in order], margin_s=round(float(margin), 3),
                margin_px=round(float(margin * v), 1), spread_s=round(float(spread), 2),
                burn_speed_px_s=round(float(v), 2), fuse_lengths_px=[round(x) for x in L],
                length_ratio=round(max(L) / min(L), 3), winner_was_shortest_fuse=bool(w == shortest),
                leader_changes=changes, last_leader_change_s=round(last_change, 2),
                junctions=len(sim["juncs"]), jumps_before_finish=len(jumps), shortcut_drops=shortcut,
                top_two_fuse_left_3s_before_s=[round(float(x), 2) for x in rem3])


def score_seed(seed):
    try:
        return drama(simulate(seed))
    except Exception as e:                        # a bad seed must never stop a search
        return dict(seed=seed, score=-1e9, valid=False, error=str(e))


# ====================================================================== look
COLS = [hex_rgb(c) for c in ("#FF7A30", "#2FD0FF", "#FF4FA8", "#FFD23F", "#48E39A")]
INK = hex_rgb("#07090C")
WHITE = (1.0, 1.0, 1.0)
STEEL = hex_rgb("#8C9AA8")
AMBER = hex_rgb("#FFB347")
FLAME = hex_rgb("#FF9A3C")
CHAR = hex_rgb("#2A2C31")
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
QA_, QB_ = "WHICH ROCKET", "LAUNCHES FIRST?"
RULE = "FIRST FLAME TO ITS ROCKET WINS"
TX = 70.0


def burst_xy(i, n):
    return lerp(ROCKET_X[i], 470.0, 0.3), BURST_Y[n]


def symbol(ctx, k, x, y, s):
    """Contestant mark k centred on (x, y), half-size s: triangle, ring, cross, square, diamond."""
    ctx.new_path()
    if k == 0:
        polygon(ctx, [(x + 1.2 * s * math.cos(a), y + 0.15 * s + 1.2 * s * math.sin(a))
                      for a in (-TAU / 4, TAU / 12, TAU * 5 / 12)])
        ctx.fill()
    elif k == 1:
        ctx.set_line_width(0.46 * s)
        ctx.arc(x, y, 0.78 * s, 0, TAU)
        ctx.stroke()
    elif k == 2:
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(TAU / 8)
        ctx.rectangle(-1.1 * s, -0.27 * s, 2.2 * s, 0.54 * s)
        ctx.rectangle(-0.27 * s, -1.1 * s, 0.54 * s, 2.2 * s)
        ctx.restore()
        ctx.fill()
    elif k == 3:
        ctx.rectangle(x - 0.82 * s, y - 0.82 * s, 1.64 * s, 1.64 * s)
        ctx.fill()
    else:
        polygon(ctx, [(x, y - 1.15 * s), (x + 0.9 * s, y), (x, y + 1.15 * s), (x - 0.9 * s, y)])
        ctx.fill()


def junction_mark(ctx, x, y, r, lit=0.0, alpha=1.0):
    ctx.new_path()
    ctx.arc(x, y, r, 0, TAU)
    ctx.set_source_rgba(*INK, 0.85 * alpha)
    ctx.fill_preserve()
    ctx.set_line_width(2.6)
    ctx.set_source_rgba(*mix(WHITE, FLAME, lit), alpha)
    ctx.stroke()
    ctx.new_path()
    ctx.arc(x, y, r * 0.36, 0, TAU)
    ctx.set_source_rgba(*mix(WHITE, FLAME, lit), alpha)
    ctx.fill()


def make_background():
    """Dark instrument panel: graphite gradient, a fine measuring grid, crop marks, grain."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (yy / H)[..., None]
    img = np.array([0.030, 0.036, 0.046], np.float32) * (1 - v) + np.array([0.020, 0.022, 0.028], np.float32) * v
    img *= np.clip(1.15 - 0.45 * (((xx - 470) / 800) ** 2 + ((yy - 880) / 1150) ** 2), 0.4, 1.0)[..., None]
    gx = (np.abs(((xx + 10) % 40) - 20) > 19.3) | (np.abs(((yy + 15) % 40) - 20) > 19.3)
    gX = (np.abs(((xx + 10) % 200) - 100) > 99.2) | (np.abs(((yy + 15) % 200) - 100) > 99.2)
    img += (gx * 0.016 + gX * 0.020)[..., None] * np.array([0.7, 0.9, 1.0], np.float32)
    rng = np.random.default_rng(5)
    img += rng.normal(0, 0.005, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    ctx.set_source_rgba(*STEEL, 0.35)
    ctx.set_line_width(2)
    for (x, y, dx, dy) in ((40, 175, 1, 1), (940, 175, -1, 1), (40, 1545, 1, -1), (940, 1545, -1, -1)):
        ctx.move_to(x, y + 34 * dy)
        ctx.line_to(x, y)
        ctx.line_to(x + 34 * dx, y)
        ctx.stroke()
    surf.flush()
    return surf, buf


def runs(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[cut + 1]])
    ends = np.concatenate([idx[cut], [idx[-1]]])
    return list(zip(starts, ends))


def path_run(ctx, x, y, a, b, step=2):
    ctx.move_to(x[a], y[a])
    for k in range(a + step, b, step):
        ctx.line_to(x[k], y[k])
    ctx.line_to(x[b], y[b])


class Scene:
    W, H, FPS, nframes = W, H, FPS, NF

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed, record=True)
        if not self.sim["valid"]:
            raise ValueError(f"seed {seed} has no valid layout")
        self.stats = drama(self.sim)
        self.v = self.sim["v"]
        self.order = self.sim["order"]
        self.w = self.order[0]
        self.tl = self.sim["t_launch"]
        self.place = {i: n for n, i in enumerate(self.order)}
        self.rem = rem_series(self.sim, np.arange(NF) / FPS)
        self.rem0 = self.rem[0].copy()
        self._prep_bursts()
        self._res = None

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_res"] = None
        return st

    def _prep_bursts(self):
        rng = np.random.default_rng(self.seed + 99)
        self.bursts = []
        self.rise = {}
        for n, i in enumerate(self.order):
            big = n == 0
            N = 230 if big else 100
            R = 300.0 if big else 185.0
            k = 2.4
            ang = rng.uniform(0, TAU, N)
            sp = R * k * rng.uniform(0.55, 1.0, N)
            tb = self.tl[self.w] + RISE + BURST_GAP * n
            self.rise[i] = max(0.9, tb - self.tl[i])
            self.bursts.append(dict(i=i, n=n, cx=burst_xy(i, n)[0], cy=burst_xy(i, n)[1], vx=sp * np.cos(ang),
                                    vy=sp * np.sin(ang), k=k, life=rng.uniform(2.2, 3.4, N),
                                    tb=self.tl[i] + self.rise[i], tw=rng.uniform(0, TAU, N), R=R, hue=rng.uniform(0, 0.35, N)))

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            bg, keep = make_background()
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep,
                             font=Font(FONT_PATH))
        return self._res

    def frame(self, f):
        R = self._resources()
        self.draw(R["ctx"], f, R["font"], R["bg"])
        R["surf"].flush()
        return bytes(R["surf"].get_data())

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, font, bg):
        t = f / FPS
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(bg, 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        kb = ease_in_out(ramp(t, T_RESET, T_BACK))          # fresh layout fading back in for the loop
        D = self.v * (t + T0)
        self._header(ctx, t, font, kb)
        if kb < 1:
            self._fuses(ctx, t, D, 1 - kb, f)
        if kb > 0:
            self._fuses(ctx, t, -1.0, kb, f, fresh=True)
        self._hub(ctx, t, kb)
        self._rockets(ctx, t, kb, font)
        self._chips(ctx, f, t, kb, font)
        self._bursts(ctx, t)
        self._bottom(ctx, t, font)

    def _header(self, ctx, t, font, kb):
        a = 1.0 - ramp(t, T_WIN + 0.6, T_WIN + 1.2) * (1 - kb)
        font.draw(ctx, "FUSE LEFT, IN SECONDS AT BURN SPEED", TX, 236, 30, STEEL, alpha=0.9 * a, tracking=1.0)
        font.draw(ctx, f"SEED {self.seed}", 950, 236, 30, STEEL, anchor="r", alpha=0.6 * a, tracking=1.0)

    def _fuses(self, ctx, t, D, alpha, f, fresh=False):
        sim = self.sim
        hot = []
        for i, fu in enumerate(sim["fuses"]):
            d = sim["dist"][i]
            x, y = fu["x"], fu["y"]
            b = d <= D
            col = COLS[i]
            for a_, c in runs(~b):
                ctx.new_path()
                path_run(ctx, x, y, a_, c)
                ctx.set_source_rgba(*darken(col, 0.62), alpha)
                ctx.set_line_width(10)
                ctx.stroke_preserve()
                ctx.set_source_rgba(*darken(col, 0.12), alpha)
                ctx.set_line_width(4.2)
                ctx.stroke()
            for a_, c in runs(b):
                ctx.new_path()
                path_run(ctx, x, y, a_, c)
                ctx.set_source_rgba(*CHAR, alpha)
                ctx.set_line_width(6)
                ctx.stroke()
            if not fresh:
                age = D - d
                for lo, hi, ga in ((0, 25, 0.55), (25, 60, 0.32), (60, 110, 0.16)):
                    for a_, c in runs((age >= lo) & (age < hi + 2)):
                        if c > a_:
                            hot.append((x, y, a_, c, ga))
        if hot:
            ctx.set_operator(cairo.OPERATOR_ADD)
            for x, y, a_, c, ga in hot:
                ctx.new_path()
                path_run(ctx, x, y, a_, c)
                ctx.set_source_rgba(*FLAME, ga * alpha)
                ctx.set_line_width(5)
                ctx.stroke()
            ctx.set_operator(cairo.OPERATOR_OVER)
        # junctions
        for n, j in enumerate(sim["juncs"]):
            dt = t - j["t_ign"]
            lit = 0.0 if fresh or dt < 0 else 1.0
            if not fresh and 0 <= dt < 0.7:
                ctx.new_path()
                ctx.arc(j["x"], j["y"], 14 + 46 * ease_out(dt / 0.7), 0, TAU)
                ctx.set_line_width(3)
                ctx.set_source_rgba(*AMBER, alpha * (1 - dt / 0.7))
                ctx.stroke()
            junction_mark(ctx, j["x"], j["y"], 13, lit, alpha)
        if fresh:
            return
        # the deciding gap: at the first launch, the runner-up's unburnt fuse lights up white until it is gone
        r2 = self.order[1]
        if self.tl[self.w] <= t < self.tl[r2] + 0.2:
            fu, d = sim["fuses"][r2], sim["dist"][r2]
            pls = 0.5 + 0.5 * math.cos((t - self.tl[self.w]) * TAU * 2.5)
            ctx.set_operator(cairo.OPERATOR_ADD)
            for a_, c in runs(d > D):
                ctx.new_path()
                path_run(ctx, fu["x"], fu["y"], a_, c, 1)
                ctx.set_source_rgba(1, 1, 1, (0.6 + 0.4 * pls) * alpha)
                ctx.set_line_width(12)
                ctx.stroke()
            ctx.set_operator(cairo.OPERATOR_OVER)
            if t < self.tl[r2]:
                for (fx, fy, fs, dr) in fronts(fu, d, D)[-1:]:
                    ctx.new_path()
                    ctx.arc(fx, fy, 30 + 8 * pls, 0, TAU)
                    ctx.set_line_width(4)
                    ctx.set_source_rgba(*mix(COLS[r2], WHITE, pls), alpha)
                    ctx.stroke()
        # flame fronts
        for i, fu in enumerate(sim["fuses"]):
            for (fx, fy, fs, dr) in fronts(fu, sim["dist"][i], D):
                self._spark(ctx, fx, fy, f, i * 1000 + int(fs), alpha)
        # jump labels
        for j in sim["juncs"]:
            dt = t - j["t_ign"]
            if j["jump"] and 0 <= dt < 1.1:
                a = alpha * (1 - ramp(dt, 0.7, 1.1))
                font = self._res["font"]
                lx = j["x"] + (24 if j["x"] < 800 else -24)
                font.draw(ctx, "JUMP", lx, j["y"] - 20, 34, AMBER, anchor="l" if j["x"] < 800 else "r",
                          alpha=a, stroke=INK, stroke_w=6, tracking=1.5)

    def _spark(self, ctx, x, y, f, key, alpha, scale=1.0):
        ctx.set_operator(cairo.OPERATOR_ADD)
        fl = 0.85 + 0.15 * math.sin(f * 1.7 + key)
        glow(ctx, x, y, 42 * scale * fl, FLAME, 0.75 * alpha)
        glow(ctx, x, y, 14 * scale, (1.0, 0.95, 0.8), 0.9 * alpha)
        rr = random.Random(f * 7919 + key)
        ctx.set_line_width(1.8)
        for _ in range(9):
            a = rr.uniform(0, TAU)
            r0, r1 = rr.uniform(4, 10) * scale, rr.uniform(14, 30) * scale
            ctx.new_path()
            ctx.move_to(x + r0 * math.cos(a), y + r0 * math.sin(a))
            ctx.line_to(x + r1 * math.cos(a), y + r1 * math.sin(a) + 4)
            ctx.set_source_rgba(1.0, rr.uniform(0.6, 0.9), 0.35, alpha * rr.uniform(0.4, 0.9))
            ctx.stroke()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.new_path()
        ctx.arc(x, y, 4.5 * scale, 0, TAU)
        ctx.set_source_rgba(1, 1, 1, alpha)
        ctx.fill()

    def _hub(self, ctx, t, kb):
        hx, hy = HUB
        ctx.new_path()
        ctx.arc(hx, hy, 24, 0, TAU)
        ctx.set_source_rgba(*INK, 1)
        ctx.fill_preserve()
        ctx.set_source_rgba(*STEEL, 0.9)
        ctx.set_line_width(3)
        ctx.stroke()
        for a in range(4):
            ang = a * TAU / 4
            ctx.move_to(hx + 30 * math.cos(ang), hy + 30 * math.sin(ang))
            ctx.line_to(hx + 40 * math.cos(ang), hy + 40 * math.sin(ang))
        ctx.stroke()
        # the hub spark: it is there on frame 0 and the converging embers become it on the last frame
        a = max(1 - ramp(t, 0.0, 0.8), ease_in(ramp(t, T_CONV + 1.0, DUR - 1 / FPS), 2))
        if a > 0.01:
            self._spark(ctx, hx, hy, int(t * FPS), 77, a, 1.15 + 0.5 * (1 - ramp(t, 0.0, 0.8)))
        if t < 0.8:
            ctx.new_path()
            ctx.arc(hx, hy, 30 + 190 * ease_out(t / 0.8), 0, TAU)
            ctx.set_line_width(4)
            ctx.set_source_rgba(*FLAME, 0.7 * (1 - t / 0.8))
            ctx.stroke()

    def _rocket_shape(self, ctx, i, x, y, ang, alpha, lit=0.0):
        col = COLS[i]
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(ang)
        ctx.new_path()
        ctx.move_to(-19, 0)
        ctx.line_to(-19, -86)
        ctx.curve_to(-19, -108, -6, -122, 0, -132)
        ctx.curve_to(6, -122, 19, -108, 19, -86)
        ctx.line_to(19, 0)
        ctx.close_path()
        ctx.set_source_rgba(*darken(col, 0.72), alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(*col, alpha)
        ctx.set_line_width(3.2)
        ctx.stroke()
        for sx in (-1, 1):
            polygon(ctx, [(sx * 19, -30), (sx * 34, -2), (sx * 34, 8), (sx * 19, 0)])
            ctx.set_source_rgba(*col, alpha)
            ctx.fill()
        ctx.rectangle(-19, -98, 38, 7)
        ctx.fill()
        ctx.set_source_rgba(*lighten(col, 0.55), alpha)
        symbol(ctx, i, 0, -52, 13)
        ctx.restore()

    def _rockets(self, ctx, t, kb, font):
        for i in range(NFUSE):
            x = ROCKET_X[i]
            # launch pad bracket
            ctx.set_source_rgba(*STEEL, 0.75)
            ctx.set_line_width(3)
            ctx.new_path()
            ctx.move_to(x - 36, PAD_Y + 6)
            ctx.line_to(x + 36, PAD_Y + 6)
            ctx.move_to(x - 30, PAD_Y + 6)
            ctx.line_to(x - 38, PAD_Y + 20)
            ctx.move_to(x + 30, PAD_Y + 6)
            ctx.line_to(x + 38, PAD_Y + 20)
            ctx.stroke()
            dt = t - self.tl[i]
            if kb > 0:
                self._rocket_shape(ctx, i, x, PAD_Y, 0.0, kb)
            if dt < 0:
                if kb <= 0:
                    self._rocket_shape(ctx, i, x, PAD_Y, 0.0, 1.0)
                continue
            n = self.place[i]
            if t < T_RESET + 0.6:
                a = 1 - ramp(t, T_RESET - 0.4, T_RESET + 0.6)
                font.draw(ctx, ("1ST", "2ND", "3RD", "4TH", "5TH")[n], x, PAD_Y - 40, 46 if n == 0 else 38,
                          COLS[i] if n == 0 else STEEL, anchor="c", alpha=a * ramp(dt, 0.3, 0.6), stroke=INK,
                          stroke_w=6)
            bx, by = burst_xy(i, n)
            rise = self.rise[i]
            if dt < rise:
                u = ease_in(dt / rise, 2)
                px, py = lerp(x, bx, u), lerp(PAD_Y, by, u)
                ang = math.atan2(bx - x, PAD_Y - by)
                # trail
                ctx.set_operator(cairo.OPERATOR_ADD)
                g = cairo.LinearGradient(x, PAD_Y, px, py)
                g.add_color_stop_rgba(0, *COLS[i], 0.0)
                g.add_color_stop_rgba(1, *lighten(COLS[i], 0.4), 0.9)
                ctx.set_source(g)
                ctx.set_line_width(5)
                ctx.new_path()
                ctx.move_to(x, PAD_Y)
                ctx.line_to(px, py)
                ctx.stroke()
                glow(ctx, px, py, 60, lighten(COLS[i], 0.3), 0.7)
                ctx.set_operator(cairo.OPERATOR_OVER)
                if kb <= 0:
                    self._rocket_shape(ctx, i, px, py, ang, 1.0)
                if dt < 0.35:
                    ctx.set_operator(cairo.OPERATOR_ADD)
                    glow(ctx, x, PAD_Y, 90, FLAME, 0.8 * (1 - dt / 0.35))
                    ctx.set_operator(cairo.OPERATOR_OVER)

    def _chips(self, ctx, f, t, kb, font):
        r2 = self.order[1]
        if 1 - ramp(t, T_WIN + 2.0, T_WIN + 2.5) * (1 - kb) <= 0.003:
            return
        rem = self.rem[f]
        live = rem.copy()
        lead = int(np.argmin(np.where(np.array(self.tl) > t, rem, np.inf))) if t < self.tl[self.w] else self.w
        for i in range(NFUSE):
            a_live = 1 - ramp(t, T_WIN + (2.0 if i == r2 else 0.8), T_WIN + (2.5 if i == r2 else 1.3)) * (1 - kb)
            if a_live <= 0.003:
                continue
            x = ROCKET_X[i]
            val = self.rem0[i] if kb > 0 else live[i]
            is_lead = i == lead and kb < 0.5
            done = t >= self.tl[i] and kb <= 0
            w_, h_, y0 = 150, 66, 290
            # flash when the fuse suddenly got shorter (a jump landed ahead of the flame)
            fl = 0.0
            for b in range(1, 12):
                if f - b >= 1 and self.rem[f - b - 1, i] - self.rem[f - b, i] > 0.5:
                    fl = max(fl, 1 - b / 12)
            rounded_rect(ctx, x - w_ / 2, y0, w_, h_, 10)
            ctx.set_source_rgba(*mix(INK, COLS[i], 0.14 + 0.5 * fl), 0.92 * a_live)
            ctx.fill_preserve()
            ctx.set_line_width(4 if is_lead else 2)
            ctx.set_source_rgba(*(COLS[i] if is_lead or fl > 0 else STEEL), (1.0 if is_lead else 0.55) * a_live)
            ctx.stroke()
            ctx.set_source_rgba(*COLS[i], a_live)
            symbol(ctx, i, x - 46, y0 + h_ / 2, 12)
            miss = i == r2 and kb <= 0 and t >= self.tl[self.w]
            pu = 0.5 + 0.5 * math.cos((t - self.tl[self.w]) * TAU * 2.5) if miss else 0.0
            if miss:
                rounded_rect(ctx, x - w_ / 2 - 7, y0 - 7, w_ + 14, h_ + 14, 15)
                ctx.set_line_width(5)
                ctx.set_source_rgba(*mix(COLS[i], WHITE, pu), a_live)
                ctx.stroke()
                gap = self.tl[r2] - self.tl[self.w]
                font.draw(ctx, f"MISSED BY {gap:.2f} S", x, y0 - 14, 30, mix(COLS[i], WHITE, 0.3 * pu), anchor="c",
                          alpha=a_live, stroke=INK, stroke_w=6, tracking=1.0)
            txt = (f"+{self.tl[i] - self.tl[self.w]:.2f}" if miss else "LIT") if done else f"{val:4.1f}"
            font.draw(ctx, txt, x + 22, y0 + 50, 42, WHITE if not done else COLS[i], anchor="c", alpha=a_live)
            if is_lead:
                font.draw(ctx, "LEADS", x, y0 - 12, 26, COLS[i], anchor="c", alpha=a_live, tracking=2.0)

    def _bursts(self, ctx, t):
        conv = ease_in_out(ramp(t, T_CONV, DUR - 1 / FPS))
        ctx.set_operator(cairo.OPERATOR_ADD)
        hx, hy = HUB
        for B in self.bursts:
            tt = t - B["tb"]
            if tt < 0:
                continue
            col = COLS[B["i"]]
            if tt < 0.6:
                glow(ctx, B["cx"], B["cy"], B["R"] * 0.8, lighten(col, 0.3), 0.55 * (1 - tt / 0.6) ** 2)
            if tt < 1.0:
                ctx.new_path()
                ctx.arc(B["cx"], B["cy"], B["R"] * 1.05 * ease_out(tt / 1.0), 0, TAU)
                ctx.set_line_width(2.5)
                ctx.set_source_rgba(*lighten(col, 0.4), 0.5 * (1 - tt))
                ctx.stroke()
            k = B["k"]
            e = 1 - math.exp(-k * tt)
            g = 230.0
            x = B["cx"] + B["vx"] / k * e
            y = B["cy"] + (B["vy"] - g / k) / k * e + g * tt / k
            tp = max(0.0, tt - 0.07)
            ep = 1 - math.exp(-k * tp)
            xp = B["cx"] + B["vx"] / k * ep
            yp = B["cy"] + (B["vy"] - g / k) / k * ep + g * tp / k
            life = np.clip(1 - tt / B["life"], 0, 1) ** 1.2
            life = np.maximum(life, 0.38 * ramp(tt, 0.4, 1.4)) * (0.72 + 0.28 * np.sin(B["tw"] + tt * 19.0))
            if conv > 0:
                ea = 0.75 * ramp(t, T_CONV, T_CONV + 0.7)
                life = np.maximum(life, ea)
                x, y = x + (hx - x) * conv, y + (hy - y) * conv
                xp, yp = xp + (hx - xp) * conv, yp + (hy - yp) * conv
            for n in range(len(x)):
                a = life[n]
                if a <= 0.02:
                    continue
                c = mix(col, (1.0, 0.85, 0.6), B["hue"][n])
                ctx.new_path()
                ctx.move_to(xp[n], yp[n])
                ctx.line_to(x[n], y[n])
                ctx.set_line_width((4.0 if B["n"] == 0 else 3.0) if conv < 0.5 else 2.4)
                ctx.set_source_rgba(*c, a)
                ctx.stroke()
        ctx.set_operator(cairo.OPERATOR_OVER)

    def _bottom(self, ctx, t, font):
        ra = ramp(t, T_WIN + 0.55, T_WIN + 0.95) * (1 - ramp(t, T_RESET - 0.6, T_RESET + 0.1))
        ha = 1 - ramp(t, T_WIN + 0.15, T_WIN + 0.5) * (1 - ramp(t, T_RESET + 0.2, T_BACK))
        if ha > 0.003:
            ctx.rectangle(TX - 28, 1262, 8, 270)
            ctx.set_source_rgba(*AMBER, ha)
            ctx.fill()
            font.draw(ctx, QA_, TX, 1340, 96, WHITE, alpha=ha, tracking=1.0)
            font.draw(ctx, QB_, TX, 1436, 96, WHITE, alpha=ha, tracking=1.0)
            font.draw(ctx, RULE, TX, 1488, 44, AMBER, alpha=ha, tracking=1.2)
            wpre = font.width("FLAME JUMPS BOTH WAYS AT EVERY ", 31, 1.2)
            font.draw(ctx, "FLAME JUMPS BOTH WAYS AT EVERY", TX, 1530, 31, STEEL, alpha=ha, tracking=1.2)
            junction_mark(ctx, TX + wpre + 14, 1519, 11, 0.0, ha)
            font.draw(ctx, "JUNCTION", TX + wpre + 34, 1530, 31, STEEL, alpha=ha, tracking=1.2)
        if ra > 0.003:
            w, r2 = self.w, self.order[1]
            ctx.rectangle(TX - 28, 1262, 8, 270)
            ctx.set_source_rgba(*COLS[w], ra)
            ctx.fill()
            size = 88
            txt = f"{NAMES[w]} LAUNCHES FIRST"
            size = min(size, 800 / max(1, font.width(txt, 1) + 0.82))
            ctx.set_source_rgba(*COLS[w], ra)
            symbol(ctx, w, TX + 0.30 * size, 1360 - 0.36 * size, 0.30 * size)
            font.draw(ctx, txt, TX + 0.82 * size, 1360, size, COLS[w], alpha=ra, tracking=1.0)
            m = self.tl[r2] - self.tl[w]
            font.draw(ctx, f"{NAMES[r2]} WAS {m:.2f} S OF FUSE BEHIND", TX, 1436, 54, WHITE, alpha=ra, tracking=1.0)
            font.draw(ctx, "SAME BURN SPEED ON EVERY FUSE", TX, 1488, 36, STEEL, alpha=ra, tracking=1.2)
            font.draw(ctx, "DID YOU CALL IT?", TX, 1532, 36, AMBER, alpha=ra, tracking=1.2)

    # ------------------------------------------------------------ sound
    def audio(self):
        """Hiss per burning fuse (band-passed noise gated to that fuse having a live flame, its center rising
        as the flame nears the rocket), sparse crackle, a tick on every junction, a soft pulse that tightens
        in the last seconds, launch sweeps, burst thumps with pentatonic pings, and a converging shimmer."""
        sr = au.SR
        n, tail = int(round(DUR * sr)), 3 * sr
        rng = np.random.default_rng(self.seed + 4242)
        wet = np.zeros((n + tail, 2))
        dry = np.zeros((n + tail, 2))
        sim, v = self.sim, self.v
        tf = np.arange(NF) / FPS
        # per-frame activity and progress of each fuse
        act = np.zeros((NF, NFUSE))
        fc = np.zeros((NF, NFUSE))
        pan = np.zeros((NF, NFUSE))
        crk = []
        for f in range(NF):
            if tf[f] > T_WIN + 4.0:
                break
            D = v * (tf[f] + T0)
            for i, fu in enumerate(sim["fuses"]):
                fr = fronts(fu, sim["dist"][i], D)
                if fr:
                    act[f, i] = min(1.0, 0.6 + 0.2 * len(fr))
                    fc[f, i] = 900.0 * 4.0 ** (1 - self.rem[f, i] * v / fu["L"])
                    pan[f, i] = (fr[0][0] - 470) / 520
                    for fx, fy, _, _ in fr:
                        if rng.random() < 0.10:
                            crk.append((tf[f] + rng.uniform(0, 1 / FPS), (fx - 470) / 520))
        hiss = self._hiss(rng, n, act, fc, pan)
        dry[:n] += 0.05 * hiss
        for tc, p in crk:
            m = int(rng.uniform(0.002, 0.006) * sr)
            x = au.bandpass(rng.standard_normal(m + 64), rng.uniform(1800, 3000), rng.uniform(5000, 8000))[:m]
            x *= np.exp(-np.arange(m) / sr / 0.0015)
            au.place(dry, x, tc, rng.uniform(0.25, 0.6), float(np.clip(p, -0.8, 0.8)))
        # pulse: every 0.5 s, every 0.25 s in the last 3 s before the first launch
        tp, k = 0.0, 0
        while tp < T_WIN:
            close = tp > T_WIN - 3.0
            au.place(dry, au.tick(1300 + 400 * close, 0.08), tp, 0.22 + 0.08 * close)
            if k % 4 == 0 and not close:
                au.place(wet, au.marimba(au.hz(57), 0.5), tp, 0.16)
            tp += 0.25 if close else 0.5
            k += 1
        # junction ticks
        for j in sim["juncs"]:
            if 0 <= j["t_ign"] < DUR:
                p = float(np.clip((j["x"] - 470) / 520, -0.8, 0.8))
                au.place(wet, au.tick(2400, 0.07), j["t_ign"], 0.30 if j["jump"] else 0.15, p)
                if j["jump"]:
                    au.place(wet, au.glide(900, 1700, 0.09, 0.05), j["t_ign"], 0.12, p)
        # sudden shortcut on a readout
        for f in range(1, NF):
            for i in range(NFUSE):
                if self.rem[f - 1, i] - self.rem[f, i] > 0.5:
                    au.place(wet, au.chip(au.hz(76 + 2 * i), 0.35), tf[f], 0.16)
        # launches and bursts
        penta = (72, 74, 76, 79, 81, 84, 86, 88)
        for B in self.bursts:
            i, nn = B["i"], B["n"]
            tlz = self.tl[i]
            p = float(np.clip((ROCKET_X[i] - 470) / 520, -0.7, 0.7))
            g = 1.0 if nn == 0 else 0.7
            rs = self.rise[i]
            au.place(dry, au.glide(200, 1800, rs, harm=0.2), tlz, 0.11 * g, p)
            au.place(dry, au.swept_noise(rng, rs, 500, 5000) * np.linspace(0.3, 1, int(rs * sr)), tlz,
                     0.09 * g, p)
            tb = B["tb"]
            bp = float(np.clip((B["cx"] - 470) / 520, -0.7, 0.7))
            tt = np.arange(int(1.0 * sr)) / sr
            au.place(dry, np.sin(TAU * 60 * tt) * np.exp(-tt / 0.22) * np.minimum(1, tt / 0.003), tb, 0.45 * g, bp)
            au.place(dry, au.noise(rng, 0.8, 250, 7000) * np.exp(-tt[:int(0.8 * sr)] / 0.16), tb, 0.45 * g, bp)
            au.place(dry, au.clap(rng, 0.16), tb, 0.9 * g, bp)
            for q in range(6 if nn == 0 else 3):
                au.place(wet, au.bell(au.hz(penta[rng.integers(len(penta))]), 1.2), tb + rng.uniform(0.12, 1.3),
                         0.22 * g, float(rng.uniform(-0.6, 0.6)))
        # result pad and the converging shimmer into the loop
        for mnote, gg in ((57, 0.5), (64, 0.4), (69, 0.3), (76, 0.2)):
            au.place(wet, au.pad(au.hz(mnote), T_RESET + 1.0 - (T_WIN + 0.8), 0.3, 1.2), T_WIN + 0.8, 0.05 * gg)
        cl = DUR - T_CONV
        env = (np.arange(int(cl * sr)) / (cl * sr)) ** 2
        au.place(dry, au.swept_noise(rng, cl, 4500, 900) * env, T_CONV, 0.22)
        au.place(wet, au.glide(330, 880, cl) * env, T_CONV, 0.05)
        tq = T_CONV + 0.3                       # embers ticking home, then the hub catching again
        while tq < DUR - 0.3:
            pq = float(rng.uniform(-0.5, 0.5))
            au.place(wet, au.bell(au.hz(penta[rng.integers(len(penta))] + 12), 0.6), tq, 0.18, pq)
            au.place(dry, au.tick(2600, 0.05), tq, 0.25, pq)
            tq += rng.uniform(0.35, 0.6)
        mix_ = au.reverb(wet, rng, decay=1.4, wet=0.25) + dry
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]
        edge = np.minimum(1.0, np.minimum(np.arange(n), n - np.arange(n)) / (0.01 * sr))
        out *= edge[:, None]
        return au.eq(out, ((40, -8.0), (90, -3.0), (250, -1.0), (600, 0.0), (1500, 1.5), (3500, 3.0),
                           (9000, 2.0), (16000, -2.0)))

    @staticmethod
    def _hiss(rng, n, act, fc, pan):
        """Stereo noise through a per-hop spectral gain: one gaussian band per burning fuse, each gated by
        that fuse having a live flame (no flame anywhere = silence)."""
        sr = au.SR
        win, hop = 2048, 512
        x = rng.standard_normal((n + win, 2))
        out = np.zeros((n + win, 2))
        wnd = np.hanning(win)
        fr = np.fft.rfftfreq(win, 1 / sr)
        lf = np.log(np.maximum(fr, 1.0))
        for i0 in range(0, n, hop):
            f = min(len(act) - 1, int(i0 / sr * FPS))
            a = act[f]
            if a.max() <= 0:
                continue
            gl = np.zeros_like(fr)
            gr = np.zeros_like(fr)
            for i in range(NFUSE):
                if a[i] <= 0:
                    continue
                g = a[i] * np.exp(-0.5 * ((lf - math.log(fc[f, i])) / 0.35) ** 2)
                ang = (float(np.clip(pan[f, i], -0.8, 0.8)) + 1) * math.pi / 4
                gl += (g * math.cos(ang)) ** 2
                gr += (g * math.sin(ang)) ** 2
            for ch, gg in ((0, np.sqrt(gl)), (1, np.sqrt(gr))):
                out[i0:i0 + win, ch] += np.fft.irfft(np.fft.rfft(x[i0:i0 + win, ch] * wnd) * gg, win) * wnd
        return out[:n] / 1.5

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        w, r2 = self.w, self.order[1]
        st = self.stats
        title = "Which rocket launches first? 5 tangled fuses, 1 spark"
        assert len(title) <= 60
        m = st["margin_s"]
        return dict(
            title=title,
            description=(
                "One spark lights a hub. Five tangled fuses run from it to five rockets, and every fuse is a "
                "smooth random curve whose length is within 12% of the others. The flame burns at the same "
                "constant speed on every fuse, the whole way, and the speed never changes. Wherever two fuses "
                "cross there is a marked junction: the flame jumps across in both directions and then burns both "
                "ways along the fuse it lands on. The first flame to reach its rocket launches it, and the rest "
                "follow in arrival order. The numbers over the rockets are the fuse left in front of each flame, "
                "in seconds at burn speed, so every shortcut shows up as a sudden drop.\n"
                f"This run: {NAMES[w]} launched first and {NAMES[r2]} was {m:.2f} s of fuse behind. The burn is an "
                "exact shortest-path calculation over the fuse layout; the close finish comes from searching seeds "
                f"for one, not from editing the race. Layout, motion and sound are generated by code (seed "
                f"{self.seed}), with no footage, samples or stock media.\n"
                "Which rocket did you pick?"),
            tags=["simulation", "fuse race", "rockets", "fireworks", "shortest path", "prediction",
                  "generative animation", "made with code"],
            pinned_comment=("Pick your rocket before the flame reaches the first junction: orange, cyan, magenta, "
                            "gold or green? Same burn speed on every fuse, so only the layout and the jumps decide "
                            "it. Comment your color, then check the finish."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylised 2D animation of abstract lines and shapes; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            duration_seconds=DUR,
            drama=st,
        )
