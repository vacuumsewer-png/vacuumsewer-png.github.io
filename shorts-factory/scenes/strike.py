"""STRIKE: a lightning leader grows from the cloud toward five towers. The first tower it touches is struck.

A simplified model (dielectric breakdown growth), NOT real lightning physics, and real lightning does not always hit the
tallest object. Rules, all on screen:
  * the sky is a 136 x 91 grid; the cloud is at the top, the ground and every tower are held at potential 1, the leader
    (and the cloud) at 0; the potential in between is relaxed with Jacobi iterations after every step;
  * every step the leader adds one cell next to itself; the cell is picked with probability proportional to the
    potential (to the power 0.7), so branches fork and several tips compete;
  * the first cell that touches a tower triggers the return stroke: that tower is struck;
  * the leader and the towers are drawn from frame 0; each tower shows its gap (cells between the leader and the tower)
    and a streamer that grows as the leader gets close.
The simulation is deterministic per seed, so make.py can score many seeds without rendering and render only the best one.
"""
import math
import os

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_out, ease_out_back, glow, hex_rgb, lerp, lighten, mix,
                        polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
T_TOTAL = 25.0                  # seconds; frame 0 is the clean loop point
T_TOUCH = 23.5                  # the return stroke: the winner exists only in the last 1.5 s
T_FIRST = 0.12                  # first leader cell
ROWS, COLS, CELL = 91, 136, 7
G = ROWS - 1                    # ground row (potential 1)
X0, Y0 = 0.0, 720.0             # pixel position of cell (0, 0); the grid ends at x = 952 (right 12% stays free)
NT = 5
TW = 6                          # tower width in cells
JAC = 16                        # Jacobi iterations per growth step
ETA = 0.7                       # chance of a cell ~ potential ** ETA (0.7 keeps the leader bushy: 2 gives a bare line)
MAXS = 1400
S_MIN, S_MAX = 240, 760         # allowed number of growth steps (about 13 to 33 per second)
NAMES = ("CORAL", "LIME", "SKY", "VIOLET", "GOLD")      # symbols: circle, square, triangle, diamond, star


def cx_of(c):
    return X0 + (c + 0.5) * CELL


def cy_of(r):
    return Y0 + (r + 0.5) * CELL


# ====================================================================== simulation
def layout(rng):
    """Five towers: heights, spacing and the start point are all random per seed."""
    while True:
        cols = np.sort(rng.integers(9, 126, NT))
        if np.all(np.diff(cols) >= 20):
            break
    hs = np.array([13, 23, 33, 43, 52]) + rng.integers(-3, 4, NT)
    hs = rng.permutation(hs)
    order = rng.permutation(NT)            # which color/symbol each position carries
    towers = []
    for i in range(NT):
        c0 = int(cols[i]) - TW // 2
        h = int(hs[i])
        towers.append(dict(k=int(order[i]), c0=c0, c1=c0 + TW - 1, r0=G - h, r1=G - 1, h=h,
                           sc0=c0 + 2, sc1=c0 + 3, sr0=G - h - 3, sr1=G - h - 1))
    start = int(rng.integers(40, 96))
    return towers, start


def dist_maps(towers):
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    out = np.zeros((NT, ROWS, COLS), np.int16)
    for i, t in enumerate(towers):
        d1 = np.maximum(np.maximum(t["c0"] - cc, cc - t["c1"]), np.maximum(t["r0"] - rr, 0))
        d1 = np.maximum(d1, 0)
        dx = np.maximum(np.maximum(t["sc0"] - cc, cc - t["sc1"]), 0)
        dy = np.maximum(np.maximum(t["sr0"] - rr, rr - t["sr1"]), 0)
        d2 = np.maximum(dx, dy)
        out[i] = np.minimum(d1, d2)
    return out


def simulate(seed):
    rng = np.random.default_rng(seed * 7919 + 13)
    towers, start = layout(rng)
    dk = dist_maps(towers)
    solid = (dk == 0).any(axis=0)
    fixed = np.zeros((ROWS, COLS), bool)
    val = np.zeros((ROWS, COLS), np.float32)
    fixed[0, :] = True
    fixed[G, :] = True
    val[G, :] = 1.0
    fixed |= solid
    val[solid] = 1.0
    phi = np.tile(np.linspace(0, 1, ROWS, dtype=np.float32)[:, None], (1, COLS))
    phi = np.where(fixed, val, phi)

    def relax(p, n):
        for _ in range(n):
            q = p.copy()
            q[1:-1, 1:-1] = 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
            q[:, 0] = q[:, 1]
            q[:, -1] = q[:, -2]
            p = np.where(fixed, val, q)
        return p

    phi = relax(phi, 250)
    cluster = np.zeros((ROWS, COLS), bool)
    front = np.zeros((ROWS, COLS), bool)
    cells, parent, nchild = [], [], []
    gaps, near = [], []
    cur_d = np.full(NT, 999, np.int32)
    cur_n = np.zeros((NT, 2), np.int32)
    fork_s = None
    touch = None                       # (tower index, step) of the first touch
    ground = False

    def add(r, c, par, s):
        nonlocal cur_d, cur_n
        cluster[r, c] = True
        front[r, c] = False
        phi[r, c] = 0.0
        fixed[r, c] = True
        val[r, c] = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr_, cc_ = r + dr, c + dc
                if 1 <= rr_ < G and 0 <= cc_ < COLS and not cluster[rr_, cc_] and not solid[rr_, cc_]:
                    front[rr_, cc_] = True
        cells.append((r, c))
        parent.append(par)
        nchild.append(0)
        d = dk[:, r, c].astype(np.int32)
        better = d < cur_d
        cur_d = np.where(better, d, cur_d)
        cur_n[better] = (r, c)
        gaps.append(cur_d.copy())
        near.append(cur_n.copy())

    add(1, start, -1, 0)
    index = {(1, start): 0}
    for s in range(1, MAXS):
        phi = relax(phi, JAC)
        idx = np.flatnonzero(front)
        if len(idx) == 0:
            break
        p = np.maximum(phi.ravel()[idx], 0.0) ** ETA
        tot = float(p.sum())
        if tot <= 0:
            k = int(rng.integers(len(idx)))
        else:
            k = int(np.searchsorted(np.cumsum(p), rng.random() * tot))
            k = min(k, len(idx) - 1)
        r, c = divmod(int(idx[k]), COLS)
        nb = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
              if (dr or dc) and (r + dr, c + dc) in index]
        orth = [q for q in nb if q[0] == r or q[1] == c]
        pr, pc = (orth or nb)[int(rng.integers(len(orth or nb)))]
        par = index[(pr, pc)]
        if nchild[par] >= 1 and fork_s is None:
            fork_s = s
        nchild[par] += 1
        index[(r, c)] = s
        add(r, c, par, s)
        hit = np.nonzero(dk[:, r, c] <= 1)[0]
        if len(hit):
            touch = (int(hit[np.argmin(dk[hit, r, c])]), s)
            break
        if r >= G - 1:
            ground = True
            break
    return dict(seed=seed, towers=towers, start=start, cells=cells, parent=parent, gaps=np.array(gaps),
                near=np.array(near), fork_s=fork_s, touch=touch, ground=ground)


def times_for(n):
    return T_FIRST + (T_TOUCH - T_FIRST) * np.arange(n) / max(1, n - 1)


def drama(sim):
    stats = dict(seed=sim["seed"], valid=False, score=-100.0)
    if sim["touch"] is None:
        stats["note"] = "ground or nothing hit first" if sim["ground"] else "no touch"
        return stats
    win, s_end = sim["touch"]
    S = s_end + 1
    ts = times_for(S)
    gaps = np.maximum(sim["gaps"] - 1, 0)            # cells between the leader and each tower
    towers = sim["towers"]
    hs = [t["h"] for t in towers]
    s1 = int(np.searchsorted(ts, T_TOUCH - 1.0))
    g1 = np.sort(gaps[min(s1, S - 1)])
    last = gaps[S - 1].astype(int)
    others = [i for i in range(NT) if i != win]
    run = min(others, key=lambda i: last[i])
    close_id = np.argmin(gaps, axis=1)
    sw = int(np.searchsorted(ts, 3.0))
    changes = int(np.sum(close_id[sw + 1:] != close_id[sw:-1]))
    fork_t = ts[sim["fork_s"]] if sim["fork_s"] is not None else 99.0
    s75 = int(0.75 * S)
    gates = dict(
        steps=S_MIN <= S <= S_MAX,
        early_fork=bool(fork_t <= 2.0),
        near_two=bool(g1[0] <= 8 and g1[1] <= 8),
        near_miss=bool(last[run] <= 5),
        winner_not_tallest=bool(hs[win] < max(hs)),
        not_closest_at_75=bool(int(close_id[s75]) != win),
    )
    ok = all(gates.values())
    score = (100.0 if ok else 0.0) + 8.0 * sum(gates.values())
    score += -3.0 * float(last[run]) - 1.2 * float(g1[1]) + 1.5 * min(changes, 6) + 6.0 * ramp(
        float(ts[min(np.nonzero(close_id != win)[0][-1] + 1, S - 1)]) if len(np.nonzero(close_id != win)[0]) else 0.0,
        10.0, T_TOUCH - 0.6)
    stats = dict(seed=sim["seed"], score=round(float(score), 2), valid=bool(ok), gates={k: bool(v) for k, v in gates.items()},
                 winner=NAMES[towers[win]["k"]], winner_index=int(win), winner_height_cells=int(hs[win]),
                 runner_up=NAMES[towers[run]["k"]], runner_up_gap_cells=int(last[run]),
                 gap_second_closest_one_second_out=int(g1[1]), steps=int(S),
                 tower_heights={NAMES[t["k"]]: int(t["h"]) for t in towers},
                 first_fork_s=round(float(fork_t), 2), closest_tower_changes=changes,
                 leader_cells=int(len(sim["cells"])))
    return stats


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
COL = [hex_rgb("#FF6B57"), hex_rgb("#B8F04A"), hex_rgb("#4CC9FF"), hex_rgb("#B38CFF"), hex_rgb("#FFD23F")]
INK = hex_rgb("#06121A")
CREAM = hex_rgb("#F3F0E6")
MUTED = hex_rgb("#8FB0B5")
AMBER = hex_rgb("#FFC23D")
BOLT = (0.88, 0.96, 1.0)
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
HX = 475.0
Q1, Q2 = "WHICH TOWER", "GETS STRUCK?"
RULE = "FIRST TOWER THE LEADER TOUCHES IS HIT"
SUBTXT = "LEADER GROWS TOWARD THE STRONGEST FIELD   •   SIMPLIFIED MODEL"
END_TEXT = "Got it right? +1 streak. Subscribe to keep it."


def symbol(ctx, k, x, y, s):
    if k == 0:
        ctx.new_sub_path()
        ctx.arc(x, y, s, 0, TAU)
    elif k == 1:
        ctx.rectangle(x - s * 0.86, y - s * 0.86, s * 1.72, s * 1.72)
    elif k == 2:
        polygon(ctx, [(x, y - s * 1.15), (x + s * 1.1, y + s * 0.8), (x - s * 1.1, y + s * 0.8)])
    elif k == 3:
        polygon(ctx, [(x, y - s * 1.2), (x + s * 0.95, y), (x, y + s * 1.2), (x - s * 0.95, y)])
    else:
        pts = []
        for i in range(10):
            a = -TAU / 4 + i * TAU / 10
            rr = s * 1.25 if i % 2 == 0 else s * 0.52
            pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
        polygon(ctx, pts)


def badge(ctx, k, x, y, r, alpha=1.0):
    ctx.new_path()
    ctx.arc(x, y, r, 0, TAU)
    ctx.set_source_rgba(*COL[k], alpha)
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK, alpha)
    ctx.set_line_width(3)
    ctx.stroke()
    ctx.new_path()
    symbol(ctx, k, x, y, r * 0.5)
    ctx.set_source_rgba(*INK, alpha)
    ctx.fill()


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


def make_background(sim):
    rng = np.random.default_rng(5)
    yy = np.arange(H, dtype=np.float32)[:, None]
    top, mid, hor = (np.array(x, np.float32) for x in ((0.030, 0.075, 0.105), (0.075, 0.185, 0.225), (0.17, 0.36, 0.37)))
    u1 = np.clip(yy / 760.0, 0, 1)
    u2 = np.clip((yy - 760.0) / 600.0, 0, 1) ** 1.3
    img = (top[None, :] + (mid - top)[None, :] * u1) + (hor - mid)[None, :] * u2
    img = np.repeat(img[:, None, :], W, axis=1)
    img += rng.normal(0, 0.004, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    del img
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    # far ridge and a low city line
    ctx.set_source_rgba(0.04, 0.11, 0.14, 1)
    ctx.move_to(0, 1350)
    x = 0.0
    while x <= W:
        ctx.line_to(x, 1262 - 55 * math.sin(x / 190.0 + 1.0) - 28 * math.sin(x / 61.0))
        x += 20
    ctx.line_to(W, 1350)
    ctx.close_path()
    ctx.fill()
    for _ in range(26):
        bx, bw, bh = rng.uniform(0, 1000), rng.uniform(22, 52), rng.uniform(30, 90)
        ctx.set_source_rgba(0.03, 0.085, 0.11, 1)
        ctx.rectangle(bx, 1350 - bh, bw, bh)
        ctx.fill()
        for wy in range(int(1350 - bh + 8), 1340, 16):
            for wx in range(int(bx + 6), int(bx + bw - 6), 12):
                if rng.random() < 0.18:
                    ctx.set_source_rgba(0.95, 0.80, 0.45, 0.35)
                    ctx.rectangle(wx, wy, 5, 7)
                    ctx.fill()
    # cloud mass
    for lay, (a, rr0) in enumerate(((0.95, 1.0), (0.9, 0.8))):
        rng2 = np.random.default_rng(11 + lay)
        for _ in range(46):
            x = rng2.uniform(-30, 1010)
            y = 650 + rng2.uniform(-18, 70) * rr0 + (18 if lay else 0)
            r = rng2.uniform(55, 105) * rr0
            g = cairo.RadialGradient(x, y, 0, x, y, r)
            c = (0.07, 0.12, 0.15) if lay == 0 else (0.11, 0.17, 0.20)
            g.add_color_stop_rgba(0, *c, a)
            g.add_color_stop_rgba(0.7, *c, a * 0.8)
            g.add_color_stop_rgba(1, *c, 0)
            ctx.set_source(g)
            ctx.arc(x, y, r, 0, TAU)
            ctx.fill()
    # ground
    g = cairo.LinearGradient(0, Y0 + G * CELL, 0, 1536)
    g.add_color_stop_rgba(0, 0.05, 0.12, 0.14, 1)
    g.add_color_stop_rgba(1, 0.02, 0.06, 0.08, 1)
    ctx.set_source(g)
    ctx.rectangle(0, Y0 + G * CELL, W, H - (Y0 + G * CELL))
    ctx.fill()
    ctx.set_source_rgba(0.45, 0.78, 0.76, 0.55)
    ctx.set_line_width(3)
    ctx.move_to(0, Y0 + G * CELL)
    ctx.line_to(W, Y0 + G * CELL)
    ctx.stroke()
    for t in sim["towers"]:
        draw_tower(ctx, t)
    return surf, buf


def tower_rect(t):
    return (X0 + t["c0"] * CELL, Y0 + t["r0"] * CELL, TW * CELL, (t["r1"] - t["r0"] + 1) * CELL)


def draw_tower(ctx, t, tint=0.0, a=1.0):
    k = t["k"]
    x, y, w, h = tower_rect(t)
    body = mix(darken(COL[k], 0.62), COL[k], tint)
    ctx.rectangle(x, y, w, h)
    ctx.set_source_rgba(*body, a)
    ctx.fill()
    ctx.set_source_rgba(*mix(COL[k], (1, 1, 1), 0.15 + 0.5 * tint), 0.55 * a)
    ctx.set_line_width(2)
    for yy in np.arange(y + 28, y + h - 6, 28):                 # lattice braces
        ctx.move_to(x, yy)
        ctx.line_to(x + w, yy + 28)
        ctx.move_to(x + w, yy)
        ctx.line_to(x, yy + 28)
    ctx.stroke()
    ctx.rectangle(x, y, w, h)
    ctx.set_source_rgba(*COL[k], a)
    ctx.set_line_width(3.5)
    ctx.stroke()
    ctx.rectangle(x, y, w, 12)                                  # colored cap
    ctx.set_source_rgba(*COL[k], a)
    ctx.fill()
    ctx.rectangle(X0 + t["sc0"] * CELL, Y0 + t["sr0"] * CELL, 2 * CELL, 3 * CELL)       # spike
    ctx.set_source_rgba(*lighten(COL[k], 0.35 + 0.5 * tint), a)
    ctx.fill()
    ctx.new_path()                                              # symbol plate
    px, py = x + w / 2, y + 40
    ctx.arc(px, py, 17, 0, TAU)
    ctx.set_source_rgba(*COL[k], a)
    ctx.fill()
    ctx.new_path()
    symbol(ctx, k, px, py, 9)
    ctx.set_source_rgba(*INK, a)
    ctx.fill()


def make_header(font):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, 460)
    ctx = cairo.Context(surf)
    s1 = min(124.0, 850.0 / font.width(Q2, 1.0))
    font.draw(ctx, Q1, HX, 124, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    font.draw(ctx, Q2, HX, 124 + s1 * 1.1, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    chip(ctx, font, RULE, HX, 124 + s1 * 1.1 + 92, 42, pad=22, bg=AMBER)
    font.draw(ctx, SUBTXT, HX, 124 + s1 * 1.1 + 148, 25, CREAM, anchor="c", stroke=INK, stroke_w=6, tracking=0.6)
    surf.flush()
    return surf


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((50, -8.0), (120, -4.0), (300, -1.0), (900, 0.0), (3000, 2.0), (8000, 0.5), (16000, -4.0))
    DRIVE = 3.2
    CEILING_DB = -3.4
    HEAD_Y = 150

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed)
        if self.sim["touch"] is None:
            raise ValueError(f"seed {seed}: no tower is touched first")
        self.stats = drama(self.sim)
        self.win, s_end = self.sim["touch"]
        self.S = s_end + 1
        self.ts = times_for(self.S)
        self.nframes = int(round(T_TOTAL * FPS))
        self.dur = self.nframes / FPS
        self.towers = self.sim["towers"]
        self.wk = self.towers[self.win]["k"]
        self.gapv = np.maximum(self.sim["gaps"] - 1, 0)
        cells = self.sim["cells"]
        self.px = np.array([(cx_of(c), cy_of(r)) for r, c in cells])
        par = self.sim["parent"]
        self.par = np.array(par)
        path, i = [], self.S - 1
        while i >= 0:
            path.append(i)
            i = par[i]
        self.path = path[::-1]
        self.on_path = np.zeros(self.S, bool)
        self.on_path[self.path] = True
        self.forks = []
        seen = {}
        for s in range(1, self.S):
            seen[par[s]] = seen.get(par[s], 0) + 1
            if seen[par[s]] == 2:
                self.forks.append(s)
        self._res = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            font = Font(FONT_PATH)
            bg, keep = make_background(self.sim)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep, font=font,
                             head=make_header(font))
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
        t = f / FPS
        d = t - T_TOUCH
        font = res["font"]
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        n = int(np.searchsorted(self.ts, t - 1e-9, side="right"))     # cells revealed
        n = min(n, self.S) if d < 0 else self.S
        self._cloud(ctx, t)
        self._rain(ctx, t)
        self._leader(ctx, t, d, n)
        gi = max(0, n - 1)
        gap = self.gapv[gi] if d < 1.0 else self.gapv[0]       # numbers reset with the sky for the loop
        fade_end = 1.0 - ramp(d, 0.2, 0.8)
        self._streamers(ctx, f, d, n, gap, fade_end)
        if d >= 0:
            self._strike(ctx, d, font)
        self._labels(ctx, font, gap, d)
        ctx.set_source_surface(res["head"], 0, self.HEAD_Y)
        ctx.paint()
        self._countdown(ctx, font, t, d)
        if d >= 0:
            self._result(ctx, d, font)
            fl = self._flash(d)
            if fl > 0.004:
                ctx.set_source_rgba(0.80, 0.84, 0.88, fl)       # about 80% white, one soft flash
                ctx.paint()
        self._endcard(ctx, font, d)

    def _flash(self, d):
        if d < 0:
            return 0.0
        return 0.62 * ramp(d, 0.0, 0.10) * math.exp(-max(0.0, d - 0.10) / 0.16)

    def _cloud(self, ctx, t):
        for i in range(7):
            k = 1 + (i % 3)
            x = 120 + 130 * i + 30 * math.sin(TAU * k * t / T_TOTAL + i)
            a = 0.05 + 0.04 * math.sin(TAU * (i % 4 + 1) * t / T_TOTAL + 2 * i)
            glow(ctx, x, 690 + 10 * math.sin(i), 120, (0.45, 0.62, 0.78), max(0.0, a))
        sx = cx_of(self.sim["start"])
        a = 0.35 + 0.15 * math.sin(TAU * 5 * t / T_TOTAL)
        glow(ctx, sx, Y0 + 4, 34, BOLT, a * max(1 - ramp(t, 0.1, 1.2), ramp(t, T_TOTAL - 0.8, T_TOTAL - 0.1)) + 0.2)

    def _rain(self, ctx, t):
        ctx.set_source_rgba(0.75, 0.88, 0.95, 0.13)
        ctx.set_line_width(2)
        span = 1350.0 - 760.0
        for i in range(70):
            k = 2 + i % 4                                   # whole cycles over the video so the loop is clean
            ph = (hash1(i, 3.0) + k * t / T_TOTAL * (1 + (i % 3))) % 1.0
            x = hash1(i, 9.0) * 950 - 40 * ph
            y = 760 + span * ph
            ctx.move_to(x, y)
            ctx.line_to(x - 7, y + 28)
        ctx.stroke()

    def _leader(self, ctx, t, d, n):
        if n <= 0:
            return
        px, par, ts = self.px, self.par, self.ts
        a_all = 1.0 if d < 0 else 1.0 - ramp(d, 0.15, 1.0)
        if a_all <= 0.003:
            return
        n_old = int(np.searchsorted(ts, t - 0.7, side="right")) if d < 0 else n
        n_old = min(n_old, n)
        for width, al in ((13.0, 0.16), (4.6, 0.62)):
            ctx.new_path()
            for s in range(1, n_old):
                p = par[s]
                ctx.move_to(*px[p])
                ctx.line_to(*px[s])
            ctx.set_source_rgba(*mix(BOLT, (0.35, 0.55, 0.75), 0.4), al * a_all)
            ctx.set_line_width(width)
            ctx.stroke()
        for s in range(max(1, n_old), n):
            age = t - ts[s]
            e = math.exp(-age / 0.35)
            for width, al in ((14.0, 0.22 * e + 0.08), (4.5, 0.5 + 0.5 * e)):
                ctx.new_path()
                ctx.move_to(*px[par[s]])
                ctx.line_to(*px[s])
                ctx.set_source_rgba(*BOLT, al * a_all)
                ctx.set_line_width(width)
                ctx.stroke()
        # recent forks pulse once
        for s in self.forks[:6]:
            u = (t - ts[s]) / 0.8
            if 0 <= u <= 1 and s < n and d < 0:
                ctx.new_path()
                ctx.arc(px[s][0], px[s][1], 12 + 34 * ease_out(u), 0, TAU)
                ctx.set_source_rgba(*BOLT, 0.8 * (1 - u))
                ctx.set_line_width(3)
                ctx.stroke()
        # leading tip (newest cell)
        if d < 0:
            x, y = px[n - 1]
            glow(ctx, x, y, 30, BOLT, 0.6)

    def _streamers(self, ctx, f, d, n, gap, fade):
        if fade <= 0.01:
            return
        near = self.sim["near"][max(0, n - 1)]
        for t_ in self.towers:
            k = t_["k"]
            i = self.towers.index(t_)
            g = float(gap[i])
            u = clamp(1 - g / 20.0) ** 1.2
            if d >= 0:
                u *= fade
            tipx = cx_of((t_["sc0"] + t_["sc1"]) / 2.0 + 0.5) - CELL / 2
            tipy = Y0 + t_["sr0"] * CELL
            nx, ny = cx_of(near[i][1]), cy_of(near[i][0])
            if g <= 10 and u > 0:                                     # dashed gap line to the nearest leader cell
                qx = clamp(nx, X0 + t_["c0"] * CELL, X0 + (t_["c1"] + 1) * CELL)
                qy = clamp(ny, Y0 + t_["sr0"] * CELL, Y0 + (t_["r1"] + 1) * CELL)
                ctx.save()
                ctx.set_dash([7, 7])
                ctx.new_path()
                ctx.move_to(qx, qy)
                ctx.line_to(nx, ny)
                ctx.set_source_rgba(*COL[k], 0.95 * fade)
                ctx.set_line_width(4)
                ctx.stroke()
                ctx.restore()
                ctx.new_path()
                ctx.arc(nx, ny, 17 + 3 * math.sin(f * 0.4), 0, TAU)
                ctx.set_source_rgba(*COL[k], 0.9 * fade)
                ctx.set_line_width(3.5)
                ctx.stroke()
            if u < 0.03:
                continue
            ang0 = math.atan2(ny - tipy, nx - tipx)
            L = (30 + 120 * u) * min(1.0, 0.85 * math.hypot(nx - tipx, ny - tipy) / 120.0)
            glow(ctx, tipx, tipy, 28 + 60 * u, COL[k], 0.30 * u * (1 if d < 0 else fade))
            for b in range(3):
                ang = ang0 + (b - 1) * 0.5 * (1 - 0.4 * u)
                x, y = tipx, tipy
                ctx.new_path()
                ctx.move_to(x, y)
                segs = 6
                for q in range(1, segs + 1):
                    j = (hash1(f // 3, i * 7 + b * 3 + q) - 0.5) * 22
                    x = tipx + math.cos(ang) * L * q / segs - math.sin(ang) * j * (q < segs)
                    y = tipy + math.sin(ang) * L * q / segs + math.cos(ang) * j * (q < segs)
                    ctx.line_to(x, y)
                ctx.set_source_rgba(*lighten(COL[k], 0.5), 0.9 * u * (1 if d < 0 else fade))
                ctx.set_line_width(3.2)
                ctx.stroke()

    def _strike(self, ctx, d, font):
        w = self.win
        t_ = self.towers[w]
        k = t_["k"]
        px = self.px
        # the return stroke runs up the winning channel only
        a = 1.0 - ramp(d, 0.55, 1.25)
        if a > 0.01:
            run = ramp(d, 0.0, 0.12)
            m = max(2, int(len(self.path) * run))
            ctx.new_path()
            ctx.move_to(*px[self.path[0]])
            for s in self.path[1:m]:
                ctx.line_to(*px[s])
            for wd, al, c in ((34.0, 0.18, COL[k]), (16.0, 0.45, lighten(COL[k], 0.5)), (7.0, 1.0, (1, 1, 1))):
                ctx.set_source_rgba(*c, al * a)
                ctx.set_line_width(wd)
                ctx.stroke_preserve()
            ctx.new_path()
        x0, y0, bw, bh = tower_rect(t_)
        tint = 0.9 * (1 - ramp(d, 0.45, 1.3))
        if tint > 0.01:
            draw_tower(ctx, t_, tint, 1.0)
            glow(ctx, x0 + bw / 2, y0, 240, COL[k], 0.55 * tint)
            glow(ctx, x0 + bw / 2, y0 + bh * 0.4, 160, lighten(COL[k], 0.5), 0.35 * tint)
        for dly, wd, al in ((0.0, 9.0, 0.9), (0.2, 5.0, 0.55)):
            u = max(0.0, d - dly) / 1.15
            if 0 < u < 1:
                ctx.new_path()
                ctx.arc(x0 + bw / 2, y0, 30 + 640 * ease_out(u), 0, TAU)
                ctx.set_source_rgba(*lighten(COL[k], 0.5), al * (1 - u))
                ctx.set_line_width(wd)
                ctx.stroke()

    def _labels(self, ctx, font, gap, d):
        for i, t_ in enumerate(self.towers):
            k = t_["k"]
            cx = X0 + (t_["c0"] + TW / 2) * CELL
            win = i == self.win
            a = 1.0
            badge(ctx, k, cx, 1392, 21, a)
            font.draw(ctx, NAMES[k], cx, 1430, 30, COL[k], anchor="c", stroke=INK, stroke_w=7, tracking=0.4)
            g = int(gap[i])
            font.draw(ctx, f"GAP {g}", cx, 1462, 25, CREAM, anchor="c", alpha=0.95, stroke=INK, stroke_w=6, tracking=0.5)

    def _countdown(self, ctx, font, t, d):
        if t < 3.0:
            a = 1.0 - ramp(t, 2.75, 3.0)
            cur = 3 - int(t)
            u = t - int(t)
        elif d > 0.9:
            a = ramp(d, 0.9, 1.3)                        # back to the opening look for the loop
            cur, u = 3, 0.0
        else:
            return
        if a <= 0.01:
            return
        y = 705
        rounded_rect(ctx, HX - 400, y - 126, 800, 150, 24)
        ctx.set_source_rgba(0.02, 0.06, 0.09, 0.78 * a)
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
        k1 = ramp(d, 0.18, 0.40)
        fo = 1 - ramp(d, 1.05, 1.35)
        a = k1 * fo
        if a <= 0.01:
            return
        k = self.wk
        oy = 0.0
        rounded_rect(ctx, HX - 410, 790 + oy, 820, 240, 26)
        ctx.set_source_rgba(0.02, 0.05, 0.08, 0.72 * a)
        ctx.fill()
        s = 0.8 + 0.2 * ease_out_back(k1)
        ctx.save()
        ctx.translate(HX, 910 + oy)
        ctx.scale(s, s)
        txt = NAMES[k]
        tw = font.width(txt, 120)
        badge(ctx, k, -tw / 2 - 40, -42, 36, a)
        font.draw(ctx, txt, 40, 0, 120, COL[k], anchor="c", alpha=a, stroke=INK, stroke_w=13, shadow=5)
        ctx.restore()
        font.draw(ctx, "TOWER STRUCK", HX, 985 + oy, 62, CREAM, anchor="c", alpha=a, stroke=INK, stroke_w=9, tracking=1.0)

    def _endcard(self, ctx, font, d):
        a = ramp(d, -0.6, -0.3) * (1 - ramp(d, 1.05, 1.35))
        if a <= 0.01:
            return
        size = 38.0
        size = min(size, 880.0 / font.width(END_TEXT, 1.0))
        chip(ctx, font, END_TEXT, HX, 1508, size, pad=22, h=size * 1.5)

    # ------------------------------------------------------------ sound
    def audio(self):
        rng = np.random.default_rng(self.seed + 4242)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        ts_ = np.arange(n) / sr

        def tt(dd):
            return np.arange(int(dd * sr)) / sr

        # wind bed: band-passed noise swelling slowly, and a thin rain hiss
        wind = au.noise(rng, self.dur, 150, 1400)[:n] * (0.55 + 0.45 * np.sin(TAU * ts_ / 6.5 + 1.0))
        hiss = au.noise(rng, self.dur, 2500, 9000)[:n] * 0.22
        low = au.noise(rng, self.dur, 60, 320)[:n] * 0.6
        w = (wind * 0.55 + hiss * 0.35 + low * 0.5) * 0.10
        bed[:n, 0] += w
        bed[:n, 1] += np.roll(w, 480)
        # countdown 3-2-1: three soft pings, the last one a step higher
        for j, m_ in enumerate((69, 72, 76)):
            t_ = tt(0.45)
            x = np.sin(TAU * au.hz(m_) * t_) * np.minimum(1.0, t_ / 0.004) * np.exp(-t_ / 0.12)
            au.place(fx, x, 0.0 + j, 0.26)
        # growth ticks: one very quiet high click per leader step, louder and (below) denser as it descends
        S = self.S
        for s in range(S):
            prog = s / max(1, S - 1)
            f0 = 2600 + 1700 * hash1(s, 1.0)
            x = au.tick(f0, 0.05)
            au.place(fx, x, float(self.ts[s]), 0.10 + 0.16 * prog ** 1.5, clamp((self.px[s][0] - 475) / 520, -0.8, 0.8))
        # a soft knock that speeds up as the leader descends, so there is always a readable beat
        tk = 3.4
        while tk < T_TOUCH - 0.3:
            u = clamp((tk - 3.0) / (T_TOUCH - 3.0))
            au.place(fx, au.marimba(au.hz(60 + 7 * u), 0.35), tk, 0.20 + 0.10 * u)
            tk += lerp(1.2, 0.34, u ** 1.3)
        # forks: a small two-tone pip
        for s in self.forks[:8]:
            t0 = float(self.ts[s])
            for j, m_ in enumerate((81, 86)):
                t_ = tt(0.18)
                x = np.sin(TAU * au.hz(m_) * t_) * np.minimum(1.0, t_ / 0.003) * np.exp(-t_ / 0.05)
                au.place(fx, x, t0 + 0.05 * j, 0.10)
        # charge-up: rising hiss over the last stretch, a buzz that follows the tightest gap
        dur_r = 2.0
        sw = au.swept_noise(rng, dur_r, 1500, 7500)
        au.place(fx, sw * (np.arange(len(sw)) / len(sw)) ** 2.0, T_TOUCH - dur_r, 0.34)
        rise = au.glide(au.hz(52), au.hz(88), dur_r, decay=3.0, harm=0.25)
        au.place(fx, rise * (np.arange(len(rise)) / len(rise)) ** 1.5, T_TOUCH - dur_r, 0.12)
        # strike: 30 ms full-band crack, a short snap, then thunder (brown noise, closing low-pass) and a 40-60 Hz rumble
        nc = int(0.03 * sr)
        crack = au.noise(rng, 0.03, 60, 14000)[:nc] * np.minimum(1.0, np.arange(nc) / (0.0015 * sr)) * np.linspace(1, 0.5, nc)
        au.place(fx, crack, T_TOUCH, 1.0)
        tb = tt(0.35)
        au.place(fx, au.noise(rng, 0.35, 200, 9000) * np.exp(-tb * 9.0), T_TOUCH + 0.03, 0.55)
        dth = T_TOTAL - T_TOUCH + 0.3
        brown = np.cumsum(rng.standard_normal(int(dth * sr)))
        brown -= np.convolve(brown, np.ones(2401) / 2401, mode="same")
        brown /= max(1e-9, np.abs(brown).max())
        win, hop = 2048, 512
        nb = len(brown)
        out = np.zeros(nb + win)
        wn = np.hanning(win)
        fr = np.fft.rfftfreq(win, 1 / sr)
        for i in range(0, nb, hop):
            fc = 2600.0 * (140.0 / 2600.0) ** min(1.0, i / sr / 1.6)
            g = np.exp(-(fr / fc) ** 2)
            out[i:i + win] += np.fft.irfft(np.fft.rfft(np.pad(brown, (0, win))[i:i + win] * wn) * g, win) * wn
        th = out[:nb] / 1.5
        th /= max(1e-9, np.abs(th).max())
        te = tt(dth)
        th = th * np.minimum(1.0, te / 0.01) * (0.6 * np.exp(-te / 0.9) + 0.4 * np.exp(-te / 0.3))
        au.place(fx, th, T_TOUCH + 0.05, 0.85)
        tr = tt(2.0)
        f_r = 58.0 - 14.0 * np.minimum(1.0, tr / 1.6)
        rum = (np.sin(TAU * np.cumsum(f_r) / sr) + 0.35 * np.sin(2 * TAU * np.cumsum(f_r) / sr)) * np.exp(-tr / 0.8)
        au.place(fx, rum * np.minimum(1.0, tr / 0.01), T_TOUCH, 0.60)
        mix_ = au.reverb(fx, rng, decay=1.1, wet=0.10) + bed
        out_ = mix_[:n].copy()
        out_[:tail] += mix_[n:]
        out_ = au.eq(out_, self.EQ)
        out_[:int(0.05 * sr)] *= np.linspace(0, 1, int(0.05 * sr))[:, None]
        out_[-int(0.30 * sr):] *= np.linspace(1, 0, int(0.30 * sr))[:, None]
        pk = np.abs(out_).max()
        return np.tanh(out_ * (self.DRIVE / max(1e-9, pk)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        st = self.stats
        title = "5 Towers, 1 Lightning Leader… Which Gets Struck? ⚡"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "Which tower gets struck: coral, lime, sky, violet or gold? Call it before the leader reaches the ground!\n\n"
                "A leader grows down from the cloud one cell at a time. It is a simplified model, not real lightning "
                "physics, and real lightning does not always hit the tallest object. The sky is a grid; the cloud, the "
                "leader and the ground and towers have an electric potential, and it is recomputed after every step. Each "
                "step the leader adds one cell next to itself, and the chance of a cell grows with its potential (to the power 0.7), so "
                "the leader forks and several tips race. Each tower shows its gap, the number of cells between the leader "
                "and the tower, and a streamer that grows as the leader gets close. The first tower the leader touches "
                "is struck: one soft flash, a shockwave ring, and the struck tower glows in its color. Then the sky "
                "resets for the next round.\n"
                f"Growth, picture and sound are generated by code (seed {self.seed}), with no footage, samples or stock media.\n"
                "Which tower did you pick?\n"
                "#simulation #satisfying #lightning"),
            tags=["simulation", "lightning simulation", "dielectric breakdown", "which one wins", "prediction",
                  "satisfying", "made with code", "generative animation", "lightning", "towers"],
            pinned_comment=("Coral, lime, sky, violet or gold: which tower gets struck? Pick before the leader reaches "
                            "the ground. The leader grows toward the strongest field, and the first tower it touches is hit."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of abstract towers; a toy growth model "
                                              "that could not be mistaken for real footage, people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=st,
        )
