"""MAZE FLOOD: four runners start at the bottom of one maze; water rises row by row. First one out the top wins.

Rules: the maze is a perfect maze (exactly one route between any two cells) with one exit gap in the top wall.
All four runners start in the same bottom cell and every runner moves exactly one cell per tick. Each obeys one
rule: keep a hand on the LEFT wall, keep a hand on the RIGHT wall, take the unexplored branch that points
closest to the exit (backtrack when stuck), or take a RANDOM unexplored branch (backtrack when stuck). The water
level rises row by row and speeds up on one fixed schedule for every seed. A runner whose cell center goes under
water is out. The first runner through the exit gap wins. simulate(seed) needs no rendering, so make.py scores
hundreds of seeds for drama (drama()) and renders only the best one (Scene). Nothing is edited after the fact;
the close finish comes from seed selection only.
"""
import math
import os
import random

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_in, ease_in_out, ease_out, glow, hex_rgb, lerp, lighten,
                        mix, polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
COLS_N, ROWS_N = 15, 17
CELL = 56.0
MX0, MY0 = 46.0, 530.0                     # maze top-left (px)
MX1, MY1 = MX0 + COLS_N * CELL, MY0 + ROWS_N * CELL
START_C = COLS_N // 2
P_NEW = 1.0                                # growing-tree: share of newest-cell picks
HBIAS = 3.0                                # weight of sideways carving (long floors, few stairs)
NR = 4
NAMES = ("YELLOW", "PINK", "LIME", "VIOLET")      # symbols: triangle, square, diamond, cross
RULES = ("LEFT WALL", "RIGHT WALL", "NEAREST THE EXIT", "RANDOM")
# water: level in rows above the maze floor. One curve shape for every seed; the seed draws only the tick at
# which the water reaches the top (K_FULL), and that moment is shown on screen from frame 0 as a countdown.
L0 = 0.20                                  # level(k) = L0 + (ROWS - L0) * u^3, u = k / K_FULL: it keeps speeding up
K_FULL_RANGE = (90, 260)
T0 = 0.0                                   # race starts on frame 0 (runners already moving)
T_EXIT = 20.4                              # video time of the first exit (sets the one tick length)
TD_MIN, TD_MAX = 0.10, 0.19                # allowed tick length (s): pace stays watchable
N, E, S, Wd = 0, 1, 2, 3                   # directions (row index grows downward on screen; N = up)
DC = (0, 1, 0, -1)
DR = (-1, 0, 1, 0)


def level(k, kfull):
    u = k / kfull
    return L0 + (ROWS_N - L0) * u * u * u


# ====================================================================== maze
def make_maze(rng):
    """Growing-tree maze (mix of newest and random cell): open[r][c] is a set of open directions.
    Row 0 is the top row on screen; the exit gap is in the top wall of (0, exit_c); start is (ROWS-1, START_C)."""
    op = [[set() for _ in range(COLS_N)] for _ in range(ROWS_N)]
    seen = np.zeros((ROWS_N, COLS_N), bool)
    r0, c0 = int(rng.integers(ROWS_N)), int(rng.integers(COLS_N))
    act = [(r0, c0)]
    seen[r0, c0] = True
    while act:
        i = len(act) - 1 if rng.random() < P_NEW else int(rng.integers(len(act)))
        r, c = act[i]
        dirs = [d for d in range(4) if 0 <= r + DR[d] < ROWS_N and 0 <= c + DC[d] < COLS_N
                and not seen[r + DR[d], c + DC[d]]]
        if not dirs:
            act.pop(i)
            continue
        wt = np.array([HBIAS if d in (E, Wd) else 1.0 for d in dirs])
        d = dirs[int(rng.choice(len(dirs), p=wt / wt.sum()))]
        rr, cc = r + DR[d], c + DC[d]
        op[r][c].add(d)
        op[rr][cc].add((d + 2) % 4)
        seen[rr, cc] = True
        act.append((rr, cc))
    exit_c = int(rng.integers(1, COLS_N - 1))
    op[0][exit_c].add(N)                    # the exit gap
    return op, exit_c


def bfs(op, src):
    dist = np.full((ROWS_N, COLS_N), 10 ** 6, int)
    dist[src] = 0
    q = [src]
    for r, c in q:
        for d in op[r][c]:
            rr, cc = r + DR[d], c + DC[d]
            if 0 <= rr < ROWS_N and 0 <= cc < COLS_N and dist[rr, cc] > dist[r, c] + 1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


# ====================================================================== runners
OUT = (-1, -1)                              # the cell just outside the exit gap


def run_runner(op, exit_c, rule, rng, kmax):
    """Path of one runner (list of cells, one per tick, starting at the start cell) until it leaves the maze or
    kmax ticks pass. Wall followers keep no memory; the two explorers remember where they have been."""
    cur = (ROWS_N - 1, START_C)
    path = [cur]
    if rule in (0, 1):
        h = N
        turns = (-1, 0, 1, 2) if rule == 0 else (1, 0, -1, 2)
        for _ in range(kmax):
            for t in turns:
                d = (h + t) % 4
                if d in op[cur[0]][cur[1]]:
                    break
            h = d
            if cur == (0, exit_c) and d == N:
                path.append(OUT)
                return path
            cur = (cur[0] + DR[d], cur[1] + DC[d])
            path.append(cur)
        return path
    ex, ey = exit_c, -0.5
    visited = {cur}
    stack = [cur]
    for _ in range(kmax):
        r, c = cur
        opts = []
        for d in sorted(op[r][c]):
            nxt = OUT if (r, c) == (0, exit_c) and d == N else (r + DR[d], c + DC[d])
            if nxt not in visited:
                opts.append((d, nxt))
        if opts:
            if rule == 2:
                d, nxt = min(opts, key=lambda o: (math.hypot((o[1][1] if o[1] != OUT else ex) - ex,
                                                             (o[1][0] if o[1] != OUT else ey) - ey), o[0]))
            else:
                d, nxt = opts[int(rng.integers(len(opts)))]
            if nxt == OUT:
                path.append(OUT)
                return path
            visited.add(nxt)
            stack.append(nxt)
            cur = nxt
        else:
            stack.pop()
            cur = stack[-1]
        path.append(cur)
    return path


def simulate(seed, record=False):
    rng = np.random.default_rng(seed * 7919 + 101)
    op, exit_c = make_maze(rng)
    kfull = int(rng.integers(K_FULL_RANGE[0], K_FULL_RANGE[1] + 1))
    kmax = 900
    paths = [run_runner(op, exit_c, i, np.random.default_rng(seed * 31 + 7), kmax) for i in range(NR)]
    dist = bfs(op, (0, exit_c))
    # fate of each runner: escape tick, or flood tick (cell center under water), whichever comes first
    fate = []
    for i, p in enumerate(paths):
        res = None
        for k, cell in enumerate(p):
            if cell == OUT:
                res = ("out", k)
                break
            row_up = ROWS_N - 1 - cell[0]           # rows above the floor
            if level(k, kfull) >= row_up + 0.5:
                res = ("flood", k)
                break
        if res is None:
            res = ("flood", len(p))                  # never got out inside kmax: the water gets it
        fate.append(res)
    return dict(valid=True, seed=seed, op=op, exit_c=exit_c, paths=paths, dist=dist, fate=fate, kfull=kfull)


def cell_at(p, k):
    return p[min(k, len(p) - 1)]


def steps_left(sim, i, k):
    cell = cell_at(sim["paths"][i], k)
    return 0 if cell == OUT else int(sim["dist"][cell]) + 1


def drama(sim):
    seed = sim["seed"]
    fate = sim["fate"]
    outs = sorted((k, i) for i, (f, k) in enumerate(fate) if f == "out")
    if not outs:
        return dict(seed=seed, score=-1e6, valid=False, reason="nobody escapes")
    kw, w = outs[0]
    if len(outs) > 1 and outs[1][0] == kw:
        return dict(seed=seed, score=-1e6, valid=False, reason="tie")
    td = clamp((T_EXIT - T0) / kw, TD_MIN, TD_MAX)
    t_exit = T0 + kw * td
    kf = sim["kfull"]
    margin = ROWS_N - level(kw, kf)                         # rows between the water and the exit gap
    alive = lambda i, k: fate[i][1] > k
    k80 = int(0.8 * kw)
    alive80 = sum(alive(i, k80) for i in range(NR))
    # leader = fewest steps to the exit (shortest route), among runners still in the maze
    lead_seq = []
    cur = None
    for k in range(kw):
        cand = [(steps_left(sim, i, k), i) for i in range(NR) if alive(i, k)]
        best = min(cand)[0]
        if cur is None or cur not in [i for _, i in cand] or steps_left(sim, cur, k) > best:
            cur = min(cand)[1]
        lead_seq.append(cur)
    changes = sum(1 for a, b in zip(lead_seq, lead_seq[1:]) if a != b)
    leader_half = lead_seq[kw // 2]
    floods = sorted((k, i) for i, (f, k) in enumerate(fate) if f == "flood")
    first_flood_s = floods[0][0] * td if floods else None
    # runner-up: second out, or the last one flooded
    if len(outs) > 1:
        r2, r2k, r2how = outs[1][1], outs[1][0], "out"
    else:
        r2k, r2 = max(floods)
        r2how = "flood"
    r2_steps_at_win = steps_left(sim, r2, kw) if alive(r2, kw) else None
    min_height = min(ROWS_N - 1 - cell_at(sim["paths"][w], k)[0] + 0.5 - level(k, kf) for k in range(kw))
    valid = (16.5 <= t_exit <= 23.0 and alive80 >= 2 and leader_half != w and 0.5 <= margin <= 3.0
             and (first_flood_s is None or first_flood_s >= 4.0))
    score = (10 + 3 * (w != 2) + 2.0 * min(changes, 8) + 4 * (alive80 >= 3) + 3 * (r2how == "out")
             - 3.0 * abs(margin - 1.4) + 4 * (min_height < 1.6) + 2 * ((first_flood_s or 99) > 7.0)
             - 0.5 * max(0, (r2_steps_at_win or 0) - 6) * (r2how == "flood"))
    return dict(seed=seed, score=round(float(score if valid else score - 100), 3), valid=bool(valid),
                winner=NAMES[w], winner_index=w, winner_rule=RULES[w],
                exit_tick=kw, tick_seconds=round(td, 4), exit_time_s=round(kw * td + T0, 2),
                margin_rows=round(float(margin), 2), runner_up=NAMES[r2], runner_up_fate=r2how,
                runner_up_tick=int(r2k), runner_up_steps_left_at_win=r2_steps_at_win,
                fates={NAMES[i]: dict(result=f, tick=int(k), time_s=round(k * td + T0, 2))
                       for i, (f, k) in enumerate(fate)},
                alive_at_80pct=int(alive80), leader_at_half=NAMES[leader_half], winner_led_at_half=False
                if leader_half != w else True, lead_changes=int(changes),
                first_flood_s=None if first_flood_s is None else round(first_flood_s, 2),
                winner_lowest_height_rows=round(float(min_height), 2),
                shortest_route_cells=int(sim["dist"][ROWS_N - 1, START_C]) + 1)


def score_seed(seed):
    try:
        return drama(simulate(seed))
    except Exception as e:                        # a bad seed must never stop a search
        return dict(seed=seed, score=-1e9, valid=False, error=str(e))


# ====================================================================== look
COLS = [hex_rgb(c) for c in ("#FFD23F", "#FF5FA2", "#8BEA4A", "#B98CFF")]
INK = hex_rgb("#06080D")
WHITE = (1.0, 1.0, 1.0)
STONE = hex_rgb("#E6DCC6")                 # maze walls
STEEL = hex_rgb("#8E9AAB")
GOLD = hex_rgb("#FFC24A")
AQUA = hex_rgb("#5CD3FF")
DEEP = hex_rgb("#0B3E86")
DANGER = hex_rgb("#FF5A4E")
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
HOOK = "WHO ESCAPES THE FLOOD?"
RULE = "FIRST OUT WINS"
TX = 66.0
CHIP_Y, CHIP_W, CHIP_H = 380.0, 214.0, 122.0
CHIP_X = tuple(46.0 + 224.0 * i for i in range(NR))
SLOT = ((-12, -12), (12, -12), (-12, 12), (12, 12))
MIDI = (79, 83, 86, 91)                    # step pitch per runner (G major pentatonic)
ACCENT = (2, 3, 4, 5)                      # accent every n-th step: a shifting polyrhythm


def cell_xy(cell):
    if cell == OUT:
        return None
    r, c = cell
    return MX0 + (c + 0.5) * CELL, MY0 + (r + 0.5) * CELL


def symbol(ctx, k, x, y, s):
    """Contestant mark k centered on (x, y), half-size s: triangle, square, diamond, cross."""
    ctx.new_path()
    if k == 0:
        polygon(ctx, [(x + 1.2 * s * math.cos(a), y + 0.18 * s + 1.2 * s * math.sin(a))
                      for a in (-TAU / 4, TAU / 12, TAU * 5 / 12)])
    elif k == 1:
        ctx.rectangle(x - 0.8 * s, y - 0.8 * s, 1.6 * s, 1.6 * s)
    elif k == 2:
        polygon(ctx, [(x, y - 1.15 * s), (x + 0.92 * s, y), (x, y + 1.15 * s), (x - 0.92 * s, y)])
    else:
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(TAU / 8)
        ctx.rectangle(-1.1 * s, -0.3 * s, 2.2 * s, 0.6 * s)
        ctx.rectangle(-0.3 * s, -1.1 * s, 0.6 * s, 2.2 * s)
        ctx.restore()
    ctx.fill()


def exit_arrow(ctx, x, y, s, col, a=1.0):
    """The exit mark (also used in the legend): a bold up arrow."""
    ctx.new_path()
    polygon(ctx, [(x, y - s), (x + 0.9 * s, y - 0.1 * s), (x + 0.36 * s, y - 0.1 * s), (x + 0.36 * s, y + s),
                  (x - 0.36 * s, y + s), (x - 0.36 * s, y - 0.1 * s), (x - 0.9 * s, y - 0.1 * s)])
    ctx.set_source_rgba(*col, a)
    ctx.fill()


def wave_icon(ctx, x, y, s, col, a=1.0):
    ctx.set_line_width(0.22 * s)
    ctx.set_source_rgba(*col, a)
    for dy in (-0.32 * s, 0.32 * s):
        ctx.new_path()
        for j in range(13):
            u = j / 12
            px = x - s + 2 * s * u
            py = y + dy + 0.22 * s * math.sin(u * TAU * 1.5)
            if j == 0:
                ctx.move_to(px, py)
            else:
                ctx.line_to(px, py)
        ctx.stroke()


def rule_icon(ctx, k, x, y, s, col, a=1.0):
    """Rule glyphs, no words: hand on the left wall, hand on the right wall, head for the exit, dice."""
    ctx.set_source_rgba(*col, a)
    if k in (0, 1):
        sx = -1 if k == 0 else 1
        ctx.set_line_width(0.30 * s)                          # the wall
        ctx.new_path()
        ctx.move_to(x + sx * 0.85 * s, y - s)
        ctx.line_to(x + sx * 0.85 * s, y + s)
        ctx.stroke()
        ctx.set_line_width(0.18 * s)                          # a route hugging it
        ctx.new_path()
        ctx.move_to(x + sx * 0.38 * s, y + 0.95 * s)
        ctx.line_to(x + sx * 0.38 * s, y - 0.35 * s)
        ctx.stroke()
        polygon(ctx, [(x + sx * 0.38 * s, y - 0.95 * s), (x + sx * 0.38 * s + 0.42 * s, y - 0.3 * s),
                      (x + sx * 0.38 * s - 0.42 * s, y - 0.3 * s)])
        ctx.fill()
        for j in range(3):                                    # the hand: three short touches on the wall
            yy = y - 0.5 * s + j * 0.5 * s
            ctx.new_path()
            ctx.move_to(x + sx * 0.38 * s, yy)
            ctx.line_to(x + sx * 0.66 * s, yy)
            ctx.set_line_width(0.12 * s)
            ctx.stroke()
    elif k == 2:
        ctx.set_line_width(0.14 * s)
        ctx.new_path()
        ctx.arc(x, y, 0.98 * s, 0, TAU)
        ctx.stroke()
        exit_arrow(ctx, x, y, 0.62 * s, col, a)
    else:
        rounded_rect(ctx, x - 0.92 * s, y - 0.92 * s, 1.84 * s, 1.84 * s, 0.3 * s)
        ctx.set_line_width(0.14 * s)
        ctx.stroke()
        for px, py in ((-0.45, -0.45), (0.45, -0.45), (0, 0), (-0.45, 0.45), (0.45, 0.45)):
            ctx.new_path()
            ctx.arc(x + px * s, y + py * s, 0.15 * s, 0, TAU)
            ctx.fill()


def make_static(op, exit_c):
    """Background, maze floor and walls (all fixed for the whole video)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (yy / H)[..., None]
    img = np.array([0.028, 0.034, 0.052], np.float32) * (1 - v) + np.array([0.012, 0.016, 0.026], np.float32) * v
    img *= np.clip(1.12 - 0.42 * (((xx - 470) / 820) ** 2 + ((yy - 1000) / 1100) ** 2), 0.45, 1.0)[..., None]
    hatch = (np.abs(((xx + yy) % 26) - 13) > 12.2).astype(np.float32)
    img += (hatch * 0.010)[..., None] * np.array([0.6, 0.8, 1.0], np.float32)
    del yy, xx, v, hatch
    img += np.random.default_rng(5).normal(0, 0.004, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    del img
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    ctx = cairo.Context(surf)
    tmp = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx.set_source_surface(tmp, 0, 0)
    ctx.paint()
    del tmp
    rounded_rect(ctx, MX0 - 8, MY0 - 8, MX1 - MX0 + 16, MY1 - MY0 + 16, 12)
    ctx.set_source_rgba(0.075, 0.088, 0.115, 1)
    ctx.fill()
    for r in range(ROWS_N):                       # faint floor bands, one per row: the water counts rows
        if r % 2 == 0:
            ctx.rectangle(MX0, MY0 + r * CELL, MX1 - MX0, CELL)
            ctx.set_source_rgba(1, 1, 1, 0.022)
            ctx.fill()
    sx, sy = cell_xy((ROWS_N - 1, START_C))
    ctx.new_path()
    ctx.arc(sx, sy, 24, 0, TAU)
    ctx.set_source_rgba(*STEEL, 0.3)
    ctx.set_line_width(2)
    ctx.stroke()
    segs = []
    for r in range(ROWS_N):
        for c in range(COLS_N):
            x0, y0 = MX0 + c * CELL, MY0 + r * CELL
            if N not in op[r][c]:
                segs.append((x0, y0, x0 + CELL, y0))
            if Wd not in op[r][c]:
                segs.append((x0, y0, x0, y0 + CELL))
            if c == COLS_N - 1:
                segs.append((x0 + CELL, y0, x0 + CELL, y0 + CELL))
            if r == ROWS_N - 1:
                segs.append((x0, y0 + CELL, x0 + CELL, y0 + CELL))
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for dx, dy, lw, col, a in ((3, 5, 9, INK, 0.7), (0, 0, 8, darken(STONE, 0.35), 1.0), (0, -1, 4.5, STONE, 1.0)):
        ctx.set_line_width(lw)
        ctx.set_source_rgba(*col, a)
        for x0, y0, x1, y1 in segs:
            ctx.move_to(x0 + dx, y0 + dy)
            ctx.line_to(x1 + dx, y1 + dy)
        ctx.stroke()
    surf.flush()
    return surf


class Scene:
    W, H, FPS = W, H, FPS

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed, record=True)
        self.stats = drama(self.sim)
        if "exit_tick" not in self.stats:
            raise ValueError(f"seed {seed}: nobody escapes")
        sim = self.sim
        self.kf = sim["kfull"]
        self.kw = self.stats["exit_tick"]
        self.w = self.stats["winner_index"]
        self.td = self.stats["tick_seconds"]
        self.fate = sim["fate"]
        self.t_x = self.kw * self.td + T0
        self.t_fullc = self.kf * self.td + T0                      # countdown target: water reaches the top
        kfull_vis = self.kf * ((ROWS_N + 0.35 - L0) / (ROWS_N - L0)) ** (1 / 3)
        self.t_full = max(self.t_x + 1.0, kfull_vis * self.td + T0)
        self.t_route = self.t_full + 0.15
        self.t_reset = max(self.t_route + 3.4, 22.6)
        dur = min(31.8, max(24.2, self.t_reset + 1.25))
        self.t_reset = dur - 1.25
        self.nframes = int(round(dur * FPS))
        self.dur = self.nframes / FPS
        self.kmax = int(self.dur / self.td) + 3
        # per-tick cell of every runner (frozen at its fate tick), shared-cell slots, leader
        self.cells = [[cell_at(p, min(k, self.fate[i][1])) for k in range(self.kmax + 2)]
                      for i, p in enumerate(sim["paths"])]
        self.offs = np.zeros((NR, self.kmax + 2, 2))
        for k in range(self.kmax + 2):
            occ = {}
            for i in range(NR):
                if self.fate[i][1] > k or self.fate[i][0] == "flood":
                    occ.setdefault(self.cells[i][k], []).append(i)
            for ids in occ.values():
                if len(ids) > 1:
                    for i in ids:
                        self.offs[i, k] = SLOT[i]
        self.lead = []
        cur = None
        for k in range(self.kmax + 2):
            al = [i for i in range(NR) if self.fate[i][1] > k]
            if not al:
                self.lead.append(None)
                continue
            best = min(steps_left(sim, i, k) for i in al)
            if cur not in al or steps_left(sim, cur, k) > best:
                cur = min(al, key=lambda i: (steps_left(sim, i, k), i))
            self.lead.append(cur)
        self.lead_change_k = [k for k in range(1, self.kw) if self.lead[k] != self.lead[k - 1]]
        outs = sorted((k, i) for i, (f, k) in enumerate(self.fate) if f == "out")
        self.place = {i: n for n, (k, i) in enumerate(outs)}
        self.wave_n = max(1, round(2.2 * self.dur / TAU))
        self._res = None

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_res"] = None
        return st

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf),
                             bg=make_static(self.sim["op"], self.sim["exit_c"]), font=Font(FONT_PATH))
        return self._res

    def frame(self, f):
        R = self._resources()
        self.draw(R["ctx"], f, R["font"], R["bg"])
        R["surf"].flush()
        return bytes(R["surf"].get_data())

    # ------------------------------------------------------------ state helpers
    def tick(self, t):
        return max(0.0, (t - T0) / self.td)

    def water(self, t):
        """Water level in rows; it drains back to the start level during the reset."""
        lv = min(level(self.tick(t), self.kf), ROWS_N + 0.35)
        kr = ease_in_out(ramp(t, self.t_reset, self.dur - 1 / FPS))
        return lerp(lv, L0, kr) if kr > 0 else lv

    def exit_xy(self):
        return MX0 + (self.sim["exit_c"] + 0.5) * CELL, MY0 - 0.45 * CELL

    def pos(self, i, t):
        """Screen position and heading of runner i at time t."""
        tk = self.tick(t)
        fk = self.fate[i][1]
        k = int(tk)
        u = tk - k
        if k >= fk:
            k, u = fk, 0.0
        k1 = min(k + 1, fk)
        pa = cell_xy(self.cells[i][k]) or self.exit_xy()
        pb = cell_xy(self.cells[i][k1]) or self.exit_xy()
        oa, ob = self.offs[i, k], self.offs[i, k1]
        x = lerp(pa[0] + oa[0], pb[0] + ob[0], u)
        y = lerp(pa[1] + oa[1], pb[1] + ob[1], u)
        hx, hy = pb[0] - pa[0], pb[1] - pa[1]
        kk = k
        while abs(hx) + abs(hy) < 1 and kk > 0:
            p0 = cell_xy(self.cells[i][kk - 1]) or self.exit_xy()
            p1 = cell_xy(self.cells[i][kk]) or self.exit_xy()
            hx, hy = p1[0] - p0[0], p1[1] - p0[1]
            kk -= 1
        if abs(hx) + abs(hy) < 1:
            hx, hy = 0, -1
        return x, y, math.atan2(hy, hx)

    def height(self, i, t):
        """Height of runner i's cell center above the water, in rows (cell-based, offsets ignored)."""
        tk = self.tick(t)
        fk = self.fate[i][1]
        k = min(int(tk), fk)
        u = 0.0 if k >= fk else tk - k
        ra = ROWS_N - 1 - self.cells[i][k][0] + 0.5
        rb = ROWS_N - 1 - self.cells[i][min(k + 1, fk)][0] + 0.5 if self.cells[i][min(k + 1, fk)] != OUT else ra
        return lerp(ra, rb, u) - self.water(t)

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, font, bg):
        t = f / FPS
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(bg, 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        kr = ease_in_out(ramp(t, self.t_reset, self.dur - 1 / FPS))
        self._exit(ctx, t)
        self._trails(ctx, t, 1 - ramp(t, self.t_x + 0.4, self.t_route))
        self._ghosts(ctx, t, 1 - kr)
        self._water(ctx, t)
        self._route(ctx, t, 1 - ramp(t, self.t_reset - 0.1, self.t_reset + 0.5))
        self._start_flash(ctx, t)
        self._runners(ctx, t, kr)
        self._header(ctx, t, font, kr)
        self._chips(ctx, t, font, kr)
        font.draw(ctx, f"SEED {self.seed}", MX1, MY1 + 44, 26, STEEL, anchor="r", alpha=0.55, tracking=1.0)

    def _start_flash(self, ctx, t):
        """The start gun: a glow on the start pad that is there on frame 0 and builds again into the last frame."""
        sx, sy = cell_xy((ROWS_N - 1, START_C))
        a = max(1 - ramp(t, 0.0, 0.7), ease_in(ramp(t, self.dur - 0.55, self.dur - 1 / FPS), 2))
        if a > 0.01:
            ctx.set_operator(cairo.OPERATOR_ADD)
            glow(ctx, sx, sy - 10, 190, lighten(GOLD, 0.2), 0.55 * a)
            ctx.set_operator(cairo.OPERATOR_OVER)
        if t < 0.9:
            u = t / 0.9
            ctx.new_path()
            ctx.arc(sx, sy, 26 + 210 * ease_out(u), 0, TAU)
            ctx.set_line_width(9 * (1 - u) + 2)
            ctx.set_source_rgba(*GOLD, 0.85 * (1 - u))
            ctx.stroke()

    def _exit(self, ctx, t):
        ex = self.exit_xy()[0]
        pl = 0.5 + 0.5 * math.sin(TAU * self.wave_n * 2 * t / self.dur)
        ctx.set_operator(cairo.OPERATOR_ADD)
        glow(ctx, ex, MY0 + 4, 64, GOLD, 0.22 + 0.10 * pl)
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.rectangle(ex - CELL / 2 + 7, MY0 - 5, CELL - 14, 10)
        ctx.set_source_rgba(*GOLD, 0.9)
        ctx.fill()
        exit_arrow(ctx, ex, MY0 - 22, 10, GOLD, 0.95)

    def _trails(self, ctx, t, alpha):
        if alpha <= 0.003:
            return
        tk = self.tick(t)
        for i in range(NR):
            fk = self.fate[i][1]
            if tk >= fk + 12:
                continue
            k1 = min(int(tk), fk)
            pts = []
            for k in range(max(0, k1 - 16), k1 + 1):
                p = cell_xy(self.cells[i][k])
                if p:
                    pts.append((p[0] + self.offs[i, k][0], p[1] + self.offs[i, k][1], k))
            x, y, _ = self.pos(i, t)
            pts.append((x, y, tk))
            ctx.set_line_width(7)
            fade = 1 - ramp(tk, fk, fk + 12)
            for (x0, y0, k0), (x1, y1, _) in zip(pts, pts[1:]):
                a = alpha * clamp(1 - (tk - k0) / 17) * 0.5 * fade
                if a <= 0.01:
                    continue
                ctx.new_path()
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
                ctx.set_source_rgba(*COLS[i], a)
                ctx.stroke()

    def _ghosts(self, ctx, t, alpha):
        """Runners lost to the water stay as faint marks under the surface."""
        for i in range(NR):
            how, fk = self.fate[i]
            tf = fk * self.td + T0
            if how != "flood" or t < tf:
                continue
            x, y, _ = self.pos(i, t)
            ctx.set_source_rgba(*mix(COLS[i], DEEP, 0.35), alpha * 0.55 * ramp(t, tf + 0.3, tf + 0.9))
            symbol(ctx, i, x, y, 9)

    def _water(self, ctx, t):
        lv = self.water(t)
        ys = MY1 - lv * CELL
        ph = TAU * self.wave_n * t / self.dur
        ctx.save()
        ctx.rectangle(MX0 - 6, MY0 - 6, MX1 - MX0 + 12, MY1 - MY0 + 12)
        ctx.clip()
        n = 48
        pts = []
        for j in range(n + 1):
            x = MX0 - 6 + (MX1 - MX0 + 12) * j / n
            pts.append((x, ys + 3.2 * math.sin(x * 0.030 + ph) + 2.0 * math.sin(x * 0.071 - 2 * ph + 1.3)))
        ctx.new_path()
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.line_to(MX1 + 6, MY1 + 6)
        ctx.line_to(MX0 - 6, MY1 + 6)
        ctx.close_path()
        g = cairo.LinearGradient(0, ys, 0, MY1)
        g.add_color_stop_rgba(0, *mix(AQUA, DEEP, 0.35), 0.60)
        g.add_color_stop_rgba(1, *DEEP, 0.80)
        ctx.set_source(g)
        ctx.fill()
        ctx.new_path()
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.set_line_width(4)
        ctx.set_source_rgba(*lighten(AQUA, 0.4), 0.9)
        ctx.stroke()
        if MY1 - ys > 30:                              # rising bubbles, periodic over the video (seamless loop)
            rr = random.Random(11)
            for b in range(26):
                bx = rr.uniform(MX0 + 8, MX1 - 8)
                sp = rr.choice((2, 3, 4))
                rad = rr.uniform(2.0, 4.5)
                u = (rr.random() + sp * t / self.dur * 3) % 1.0
                by = MY1 - 6 - u * (MY1 - ys - 14)
                ctx.new_path()
                ctx.arc(bx + 4 * math.sin(u * 9 + b), by, rad, 0, TAU)
                ctx.set_line_width(1.6)
                ctx.set_source_rgba(*lighten(AQUA, 0.5), 0.45 * (1 - u))
                ctx.stroke()
        ctx.restore()

    def _route(self, ctx, t, alpha):
        if t < self.t_route or alpha <= 0.003:
            return
        p = self.sim["paths"][self.w][:self.kw]
        u = ease_in_out(ramp(t, self.t_route, self.t_route + 1.4))
        m = max(2, int(round(u * len(p))))
        pts = [cell_xy(c) for c in p[:m]]
        if m >= len(p):
            pts.append((self.exit_xy()[0], MY0 - 4))
        col = COLS[self.w]
        for lw, cc, a in ((16, INK, 0.45), (9, col, 0.95), (3, lighten(col, 0.6), 0.9)):
            ctx.new_path()
            ctx.move_to(*pts[0])
            for q in pts[1:]:
                ctx.line_to(*q)
            ctx.set_line_width(lw)
            ctx.set_source_rgba(*cc, a * alpha)
            ctx.stroke()
        hx, hy = pts[-1]
        ctx.set_operator(cairo.OPERATOR_ADD)
        glow(ctx, hx, hy, 40, col, 0.8 * alpha * (1 - ramp(t, self.t_route + 1.4, self.t_route + 2.0)))
        ctx.set_operator(cairo.OPERATOR_OVER)

    def _runner_shape(self, ctx, i, x, y, ang, a, scale=1.0, ring=0.0):
        r = 15 * scale
        ctx.new_path()
        ctx.arc(x + 2, y + 3, r + 1, 0, TAU)
        ctx.set_source_rgba(*INK, 0.5 * a)
        ctx.fill()
        tip = (x + (r + 9 * scale) * math.cos(ang), y + (r + 9 * scale) * math.sin(ang))   # arrow nose
        l = (x + r * 0.8 * math.cos(ang + 1.0), y + r * 0.8 * math.sin(ang + 1.0))
        rt = (x + r * 0.8 * math.cos(ang - 1.0), y + r * 0.8 * math.sin(ang - 1.0))
        polygon(ctx, [tip, l, rt])
        ctx.set_source_rgba(*COLS[i], a)
        ctx.fill()
        ctx.new_path()
        ctx.arc(x, y, r, 0, TAU)
        ctx.set_source_rgba(*COLS[i], a)
        ctx.fill_preserve()
        ctx.set_line_width(2.5)
        ctx.set_source_rgba(*lighten(COLS[i], 0.6), a)
        ctx.stroke()
        ctx.set_source_rgba(*INK, 0.92 * a)
        symbol(ctx, i, x, y + (1.0 if i == 0 else 0.0), 6.6 * scale)
        if ring > 0:
            ctx.new_path()
            ctx.arc(x, y, r + 7 + 5 * ring, 0, TAU)
            ctx.set_line_width(3)
            ctx.set_source_rgba(*mix(WHITE, DANGER, ring), a * (0.5 + 0.5 * ring))
            ctx.stroke()

    def _runners(self, ctx, t, kr):
        tk = self.tick(t)
        for i in sorted(range(NR), key=lambda i: i == self.w):
            how, fk = self.fate[i]
            tf = fk * self.td + T0
            if kr > 0:                                # the reset: everyone back on the start pad
                sx, sy = cell_xy((ROWS_N - 1, START_C))
                a = ease_out(ramp(t, self.t_reset + 0.45, self.dur - 1 / FPS))
                if a > 0.003:
                    self._runner_shape(ctx, i, sx + self.offs[i, 0][0], sy + self.offs[i, 0][1], -TAU / 4, a,
                                       0.6 + 0.4 * a)
            if t >= self.t_reset + 0.45:
                continue
            x, y, ang = self.pos(i, t)
            fa = 1 - ramp(t, self.t_reset - 0.1, self.t_reset + 0.45)
            if how == "flood" and t >= tf:
                dt = t - tf
                if dt < 0.9:
                    ctx.new_path()
                    ctx.arc(x, y, 16 + 40 * ease_out(dt / 0.9), 0, TAU)
                    ctx.set_line_width(3)
                    ctx.set_source_rgba(*lighten(AQUA, 0.5), 0.9 * (1 - dt / 0.9) * fa)
                    ctx.stroke()
                    self._runner_shape(ctx, i, x, y + 10 * dt, ang, (1 - dt / 0.9) ** 2 * fa, 1 - 0.3 * dt)
                continue
            if how == "out" and t >= tf:
                dt = t - tf
                if dt < 0.8:
                    ctx.set_operator(cairo.OPERATOR_ADD)
                    glow(ctx, x, y, 30 + 90 * ease_out(dt / 0.8), COLS[i], 0.7 * (1 - dt / 0.8))
                    ctx.set_operator(cairo.OPERATOR_OVER)
                    ctx.new_path()
                    ctx.arc(x, y, 18 + 70 * ease_out(dt / 0.8), 0, TAU)
                    ctx.set_line_width(4)
                    ctx.set_source_rgba(*COLS[i], (1 - dt / 0.8) * fa)
                    ctx.stroke()
                if i == self.w:
                    self._runner_shape(ctx, i, x, y, -TAU / 4, fa, 1.0 + 0.15 * ease_out(ramp(dt, 0, 0.3)))
                else:
                    self._runner_shape(ctx, i, x + (18 if i % 2 else -18) * ease_out(ramp(dt, 0, 0.4)), y + 4,
                                       -TAU / 4, fa * (1 - ramp(dt, 0.6, 1.2)), 0.85)
                continue
            h = self.height(i, t)
            ring = clamp((1.1 - h) / 0.9) * (0.5 + 0.5 * math.sin(t * TAU * 3.0)) * ramp(t, 1.5, 2.5) if h < 1.1 else 0.0
            self._runner_shape(ctx, i, x, y, ang, fa, 1.0, ring)
            k = int(tk)
            if k < len(self.lead) and self.lead[k] == i and t < self.t_x:
                ctx.new_path()
                ctx.arc(x, y, 24, 0, TAU)
                ctx.set_line_width(2.5)
                ctx.set_source_rgba(*GOLD, 0.85 * fa)
                ctx.stroke()

    def _header(self, ctx, t, font, kr):
        ha = 1 - ramp(t, self.t_x + 0.2, self.t_x + 0.6) * (1 - ramp(t, self.t_reset + 0.35, self.t_reset + 0.8))
        ra = ramp(t, self.t_x + 0.6, self.t_x + 1.0) * (1 - ramp(t, self.t_reset - 0.2, self.t_reset + 0.4))
        if ha > 0.003:
            font.draw(ctx, HOOK, TX, 258, 84, WHITE, alpha=ha, tracking=0.8)
            font.draw(ctx, RULE, TX, 326, 54, GOLD, alpha=ha, tracking=1.2)
            # the water countdown: when the water reaches the top of the maze (fixed from frame 0)
            left = max(0.0, self.t_fullc - t) if kr <= 0 else self.t_fullc - T0
            txt = f"TOP FLOODS IN {left:.1f} S" if left > 0 else "TOP IS UNDER WATER"
            urgent = 0 < left < 5 and kr <= 0
            col = mix(AQUA, WHITE, (0.5 + 0.5 * math.sin(t * TAU * 2)) if urgent else 0.0)
            font.draw(ctx, txt, 950, 326, 38, col, anchor="r", alpha=ha, tracking=1.0)
            wave_icon(ctx, 950 - font.width(txt, 38, 1.0) - 30, 312, 17, col, ha)
        if ra > 0.003:
            w = self.w
            col = COLS[w]
            txt = f"{NAMES[w]} IS FIRST OUT"
            size = min(84.0, 84.0 * 800 / max(1e-6, font.width(txt, 84, 0.8)))
            ctx.set_source_rgba(*col, ra)
            symbol(ctx, w, TX + 0.30 * size, 258 - 0.36 * size, 0.30 * size)
            font.draw(ctx, txt, TX + 0.78 * size, 258, size, col, alpha=ra, tracking=0.8)
            m = self.stats["margin_rows"]
            font.draw(ctx, f"THE WATER WAS {m:.1f} ROWS BEHIND", TX, 326, 50, WHITE, alpha=ra, tracking=1.0)

    def _chips(self, ctx, t, font, kr):
        tk = self.tick(t)
        k = int(tk)
        la = 1 - ramp(t, self.t_x + 0.2, self.t_x + 0.6) * (1 - kr)
        if la > 0.003:                              # legend line for the two chip numbers
            wave_icon(ctx, TX + 4, 357, 13, AQUA, 0.9 * la)
            font.draw(ctx, "HEIGHT ABOVE WATER", TX + 26, 367, 26, STEEL, alpha=la, tracking=1.2)
            x2 = TX + 26 + font.width("HEIGHT ABOVE WATER", 26, 1.2) + 44
            exit_arrow(ctx, x2, 356, 10, GOLD, 0.9 * la)
            font.draw(ctx, "STEPS TO EXIT", x2 + 18, 367, 26, STEEL, alpha=la, tracking=1.2)
        pu = 0.5 + 0.5 * math.sin(t * TAU * 3.0)
        for i in range(NR):
            x0, y0 = CHIP_X[i], CHIP_Y
            how, fk = self.fate[i]
            tf = fk * self.td + T0
            done = t >= tf and kr <= 0
            lead = k < len(self.lead) and self.lead[k] == i and t < self.t_x and not done and kr < 0.5
            dim = 0.45 if (done and how == "flood") else 1.0
            rounded_rect(ctx, x0, y0, CHIP_W, CHIP_H, 14)
            ctx.set_source_rgba(*mix(INK, COLS[i], 0.13 * dim), 0.93)
            ctx.fill_preserve()
            ctx.set_line_width(4.5 if lead else 2)
            ctx.set_source_rgba(*(GOLD if lead else mix(STEEL, COLS[i], 0.4)), 1.0 if lead else 0.55 * dim)
            ctx.stroke()
            ctx.new_path()
            ctx.arc(x0 + 32, y0 + 34, 19, 0, TAU)
            ctx.set_source_rgba(*COLS[i], dim)
            ctx.fill()
            ctx.set_source_rgba(*INK, 0.92)
            symbol(ctx, i, x0 + 32, y0 + (35 if i == 0 else 34), 8.5)
            rule_icon(ctx, i, x0 + 88, y0 + 34, 21, mix(WHITE, COLS[i], 0.35), 0.95 * dim)
            if lead:
                font.draw(ctx, "LEADS", x0 + CHIP_W - 12, y0 + 45, 28, GOLD, anchor="r", tracking=1.5)
            if done and how == "flood":
                wave_icon(ctx, x0 + 30, y0 + 90, 15, AQUA, 0.9)
                font.draw(ctx, "FLOODED", x0 + 56, y0 + 106, 40, mix(AQUA, WHITE, 0.3), alpha=0.95, tracking=1.0)
                continue
            if done and how == "out":
                pl = self.place[i]
                txt = ("1ST OUT", "2ND OUT", "3RD OUT", "4TH OUT")[pl]
                font.draw(ctx, txt, x0 + 16, y0 + 106, 46, COLS[i] if pl == 0 else WHITE, tracking=1.0)
                continue
            h = max(0.0, self.height(i, T0 if kr > 0 else t))
            st = steps_left(self.sim, i, 0 if kr > 0 else k)
            hot = h < 1.1 and kr <= 0 and t > 1.5
            hc = mix(WHITE, DANGER, clamp((1.1 - h) / 0.8) * (0.4 + 0.6 * pu) * ramp(t, 1.5, 2.5)) if hot else WHITE
            wave_icon(ctx, x0 + 26, y0 + 89, 12, AQUA, 0.9)
            font.draw(ctx, f"{h:.1f}", x0 + 44, y0 + 106, 48, hc, tracking=0.5)
            sw = font.width(f"{st}", 38, 0.5)
            font.draw(ctx, f"{st}", x0 + CHIP_W - 14, y0 + 106, 38, mix(GOLD, WHITE, 0.5), anchor="r", tracking=0.5)
            exit_arrow(ctx, x0 + CHIP_W - 26 - sw, y0 + 91, 9, GOLD, 0.9)

    # ------------------------------------------------------------ sound
    def audio(self):
        """Water bed (low-passed noise with a slosh that speeds up with the water, a sine drone whose pitch
        follows the level, bright bubbles that get denser), one step tick per runner at its own pitch with
        accents on different step counts (a polyrhythm that thins as runners are lost), a bell on every lead
        change, a downward blip when a runner floods, a rising arpeggio for the escape, and a drain into the loop."""
        sr = au.SR
        n, tail = int(round(self.dur * sr)), 3 * sr
        rng = np.random.default_rng(self.seed + 4242)
        wet = np.zeros((n + tail, 2))
        dry = np.zeros((n + tail, 2))
        td = self.td
        # ---- water bed
        hop = 240
        tt = np.arange(0, n, hop) / sr
        lv = np.array([self.water(x) for x in tt])
        rate = np.abs(np.gradient(lv, tt))                         # rows per second
        rn = np.clip(rate / 2.5, 0, 1)
        lvs = np.interp(np.arange(n) / sr, tt, lv)
        rns = np.interp(np.arange(n) / sr, tt, rn)
        fdr = 55.0 * 2 ** (np.clip(lvs, 0, ROWS_N) / ROWS_N)
        ph = TAU * np.cumsum(fdr) / sr
        drone = (np.sin(ph) + 0.45 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph) + 0.12 * np.sin(5 * ph))
        drone *= 0.05 + 0.05 * np.clip(lvs / ROWS_N, 0, 1)
        sl = 0.7 + 0.3 * np.sin(TAU * np.cumsum(0.5 + 2.0 * rns) / sr)
        for ch in (0, 1):
            low = au.bandpass(rng.standard_normal(n), 90, 800) * (0.015 + 0.08 * rns) * sl
            hi = au.bandpass(rng.standard_normal(n), 2200, 6500) * (0.004 + 0.014 * rns)
            dry[:n, ch] += low + hi + drone * 0.4
        tb = 0.0
        while tb < self.dur - 0.05:                                 # bubbles: denser as the water speeds up
            r_ = float(np.interp(tb, tt, rn))
            f0 = rng.uniform(500, 900)
            au.place(wet, au.glide(f0, f0 * rng.uniform(1.6, 2.4), rng.uniform(0.03, 0.06), 0.02, 0.1), tb,
                     rng.uniform(0.05, 0.10), float(rng.uniform(-0.7, 0.7)))
            tb += rng.exponential(1.0 / (2.0 + 14.0 * r_))
        # ---- steps (the race), lead changes, floods, escapes
        steps_end = min(self.t_reset, self.dur)
        for k in range(1, self.kmax):
            tk = T0 + k * td
            if tk >= steps_end:
                break
            for i in range(NR):
                if self.fate[i][1] < k:
                    continue
                p = cell_xy(self.cells[i][k]) or self.exit_xy()
                pan = float(np.clip((p[0] - 466) / 520, -0.8, 0.8))
                au.place(dry, au.tick(au.hz(MIDI[i]), 0.05), tk, 0.06, pan)
                if k % ACCENT[i] == 0:
                    au.place(dry, au.marimba(au.hz(MIDI[i] - 12), 0.25), tk, 0.30, pan)
        for k in self.lead_change_k:
            ld = self.lead[k]
            au.place(wet, au.bell(au.hz(MIDI[ld] + 5), 0.9), T0 + k * td, 0.10)
        penta = (67, 71, 74, 79, 83, 86, 91, 95)
        for i, (how, fk) in enumerate(self.fate):
            tf = T0 + fk * td
            if tf >= self.t_reset:
                continue
            p = cell_xy(self.cells[i][fk]) or self.exit_xy()
            pan = float(np.clip((p[0] - 466) / 520, -0.8, 0.8))
            if how == "flood":
                au.place(dry, au.glide(900, 170, 0.45, harm=0.25), tf, 0.26, pan)
                bt = np.arange(int(0.35 * sr)) / sr
                au.place(dry, au.noise(rng, 0.35, 300, 2600) * np.exp(-bt / 0.09), tf, 0.20, pan)
            else:
                big = i == self.w
                for j, m in enumerate(penta if big else penta[2:5]):
                    au.place(wet, au.bell(au.hz(m), 1.2), tf + j * (0.06 if big else 0.08), 0.22 if big else 0.14,
                             float(np.clip(pan + 0.1 * (j - 3), -0.8, 0.8)))
                au.place(dry, au.glide(400, 1600, 0.35, harm=0.2), tf, 0.10 if big else 0.06, pan)
                if big:
                    au.place(dry, au.clap(rng, 0.16), tf, 0.6, pan)
        # countdown clicks in the last 5 s before the top floods
        tc = max(0.0, self.t_fullc - 5.0)
        while tc < min(self.t_fullc, self.t_reset):
            au.place(dry, au.tick(1180, 0.05), tc, 0.14)
            tc += 0.5
        # the maze fills: a low swell and a thump
        au.place(dry, au.boom(rng, 1.4, 80, 34), self.t_full, 0.35)
        sw = 1.2
        au.place(dry, au.swept_noise(rng, sw, 300, 2200) * np.linspace(0.2, 1, int(sw * sr)), self.t_full - sw, 0.25)
        # the winner's route being drawn, then a held chord
        for j in range(12):
            au.place(dry, au.bell(au.hz(penta[j % len(penta)] + 12 * (j // len(penta))), 0.6),
                     self.t_route + 1.4 * j / 12, 0.12, (j - 6) / 8)
        td_ = self.t_route + 1.7                  # drips while the result holds
        while td_ < self.t_reset - 0.1:
            f0 = rng.uniform(900, 1400)
            au.place(dry, au.glide(f0, f0 * 0.55, 0.07, 0.03, 0.15), td_, 0.22, float(rng.uniform(-0.6, 0.6)))
            td_ += rng.uniform(0.55, 0.95)
        chord = (55, 62, 67, 71, 74)
        hold = self.t_reset + 0.6 - (self.t_x + 0.6)
        for m, g in zip(chord, (0.5, 0.4, 0.35, 0.25, 0.2)):
            au.place(wet, au.pad(au.hz(m), hold, 0.4, 0.9), self.t_x + 0.6, 0.05 * g)
        # the drain back into the loop
        dl = self.dur - self.t_reset
        env = np.sin(np.linspace(0, np.pi, int(dl * sr))) ** 1.5
        au.place(dry, au.swept_noise(rng, dl, 3200, 260) * env, self.t_reset, 0.40)
        au.place(wet, au.glide(660, 196, dl) * env, self.t_reset, 0.05)
        mix_ = au.reverb(wet, rng, decay=1.2, wet=0.22) + dry
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]                     # tails past the end wrap to the start: seamless loop
        edge = np.minimum(1.0, np.minimum(np.arange(n), n - np.arange(n)) / (0.01 * sr))
        out *= edge[:, None]
        return au.eq(out, ((40, -10.0), (90, -4.0), (250, -1.0), (700, 0.0), (1800, 2.0), (4000, 3.0),
                           (9000, 2.0), (16000, -2.0)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        title = "The Water Rises Faster Every Second. Who Gets Out First?"
        assert len(title) <= 60
        st = dict(self.stats)
        st["winner_distribution"] = winner_distribution(400)
        return dict(
            title=title,
            description=(
                "Four runners start in the same bottom cell of one maze, and the only way out is a gap in the top "
                "wall. Every runner moves exactly one cell per tick and follows one rule: yellow (triangle) keeps a "
                "hand on the left wall, pink (square) keeps a hand on the right wall, lime (diamond) always takes "
                "the unexplored branch that points closest to the exit, and violet (cross) takes a random "
                "unexplored branch at every junction. Lime and violet remember where they have been and back out "
                "of dead ends; the wall followers just never let go of their wall. Meanwhile the water rises row "
                "by row and keeps speeding up, and the countdown at the top shows from the first frame when it "
                "will reach the top. A runner is out the moment the water covers the middle of its cell. The first "
                "runner through the exit wins. Each runner's card shows its height above the water and how many "
                "steps the shortest route to the exit still needs.\n"
                f"The maze, the water timing and every move come from seed {self.seed}; the close race comes from "
                "searching seeds for one, not from editing the run. Picture and sound are generated by code, with "
                "no footage, samples or stock media.\n"
                "Which rule would you bet on?\n"
                "#simulation #satisfying #maze"),
            tags=["simulation", "maze", "maze solver", "flood", "wall follower", "prediction", "satisfying",
                  "made with code"],
            pinned_comment=("Left wall, right wall, nearest the exit or random: which runner gets out first? "
                            "\U0001F7E1 \U0001FA77 \U0001F7E2 \U0001F7E3 Pick before the water reaches the second "
                            "row."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of abstract shapes; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.dur, 3),
            drama=st,
        )


def winner_distribution(n):
    """Who gets out first over seeds 1..n (every seed, and only the seeds that pass the drama gate)."""
    allc, valc = {}, {}
    for s in range(1, n + 1):
        d = score_seed(s)
        key = d.get("winner", "NOBODY")
        allc[key] = allc.get(key, 0) + 1
        if d.get("valid"):
            valc[key] = valc.get(key, 0) + 1
    return dict(seeds=n, first_out_all_seeds=allc, first_out_valid_seeds=valc,
                note="rule of each color: " + ", ".join(f"{a} = {b}" for a, b in zip(NAMES, RULES)))
