"""TERRITORY BATTLE: four balls paint a tiled arena; most tiles when the timer ends wins.

simulate(seed) is deterministic and needs no rendering, so make.py can score hundreds of
seeds for drama (drama()) and render only the best one (Scene).
"""
import bisect
import math
import os
import random
from array import array

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_in, ease_in_out, ease_out, ease_out_back, glow, hex_rgb,
                        lerp, lighten, mix, polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
PLAY = 25.0                  # battle length, seconds
T_REVEAL = 25.85             # final 3 s: buzzer and tile count (tension), then the winner reveal
T_LINE = 26.35               # "DID YOU CALL IT?"
T_RESET = 27.45              # board rewinds so the last frame cuts back to the first
DUR = 28.0
NF = int(DUR * FPS)
NPLAY = int(PLAY * FPS)
SUB = 4                      # physics substeps per frame
DT = 1.0 / (FPS * SUB)

A = 810.0                    # arena side in arena units (= pixels at zoom 1)
N = 18                       # tiles per side
TILE = A / N
CORE = TILE * 0.25           # a tile is painted once the ball overlaps its middle half
R0 = 34.0
SPEED = 450.0
LAUNCH_K, LAUNCH_T = 1.8, 1.4   # balls start this much faster and settle to SPEED over LAUNCH_T seconds
GROW_R, GROW_T = 56.0, 5.0
FAST_K, FAST_T = 1.75, 4.5
BOMB_R, BOMB_V = 3.3 * TILE, 700.0   # paint-bomb radius and ripple speed
PICK_R, PICK_LIFE = 26.0, 5.5
KINDS = ("grow", "fast", "bomb")


# ====================================================================== simulation
def simulate(seed, record=False):
    rng = random.Random(seed)
    spots = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    rng.shuffle(spots)
    bx = [s[0] * A for s in spots]
    by = [s[1] * A for s in spots]
    vx, vy = [], []
    for _ in range(4):
        a = rng.randrange(4) * math.pi / 2 + rng.uniform(0.35, 1.22)
        vx.append(SPEED * LAUNCH_K * math.cos(a))
        vy.append(SPEED * LAUNCH_K * math.sin(a))
    rad = [R0] * 4
    grow_until = [-1.0] * 4
    fast_until = [-1.0] * 4
    owner = [-1] * (N * N)
    prev = [-1] * (N * N)
    ptime = [-9.0] * (N * N)
    counts = [0] * 4

    kinds = list(KINDS) + [rng.choice(KINDS), rng.choice(KINDS)]
    rng.shuffle(kinds)
    sched = [(3.5 + 4.3 * k + rng.uniform(-0.7, 0.7), kinds[k]) for k in range(5)]
    picks = []        # dict(kind, x, y, t0, t1, by)   t1 = time collected or expired
    live = []
    pending = []      # delayed bomb paints (t, tile, ball)
    walls, hits = [], []
    last_hit = {}
    counts_hist = []
    rec = dict(owner=[], prev=[], ptime=[], pos=[]) if record else None

    def paint(idx, b, t):
        o = owner[idx]
        if o == b:
            return
        if o >= 0:
            counts[o] -= 1
        counts[b] += 1
        prev[idx], owner[idx], ptime[idx] = o, b, t

    def snapshot():
        counts_hist.append(tuple(counts))
        if record:
            rec["owner"].append(bytes(o + 1 for o in owner))
            rec["prev"].append(bytes(o + 1 for o in prev))
            rec["ptime"].append(array("f", ptime))

    def wall(i, nx, ny, t):
        vn = vx[i] * nx + vy[i] * ny
        vx[i] -= 2 * vn * nx
        vy[i] -= 2 * vn * ny
        sp = math.hypot(vx[i], vy[i])
        a = math.atan2(vy[i], vx[i]) + rng.uniform(-0.14, 0.14)   # breaks up periodic orbits
        if abs(math.cos(a)) < 0.24 or abs(math.sin(a)) < 0.24:     # no dull axis-parallel shuttling
            diag = (math.floor(a / (math.pi / 2)) + 0.5) * (math.pi / 2)
            a += 0.22 if diag > a else -0.22
        vx[i], vy[i] = sp * math.cos(a), sp * math.sin(a)
        vn = vx[i] * nx + vy[i] * ny
        if vn < 0:
            vx[i] -= 2 * vn * nx
            vy[i] -= 2 * vn * ny
        walls.append((t, i, bx[i] - nx * rad[i], by[i] - ny * rad[i], nx, ny))

    for step in range(NPLAY * SUB):
        t = step * DT
        if step % SUB == 0:
            snapshot()
        if record:
            rec["pos"].append((bx[0], by[0], rad[0], bx[1], by[1], rad[1],
                               bx[2], by[2], rad[2], bx[3], by[3], rad[3]))
        # pickups appear / expire
        if sched and t >= sched[0][0]:
            _, kind = sched.pop(0)
            for _try in range(60):
                px, py = rng.uniform(0.16, 0.84) * A, rng.uniform(0.16, 0.84) * A
                if all(math.hypot(px - bx[i], py - by[i]) > 150 for i in range(4)) and \
                        all(math.hypot(px - p["x"], py - p["y"]) > 150 for p in live):
                    break
            p = dict(kind=kind, x=px, y=py, t0=t, t1=None, by=-1)
            picks.append(p)
            live.append(p)
        for p in live[:]:
            if t - p["t0"] > PICK_LIFE:
                p["t1"] = t
                live.remove(p)
        while pending and pending[0][0] <= t:
            _, idx, b = pending.pop(0)
            paint(idx, b, t)

        order = [(step + k) % 4 for k in range(4)]      # rotate so no colour has priority
        for i in order:
            rt = GROW_R if t < grow_until[i] else R0
            rad[i] += (rt - rad[i]) * min(1.0, DT * 8)
            sp = math.hypot(vx[i], vy[i])
            target = SPEED * (FAST_K if t < fast_until[i] else 1.0) * (1 + (LAUNCH_K - 1) * max(0.0, 1 - t / LAUNCH_T))
            k = 1 + (target / sp - 1) * min(1.0, DT * 5)
            vx[i] *= k
            vy[i] *= k
            bx[i] += vx[i] * DT
            by[i] += vy[i] * DT
            r = rad[i]
            if bx[i] < r:
                bx[i] = r
                if vx[i] < 0:
                    wall(i, 1, 0, t)
            elif bx[i] > A - r:
                bx[i] = A - r
                if vx[i] > 0:
                    wall(i, -1, 0, t)
            if by[i] < r:
                by[i] = r
                if vy[i] < 0:
                    wall(i, 0, 1, t)
            elif by[i] > A - r:
                by[i] = A - r
                if vy[i] > 0:
                    wall(i, 0, -1, t)
        # ball-ball: elastic, mass ~ area
        for i in range(3):
            for j in range(i + 1, 4):
                dx, dy = bx[j] - bx[i], by[j] - by[i]
                d = math.hypot(dx, dy)
                rs = rad[i] + rad[j]
                if d >= rs or d < 1e-6:
                    continue
                nx, ny = dx / d, dy / d
                mi, mj = rad[i] ** 2, rad[j] ** 2
                rel = (vx[i] - vx[j]) * nx + (vy[i] - vy[j]) * ny
                if rel > 0:
                    imp = 2 * rel / (mi + mj)
                    vx[i] -= imp * mj * nx
                    vy[i] -= imp * mj * ny
                    vx[j] += imp * mi * nx
                    vy[j] += imp * mi * ny
                    if t - last_hit.get((i, j), -1) > 0.12:
                        hits.append((t, i, j, bx[i] + nx * rad[i], by[i] + ny * rad[i], rel))
                    last_hit[(i, j)] = t
                push = rs - d
                bx[i] -= nx * push * mj / (mi + mj)
                by[i] -= ny * push * mj / (mi + mj)
                bx[j] += nx * push * mi / (mi + mj)
                by[j] += ny * push * mi / (mi + mj)
        for i in order:
            r = rad[i]
            bx[i] = min(max(bx[i], r), A - r)
            by[i] = min(max(by[i], r), A - r)
            # pickups
            for p in live[:]:
                if math.hypot(bx[i] - p["x"], by[i] - p["y"]) < r + PICK_R:
                    p["t1"], p["by"] = t, i
                    live.remove(p)
                    if p["kind"] == "grow":
                        grow_until[i] = t + GROW_T
                    elif p["kind"] == "fast":
                        fast_until[i] = t + FAST_T
                    else:
                        for ty in range(N):
                            for tx in range(N):
                                dd = math.hypot((tx + 0.5) * TILE - p["x"], (ty + 0.5) * TILE - p["y"])
                                if dd <= BOMB_R:
                                    pending.append((t + dd / BOMB_V, ty * N + tx, i))
                        pending.sort()
            # paint
            x, y = bx[i], by[i]
            r2 = r * r
            for ty in range(max(0, int((y - r) / TILE)), min(N - 1, int((y + r) / TILE)) + 1):
                ddy = abs(y - (ty + 0.5) * TILE) - CORE
                if ddy < 0:
                    ddy = 0.0
                for tx in range(max(0, int((x - r) / TILE)), min(N - 1, int((x + r) / TILE)) + 1):
                    ddx = abs(x - (tx + 0.5) * TILE) - CORE
                    if ddx < 0:
                        ddx = 0.0
                    if ddx * ddx + ddy * ddy <= r2:
                        paint(ty * N + tx, i, t)
    snapshot()                           # a bomb ripple still spreading at the buzzer does not count
    if record:
        rec["pos"].append((bx[0], by[0], rad[0], bx[1], by[1], rad[1],
                           bx[2], by[2], rad[2], bx[3], by[3], rad[3]))
    for p in picks:
        if p["t1"] is None:
            p["t1"] = PLAY
    out = dict(seed=seed, counts=counts_hist, picks=picks, walls=walls, hits=hits,
               start=[(s[0] * A, s[1] * A) for s in spots],
               grow=[], fast=[])
    for p in picks:
        if p["by"] >= 0 and p["kind"] in ("grow", "fast"):
            out[p["kind"]].append((p["t1"], p["by"]))
    if record:
        out.update(rec)
    return out


# ====================================================================== drama scoring
def drama(sim):
    """Score one simulated run. Higher = more worth watching."""
    counts = sim["counts"]
    hold = int(0.4 * FPS)                 # a lead only counts once it has lasted 0.4 s
    leaders = []
    cur = -1
    for c in counts:
        m = max(c)
        if m > 0 and (cur < 0 or c[cur] < m):
            cur = c.index(m)
        leaders.append(cur)
    runs = []                             # [leader, first_frame, length]
    for f, l in enumerate(leaders):
        if runs and runs[-1][0] == l:
            runs[-1][2] += 1
        else:
            runs.append([l, f, 1])
    solid = []
    for r in runs:
        if r[0] < 0 or (r[2] < hold and r is not runs[-1]):
            continue
        if solid and solid[-1][0] == r[0]:
            continue
        solid.append(r)
    final = counts[-1]
    rank = sorted(range(4), key=lambda i: -final[i])
    winner = rank[0]
    margin = final[rank[0]] - final[rank[1]]
    lead_changes = max(0, len(solid) - 1)
    half_leader = leaders[NPLAY // 2]
    last_take = solid[-1][1] / FPS if solid and solid[-1][0] == winner else PLAY
    low = min(min(c) for c in counts[6 * FPS:]) / (N * N)     # weakest share after 6 s
    deficit = max(max(c[i] for i in range(4) if i != winner) - c[winner] for c in counts[4 * FPS:])
    gaps = [sorted(c)[-1] - sorted(c)[-2] for c in counts[-5 * FPS:]]
    tight = sum(gaps) / len(gaps)                             # mean 1st-2nd gap over the last 5 s
    pct = [round(100 * c / (N * N)) for c in final]
    score = 0.0
    score += 1.0 * min(lead_changes, 8)
    if margin < 1:
        score -= 50.0                                         # ties are not a result
    else:
        score += 4.0 * (0.8 if margin == 1 else math.exp(-max(0, margin - 4) / 5.0))
    score += 3.0 if half_leader != winner else 0.0
    # winner takes over late, but early enough that the viewer sees it happen
    score += 2.5 * min(1.0, max(0.0, (last_take - (PLAY - 10)) / 6.0)) * (1.0 if last_take <= PLAY - 0.8 else 0.4)
    score += 1.0 * (len({r[0] for r in solid}) - 1)           # how many colours ever led
    score += 2.0 * min(1.0, max(0, deficit) / 30.0)           # winner came from behind
    score += 1.5 * (1 - min(1.0, tight / 12.0))               # neck and neck at the end
    score += 0.5 if pct[rank[0]] != pct[rank[1]] else 0.0     # scoreboard percentages differ
    score -= 40.0 * max(0.0, 0.10 - low)                      # nobody gets wiped out early
    score += 0.5 * sum(1 for p in sim["picks"] if p["by"] >= 0)
    return dict(
        seed=sim["seed"], score=round(score, 3), winner=NAMES[winner], winner_index=winner,
        final_tiles=dict(zip(NAMES, final)),
        final_percent=dict(zip(NAMES, (round(100 * c / (N * N), 2) for c in final))),
        margin_tiles=margin, lead_changes=lead_changes,
        leader_at_half=NAMES[half_leader] if half_leader >= 0 else None,
        winner_led_at_half=half_leader == winner,
        winner_took_final_lead_at_s=round(last_take, 2),          # ignoring leads shorter than 0.4 s
        last_lead_flip_at_s=round(runs[-1][1] / FPS, 2),            # including split-second flips
        colours_that_led=len({r[0] for r in solid}),
        lowest_share_after_6s_percent=round(100 * low, 2),
        biggest_deficit_winner_overcame_tiles=max(0, deficit),
        mean_top2_gap_last_5s_tiles=round(tight, 2),
        pickups_collected=[dict(kind=p["kind"], by=NAMES[p["by"]], at_s=round(p["t1"], 2))
                           for p in sim["picks"] if p["by"] >= 0],
    )


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
COLS = [hex_rgb(c) for c in ("#FFD23F", "#29C7FF", "#7C5CFF", "#FF3355")]   # checked under CVD simulation
NAMES = ("YELLOW", "BLUE", "VIOLET", "RED")                                 # symbols: triangle, ring, plus, diamond
INK = hex_rgb("#070A18")
WHITE = (1.0, 1.0, 1.0)
NEUTRAL = (0.22, 0.32, 0.85)
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")

AX, AY = 135.0, 318.0        # arena top-left on screen at zoom 1
HOOK_ZOOM = 0.84             # arena starts smaller to make room for the big question
T_HOOK = 2.0                 # question compacts, arena zooms in
POP = 0.28                   # tile pop-in time
BOARD_Y, ROW_H, ROW_GAP = 1162.0, 82.0, 10.0
T_COUNT, COUNT_LEN = PLAY + 0.13, 0.60
LABEL = dict(grow="GROW!", fast="SPEED!", bomb="BOMB!")


def symbol(ctx, k, x, y, s, rot=0.0):
    """Path of colour k's symbol (triangle, ring, plus, diamond) centred on (x, y)."""
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    if k == 0:
        polygon(ctx, [(s * 1.15 * math.cos(a), s * 1.15 * math.sin(a))
                      for a in (-TAU / 4, -TAU / 4 + TAU / 3, -TAU / 4 + 2 * TAU / 3)])
    elif k == 1:
        ctx.new_sub_path()
        ctx.arc(0, 0, s, 0, TAU)
        ctx.new_sub_path()
        ctx.arc_negative(0, 0, s * 0.5, TAU, 0)
    elif k == 2:
        a, b = s * 0.36, s
        polygon(ctx, [(-a, -b), (a, -b), (a, -a), (b, -a), (b, a), (a, a),
                      (a, b), (-a, b), (-a, a), (-b, a), (-b, -a), (-a, -a)])
    else:
        polygon(ctx, [(0, -s * 1.15), (s * 0.9, 0), (0, s * 1.15), (-s * 0.9, 0)])
    ctx.restore()


def draw_ball(ctx, i, x, y, r, rot=0.0, glow_a=0.42, shadow=True):
    col = COLS[i]
    if shadow:
        g = cairo.RadialGradient(x + r * 0.2, y + r * 0.35, r * 0.5, x + r * 0.2, y + r * 0.35, r * 1.35)
        g.add_color_stop_rgba(0, 0, 0, 0.05, 0.55)
        g.add_color_stop_rgba(1, 0, 0, 0.05, 0)
        ctx.set_source(g)
        ctx.arc(x + r * 0.2, y + r * 0.35, r * 1.35, 0, TAU)
        ctx.fill()
    if glow_a > 0:
        ctx.set_operator(cairo.OPERATOR_ADD)
        glow(ctx, x, y, r * 2.5, col, glow_a)
        ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.9)
    ctx.arc(x, y, r + 2.5, 0, TAU)
    ctx.fill()
    g = cairo.RadialGradient(x - r * 0.35, y - r * 0.42, r * 0.05, x, y, r)
    g.add_color_stop_rgb(0, *lighten(col, 0.72))
    g.add_color_stop_rgb(0.45, *col)
    g.add_color_stop_rgb(1, *darken(col, 0.42))
    ctx.set_source(g)
    ctx.arc(x, y, r, 0, TAU)
    ctx.fill()
    lw = max(2.0, r * 0.085)
    ctx.set_line_width(lw)
    ctx.set_source_rgba(1, 1, 1, 0.85)
    ctx.arc(x, y, r - lw / 2, 0, TAU)
    ctx.stroke()
    symbol(ctx, i, x, y, r * 0.44, rot)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(r * 0.2)
    ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.95)
    ctx.stroke_preserve()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill()
    ctx.save()
    ctx.translate(x - r * 0.40, y - r * 0.50)
    ctx.rotate(-0.7)
    ctx.scale(r * 0.30, r * 0.15)
    ctx.arc(0, 0, 1, 0, TAU)
    ctx.restore()
    ctx.set_source_rgba(1, 1, 1, 0.7)
    ctx.fill()


def pickup_icon(ctx, kind, x, y, s):
    """White icon path: grow = two-headed arrow, fast = bolt, bomb = starburst."""
    if kind == "grow":
        pts = [(0.92, -0.92), (0.2, -0.82), (0.44, -0.58), (-0.58, 0.44), (-0.82, 0.2), (-0.92, 0.92),
               (-0.2, 0.82), (-0.44, 0.58), (0.58, -0.44), (0.82, -0.2)]
    elif kind == "fast":
        pts = [(0.25, -1.0), (-0.62, 0.16), (-0.08, 0.16), (-0.28, 1.0), (0.62, -0.26), (0.08, -0.26)]
    else:
        pts = [((1.0 if j % 2 == 0 else 0.45) * math.cos(j * TAU / 16 - TAU / 4),
                (1.0 if j % 2 == 0 else 0.45) * math.sin(j * TAU / 16 - TAU / 4)) for j in range(16)]
    polygon(ctx, [(x + px * s, y + py * s) for px, py in pts])


def tile_sprite(col, k, dim=False, bright=1.0):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(TILE), int(TILE))
    c = cairo.Context(s)
    m, w = 1.5, TILE - 3.0
    if col is None:
        top, bot = (0.115 * bright, 0.135 * bright, 0.255 * bright), (0.08 * bright, 0.095 * bright, 0.19 * bright)
    else:
        base = tuple(min(1.0, v * bright) for v in col)
        if dim:
            lum = 0.3 * base[0] + 0.55 * base[1] + 0.15 * base[2]
            base = mix(mix(base, (lum, lum, lum), 0.45), (0.05, 0.06, 0.13), 0.62)
        top, bot = lighten(base, 0.14), darken(base, 0.16)
    g = cairo.LinearGradient(0, m, 0, m + w)
    g.add_color_stop_rgb(0, *top)
    g.add_color_stop_rgb(1, *bot)
    rounded_rect(c, m, m, w, w, 7)
    c.set_source(g)
    c.fill()
    g = cairo.LinearGradient(0, m, 0, m + w * 0.5)                     # gloss on the top half
    a = 0.05 if col is None else (0.10 if dim else 0.24)
    g.add_color_stop_rgba(0, 1, 1, 1, a)
    g.add_color_stop_rgba(1, 1, 1, 1, 0)
    rounded_rect(c, m + 2, m + 2, w - 4, w * 0.5, 5)
    c.set_source(g)
    c.fill()
    if col is not None:                                                # each colour also carries its symbol
        symbol(c, k, TILE / 2, TILE / 2, 7.5)
        c.set_source_rgba(*darken(base, 0.45), 0.55)
        c.fill()
    return s


def make_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (yy / H)[..., None]
    img = np.array([0.050, 0.065, 0.160], np.float32) * (1 - v) + np.array([0.016, 0.020, 0.055], np.float32) * v
    for cx, cy, rad, col in ((540, 720, 900, (0.050, 0.070, 0.190)), (80, 250, 620, (0.070, 0.020, 0.140)),
                             (1010, 1500, 700, (0.000, 0.060, 0.110))):      # soft light pools for depth
        d = np.hypot(xx - cx, yy - cy) / rad
        img += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array(col, np.float32)
    vig = np.clip(1.12 - 0.5 * (((xx - 540) / 720) ** 2 + ((yy - 960) / 1180) ** 2), 0.3, 1.0)
    img *= vig[..., None]
    img += np.random.default_rng(5).normal(0, 0.006, (H, W, 1)).astype(np.float32)   # grain hides banding
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    return cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4), buf


# ====================================================================== the scene
class Scene:
    W, H, FPS, nframes = W, H, FPS, NF

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed, record=True)
        self.stats = drama(self.sim)
        self.winner = self.stats["winner_index"]
        self._res = None
        self._prep()

    # ------------------------------------------------------------ precomputed animation state
    def _prep(self):
        sim = self.sim
        self.pos = np.array(sim["pos"]).reshape(-1, 4, 3)
        counts, final = sim["counts"], sim["counts"][-1]
        rate = max(final) / COUNT_LEN
        disp, leader, hot, amb, scale = [0.0] * 4, -1, [0.0] * 4, NEUTRAL, 0.4
        self.board = []                      # per frame: (shown tiles, leader glow, its rise, bar scale, ambient)
        for f in range(NF):
            t = f / FPS
            if t < PLAY:
                c = counts[f]
                disp = [d + (c[i] - d) * 0.4 for i, d in enumerate(disp)]
                shown = list(disp)
                if t >= 1.5 and (leader < 0 or c[leader] < max(c)):     # a tie never takes the lead
                    leader = c.index(max(c))
            elif t < T_COUNT:
                k = 1 - ease_in_out(ramp(t, PLAY, PLAY + 0.12))
                shown = [c * k for c in final]
                leader = -1                                             # no crown while the tiles are counted
            elif t < T_RESET:
                shown = [min(c, rate * (t - T_COUNT)) for c in final]
                leader = self.winner if t >= T_REVEAL else -1
            else:
                k = 1 - ease_in_out(ramp(t, T_RESET, T_RESET + 0.4))
                shown = [c * k for c in final]
                leader = -1
            was = hot
            hot = [h + ((1.0 if i == leader else 0.0) - h) * 0.25 for i, h in enumerate(hot)]
            amb = mix(amb, COLS[leader] if leader >= 0 else NEUTRAL, 0.25 if t >= T_RESET else 0.06)
            scale += (max(0.40, 1.12 * max(shown) / (N * N)) - scale) * 0.12
            self.board.append((shown, hot, [max(0.0, h - w) for h, w in zip(hot, was)], scale, amb))

        self.shakes = sorted([(h[0], min(7.0, 1.5 + h[5] / 260)) for h in sim["hits"]]
                             + [(p["t1"], 14.0 if p["kind"] == "bomb" else 5.0) for p in sim["picks"] if p["by"] >= 0]
                             + [(PLAY, 9.0), (T_REVEAL, 18.0)])
        rnd = random.Random(self.seed * 7919 + 1)

        def sparks(n, a0, spread, v0, v1, ncol):
            out = []
            for _ in range(n):
                a = a0 + rnd.uniform(-spread, spread)
                out.append((math.cos(a), math.sin(a), rnd.uniform(v0, v1), rnd.uniform(2.5, 5.5),
                            rnd.uniform(0.28, 0.5), rnd.randrange(ncol)))
            return out

        fx = [(0.0, 0.6, "launch", (sx, sy, i)) for i, (sx, sy) in enumerate(sim["start"])]
        for t, i, x, y_, nx, ny in sim["walls"]:
            fx.append((t, 0.5, "wall", (x, y_, i, sparks(8, math.atan2(ny, nx), 1.25, 140, 520, 2))))
        for t, i, j, x, y_, rel in sim["hits"]:
            fx.append((t, 0.55, "hit", (x, y_, i, j, sparks(16, 0, math.pi, 180, 680, 3))))
        for p in sim["picks"]:
            fx.append((p["t0"], 0.45, "spawn", (p["x"], p["y"])))
            if p["by"] >= 0:
                n, v = (34, 950) if p["kind"] == "bomb" else (18, 600)
                fx.append((p["t1"], 1.1, "collect", (p["x"], p["y"], p["by"], p["kind"], sparks(n, 0, math.pi, 220, v, 2))))
        fx.sort(key=lambda e: e[0])
        self.fx, self.fx_t = fx, [e[0] for e in fx]

        self.confetti = [(rnd.randrange(2), rnd.uniform(0.10, 0.75), rnd.uniform(900, 2100), rnd.uniform(-9, 9),
                          rnd.uniform(0, TAU), rnd.uniform(9, 17), rnd.randrange(3), rnd.uniform(0.0, 0.25))
                         for _ in range(170)]
        self.dust = [(rnd.uniform(0, W), rnd.uniform(0, H), rnd.uniform(2.0, 9.0), rnd.uniform(0.05, 0.20),
                      rnd.choice((1, 1, 2)), rnd.uniform(8, 40), rnd.choice((1, 2, 3)), rnd.uniform(0, TAU))
                     for _ in range(46)]

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            bg, keep = make_background()
            tiles = {}
            for var, br in enumerate((0.94, 1.0, 1.06)):
                tiles[(-1, var)] = tile_sprite(None, 0, bright=br)
                for k, col in enumerate(COLS):
                    tiles[(k, var)] = tile_sprite(col, k, bright=br)
                    tiles[(k, var, "dim")] = tile_sprite(col, k, dim=True, bright=br)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep,
                             tiles=tiles, font=Font(FONT_PATH))
        return self._res

    def frame(self, f):
        R = self._resources()
        self.draw(R["ctx"], f, R)
        R["surf"].flush()
        return bytes(R["surf"].get_data())

    # ------------------------------------------------------------ state helpers
    def ball(self, i, t):
        """(x, y, r) of ball i at time t: simulated, frozen after the buzzer, eased home during the rewind."""
        if t < PLAY:
            s = max(0.0, t) * FPS * SUB
            k = int(s)
            p = self.pos[k, i] + (self.pos[min(k + 1, len(self.pos) - 1), i] - self.pos[k, i]) * (s - k)
            return p[0], p[1], p[2]
        p = self.pos[-1, i]
        if t <= T_RESET:
            return p[0], p[1], p[2]
        k = ease_in_out(ramp(t, T_RESET, T_RESET + 0.5))
        sx, sy = self.sim["start"][i]
        return lerp(p[0], sx, k), lerp(p[1], sy, k), lerp(p[2], R0, k)

    def is_fast(self, i, t):
        return any(b == i and tc <= t < tc + FAST_T for tc, b in self.sim["fast"])

    def shake(self, t):
        sx = sy = 0.0
        for te, amp in self.shakes:
            if te > t:
                break
            d = t - te
            if d < 0.6:
                a = amp * math.exp(-d / 0.11)
                sx += a * math.sin(d * 71 + te * 13)
                sy += a * math.cos(d * 59 + te * 7)
        return sx, sy

    @staticmethod
    def hook(t):
        """1 while the opening layout is up (big question, smaller arena), 0 in the main layout."""
        if t < 14:
            return 1 - ease_in_out(ramp(t, T_HOOK + 0.15, T_HOOK + 0.8))
        return ease_in_out(ramp(t, T_RESET, T_RESET + 0.3))

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, R):
        t = f / FPS
        font = R["font"]
        shown, hot, rise, bscale, amb = self.board[f]
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(R["bg"], 0, 0)
        ctx.paint()
        self._ambient(ctx, t, amb)
        z = lerp(1.0, HOOK_ZOOM, self.hook(t))
        sx, sy = self.shake(t)
        ctx.save()
        ctx.translate(540 - A / 2 * z + sx, AY + A - A * z + sy)
        ctx.scale(z, z)
        self._panel(ctx, t, font)
        self._tiles(ctx, R["tiles"], f, t)
        self._edges(ctx)
        self._fx_under(ctx, t)
        self._pickups(ctx, t)
        self._balls(ctx, t)
        self._fx_over(ctx, t, font)
        m = ctx.get_matrix()
        ctx.restore()
        self._confetti(ctx, t)
        ctx.save()
        ctx.set_matrix(m)
        self._finale(ctx, t, font)
        ctx.restore()
        self._titles(ctx, t, font)
        self._scoreboard(ctx, f, t, font, shown, hot, rise, bscale)
        fl = 0.20 * (1 - ramp(t, T_REVEAL, T_REVEAL + 0.3)) ** 2 if t >= T_REVEAL else \
            0.10 * (1 - ramp(t, PLAY, PLAY + 0.2)) ** 2 if t >= PLAY else 0.0
        if fl > 0.004:
            ctx.set_operator(cairo.OPERATOR_ADD)
            ctx.set_source_rgba(1, 1, 1, fl)
            ctx.paint()
            ctx.set_operator(cairo.OPERATOR_OVER)

    def _ambient(self, ctx, t, amb):
        ctx.set_operator(cairo.OPERATOR_ADD)
        ph = TAU * t / DUR
        glow(ctx, 540 + 330 * math.cos(ph), 720 + 260 * math.sin(ph), 820, amb, 0.17)
        glow(ctx, 540 - 380 * math.cos(ph), 1250 - 300 * math.sin(ph), 640, mix(amb, (0.45, 0.25, 1.0), 0.5), 0.12)
        for x0, y0, r, a, loops, sway, cyc, p0 in self.dust:          # drifting motes, periodic over the loop
            y = (y0 - loops * H * t / DUR) % H
            x = x0 + sway * math.sin(cyc * ph + p0)
            g = cairo.RadialGradient(x, y, 0, x, y, r)
            g.add_color_stop_rgba(0, 0.75, 0.85, 1.0, a)
            g.add_color_stop_rgba(1, 0.75, 0.85, 1.0, 0)
            ctx.set_source(g)
            ctx.arc(x, y, r, 0, TAU)
            ctx.fill()
        ctx.set_operator(cairo.OPERATOR_OVER)

    def _panel(self, ctx, t, font):
        for k in range(6):                                            # soft drop shadow under the board
            e = 6 + k * 7
            rounded_rect(ctx, -18 - e, -18 - e + 22, A + 36 + 2 * e, A + 36 + 2 * e, 30 + e)
            ctx.set_source_rgba(0, 0, 0.02, 0.085)
            ctx.fill()
        g = cairo.LinearGradient(0, -18, 0, A + 18)
        g.add_color_stop_rgb(0, 0.20, 0.24, 0.44)
        g.add_color_stop_rgb(1, 0.09, 0.11, 0.24)
        rounded_rect(ctx, -18, -18, A + 36, A + 36, 30)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_line_width(2.5)
        wk = ease_out(ramp(t, T_REVEAL, T_REVEAL + 0.3)) * (1 - ramp(t, T_RESET, T_RESET + 0.3))
        ctx.set_source_rgba(*mix((0.55, 0.62, 0.95), COLS[self.winner], wk), 0.35 + 0.65 * wk)
        ctx.stroke()
        rounded_rect(ctx, -5, -5, A + 10, A + 10, 12)
        ctx.set_source_rgb(0.028, 0.035, 0.085)
        ctx.fill()
        # timer bar: drains during the battle, refills during the rewind
        rem = 1 - t / PLAY if t < PLAY else ease_in_out(ramp(t, T_RESET + 0.05, T_RESET + 0.5))
        left = PLAY - t
        beat = 0.0
        if 0 < left <= 5.0:
            step = 0.25 if left <= 2.0 else 0.5
            beat = math.exp(-((t / step) % 1.0) * step * 9)
        x1, y0, hb = A - 128, -64.0, 24.0
        rounded_rect(ctx, 0, y0, x1, hb, hb / 2)
        ctx.set_source_rgba(1, 1, 1, 0.13)
        ctx.fill()
        if rem > 0.004:
            wbar = max(hb, x1 * rem)
            if beat > 0.02:
                ctx.set_operator(cairo.OPERATOR_ADD)
                rounded_rect(ctx, -6, y0 - 6, wbar + 12, hb + 12, hb / 2 + 6)
                ctx.set_source_rgba(1, 1, 1, 0.22 * beat)
                ctx.fill()
                ctx.set_operator(cairo.OPERATOR_OVER)
            g = cairo.LinearGradient(0, y0, 0, y0 + hb)
            g.add_color_stop_rgb(0, 1, 1, 1)
            g.add_color_stop_rgb(1, 0.72, 0.78, 0.95)
            rounded_rect(ctx, 0, y0, wbar, hb, hb / 2)
            ctx.set_source(g)
            ctx.fill()
        secs = str(max(0, math.ceil(left - 1e-6))) if t < PLAY else ("0" if t < T_RESET + 0.35 else str(int(PLAY)))
        size = 68 * (1 + 0.26 * beat)                                 # seconds left: large, beating in the last 5 s
        font.draw(ctx, secs, A - 23, y0 + hb / 2 + size * font.cap_height / 2, size,
                  mix(WHITE, (1.0, 0.82, 0.45), beat), anchor="r", stroke=INK, stroke_w=7)

    def _tiles(self, ctx, spr, f, t):
        fi = min(f, NPLAY)
        own, prv, pt = self.sim["owner"][fi], self.sim["prev"][fi], self.sim["ptime"][fi]
        win, half = self.winner, TILE / 2
        after = t >= T_REVEAL
        kdim = ease_out(ramp(t, T_REVEAL, T_REVEAL + 0.35))
        wx, wy, _ = self.ball(win, t)
        flashes = []
        tau = t if t < 14 else t - DUR                                # a sheen crosses the empty board over the loop cut
        sheen = (tau + 0.3) / 1.0 * (2 * N + 6) - 4 if -0.3 < tau < 0.7 else None
        for idx in range(N * N):
            tx, ty = idx % N, idx // N
            x, y = tx * TILE, ty * TILE
            var = (tx * 7 + ty * 13 + tx * ty) % 3
            o = own[idx] - 1
            sc = 1.0
            under = None
            if o < 0:
                under = (-1, var)
            else:
                age = t - pt[idx]
                if age < POP:
                    u = prv[idx] - 1
                    under = (u, var) if u >= 0 else (-1, var)
                    sc = 0.3 + 0.7 * ease_out_back(age / POP)
                    flashes.append((x, y, sc, 0.7 * (1 - age / POP) ** 2))
                if t >= T_RESET:
                    under = (-1, var)
                    sc = 1 - ease_in(ramp(t, T_RESET + (tx + ty) / (2 * N - 2) * 0.3,
                                          T_RESET + (tx + ty) / (2 * N - 2) * 0.3 + 0.22))
            if under is not None:
                ctx.set_source_surface(spr[under], x, y)
                ctx.rectangle(x, y, TILE, TILE)
                ctx.fill()
                if sheen is not None:
                    a = 0.20 * math.exp(-((tx + ty - sheen) / 2.4) ** 2)
                    if a > 0.01:
                        flashes.append((x, y, 1.0, a))
            if o < 0 or sc <= 0.02:
                continue
            loser = after and o != win
            key = (o, var, "dim") if loser and kdim >= 1 else (o, var)
            if sc == 1.0:
                ctx.set_source_surface(spr[key], x, y)
                ctx.rectangle(x, y, TILE, TILE)
                ctx.fill()
                if loser and kdim < 1:
                    ctx.save()
                    ctx.rectangle(x, y, TILE, TILE)
                    ctx.clip()
                    ctx.set_source_surface(spr[(o, var, "dim")], x, y)
                    ctx.paint_with_alpha(kdim)
                    ctx.restore()
            else:
                ctx.save()
                ctx.translate(x + half, y + half)
                ctx.scale(sc, sc)
                ctx.set_source_surface(spr[key], -half, -half)
                ctx.rectangle(-half, -half, TILE, TILE)
                ctx.fill()
                ctx.restore()
            if PLAY <= t < T_REVEAL:                                  # counting sweep, top to bottom
                d = t - (T_COUNT + (ty + 0.5) / N * COUNT_LEN)
                if 0 <= d < 0.16:
                    flashes.append((x, y, 1.0, 0.5 * (1 - d / 0.16)))
            elif after and o == win and t < T_RESET:                  # winner's tiles ripple and breathe
                d = math.hypot(x + half - wx, y + half - wy)
                ph = (t - T_REVEAL) * 1100 - d
                a = 0.6 * math.exp(-(ph / 150) ** 2) + 0.09 * (0.5 + 0.5 * math.sin(t * 7.5 - d / 70))
                flashes.append((x, y, 1.0, a))
        ctx.set_operator(cairo.OPERATOR_ADD)
        for x, y, sc, a in flashes:
            h = (half - 1.5) * sc
            rounded_rect(ctx, x + half - h, y + half - h, 2 * h, 2 * h, 7 * sc)
            ctx.set_source_rgba(1, 1, 1, a)
            ctx.fill()
        ctx.set_operator(cairo.OPERATOR_OVER)

    def _edges(self, ctx):
        """Inner shadow so the tile floor reads as recessed."""
        for x0, y0, x1, y1, rect in ((0, 0, 0, 26, (0, 0, A, 26)), (0, A, 0, A - 16, (0, A - 16, A, 16)),
                                     (0, 0, 20, 0, (0, 0, 20, A)), (A, 0, A - 20, 0, (A - 20, 0, 20, A))):
            g = cairo.LinearGradient(x0, y0, x1, y1)
            g.add_color_stop_rgba(0, 0, 0, 0.04, 0.42)
            g.add_color_stop_rgba(1, 0, 0, 0.04, 0)
            ctx.set_source(g)
            ctx.rectangle(*rect)
            ctx.fill()

    def _active_fx(self, t):
        lo = bisect.bisect_left(self.fx_t, t - 1.2)
        hi = bisect.bisect_right(self.fx_t, t)
        return [(t - e[0], e) for e in self.fx[lo:hi] if t - e[0] < e[1]]

    def _fx_under(self, ctx, t):
        """Wall flashes and motion trails (additive, under the balls)."""
        ctx.set_operator(cairo.OPERATOR_ADD)
        for age, (_, life, kind, d) in self._active_fx(t):
            if kind == "wall":
                glow(ctx, d[0], d[1], 95, lighten(COLS[d[2]], 0.25), 0.75 * (1 - age / life) ** 2)
        fade = 1 - ramp(t, PLAY, PLAY + 0.25)
        if t > T_RESET:
            fade = ramp(t, T_RESET, T_RESET + 0.1) * (1 - ramp(t, T_RESET + 0.35, T_RESET + 0.5))
        if fade > 0.01:
            for i in range(4):
                fast = self.is_fast(i, t)
                n, dt, a0 = (34, 1 / 60, 0.30) if fast else (20, 1 / 60, 0.24)
                if t > T_RESET:                                       # balls fly home fast: sample finer
                    n, dt, a0 = 60, 1 / 240, 0.10
                col = lighten(COLS[i], 0.5 if fast else 0.3)
                for k in range(n, 0, -1):
                    tt = t - k * dt
                    if tt < 0:
                        continue
                    x, y, r = self.ball(i, tt)
                    q = 1 - k / (n + 1)
                    ctx.set_source_rgba(col[0], col[1], col[2], a0 * q ** 1.5 * fade)
                    ctx.arc(x, y, r * (0.25 + 0.72 * q), 0, TAU)
                    ctx.fill()
        ctx.set_operator(cairo.OPERATOR_OVER)

    def _pickups(self, ctx, t):
        for p in self.sim["picks"]:
            if not p["t0"] <= t < p["t1"] + 0.001 or t >= PLAY + 0.3:
                continue
            age = t - p["t0"]
            sc = ease_out_back(ramp(age, 0, 0.3)) * (1 + 0.07 * math.sin(age * 7))
            a = 1.0
            if p["by"] < 0:
                a = 1 - ramp(t, p["t1"] - 0.5, p["t1"])
            a *= 1 - ramp(t, PLAY, PLAY + 0.3)
            if sc <= 0.02 or a <= 0.02 or t >= p["t1"]:
                continue
            x, y, r = p["x"], p["y"], PICK_R * sc
            ctx.set_operator(cairo.OPERATOR_ADD)
            glow(ctx, x, y, r * 3.0, (1, 1, 1), 0.38 * a)
            ctx.set_operator(cairo.OPERATOR_OVER)
            ctx.arc(x, y, r, 0, TAU)
            ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.92 * a)
            ctx.fill_preserve()
            ctx.set_line_width(4.5)
            ctx.set_source_rgba(1, 1, 1, a)
            ctx.stroke()
            ctx.set_dash([11, 9], -age * 46)
            ctx.set_line_width(3.5)
            ctx.arc(x, y, r + 10, 0, TAU)
            ctx.set_source_rgba(1, 1, 1, 0.75 * a)
            ctx.stroke()
            ctx.set_dash([])
            pickup_icon(ctx, p["kind"], x, y, r * 0.62)
            ctx.set_source_rgba(1, 1, 1, a)
            ctx.fill()

    def _balls(self, ctx, t):
        win = self.winner
        kwin = ease_out_back(ramp(t, T_REVEAL, T_REVEAL + 0.35)) * (1 - ramp(t, T_RESET, T_RESET + 0.3))
        charge = ease_in(ramp(t, DUR - 0.3, DUR), 2)                  # glow builds into the loop point
        spin = min(t, PLAY) / 3.0 * TAU                               # whole turns over the battle: loops clean
        for i in sorted(range(4), key=lambda i: i == win and t >= T_REVEAL):
            x, y, r = self.ball(i, t)
            ga = 0.42 + (0.25 if self.is_fast(i, t) else 0.0) + 0.5 * charge
            if i == win:
                r *= 1 + 0.25 * kwin
                ga += 0.4 * kwin
            draw_ball(ctx, i, x, y, r, rot=spin * (1 if i % 2 else -1) + i, glow_a=ga)
            if i != win and kwin > 0:
                ctx.arc(x, y, r + 3, 0, TAU)
                ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.55 * kwin)
                ctx.fill()
            if charge > 0:
                ctx.set_operator(cairo.OPERATOR_ADD)
                glow(ctx, x, y, 200, lighten(COLS[i], 0.2), 0.9 * charge ** 2)
                ctx.set_line_width(5)
                ctx.set_source_rgba(1, 1, 1, 0.9 * charge)
                ctx.arc(x, y, r + 8 + 90 * (1 - charge), 0, TAU)
                ctx.stroke()
                ctx.set_operator(cairo.OPERATOR_OVER)

    @staticmethod
    def _sparks(ctx, x, y, sparks, age, cols):
        for ca, sa, v, size, life, ci in sparks:
            if age >= life:
                continue
            k = age / life
            d1 = v * (1 - math.exp(-age * 5)) / 5
            d0 = v * (1 - math.exp(-max(0.0, age - 0.04) * 5)) / 5
            c = cols[ci]
            ctx.set_source_rgba(c[0], c[1], c[2], 1 - k)
            ctx.set_line_width(size * (1 - k) + 0.6)
            ctx.move_to(x + ca * d0, y + sa * d0)
            ctx.line_to(x + ca * d1, y + sa * d1)
            ctx.stroke()

    @staticmethod
    def _ring(ctx, x, y, r, width, col, a):
        if a > 0.01:
            ctx.set_line_width(width)
            ctx.set_source_rgba(col[0], col[1], col[2], a)
            ctx.arc(x, y, r, 0, TAU)
            ctx.stroke()

    def _fx_over(self, ctx, t, font):
        """Sparks, shock rings and pickup call-outs (above the balls)."""
        ctx.set_operator(cairo.OPERATOR_ADD)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        labels = []
        for age, (_, life, kind, d) in self._active_fx(t):
            k = age / life
            if kind == "wall":
                c = COLS[d[2]]
                self._sparks(ctx, d[0], d[1], d[3], age, (lighten(c, 0.3), (1, 1, 1)))
            elif kind == "hit":
                self._sparks(ctx, d[0], d[1], d[4], age, (lighten(COLS[d[2]], 0.2), lighten(COLS[d[3]], 0.2), (1, 1, 1)))
                self._ring(ctx, d[0], d[1], 14 + 85 * ease_out(k), 7 * (1 - k) + 1, (1, 1, 1), 0.85 * (1 - k) ** 1.5)
                glow(ctx, d[0], d[1], 70, (1, 1, 1), 0.7 * (1 - ramp(age, 0, 0.14)))
            elif kind == "launch":
                glow(ctx, d[0], d[1], 200, lighten(COLS[d[2]], 0.2), 0.9 * (1 - k) ** 2)
                self._ring(ctx, d[0], d[1], R0 + 8 + 110 * ease_out(k), 5 * (1 - k) + 1, (1, 1, 1), 0.9 * (1 - k) ** 1.5)
            elif kind == "spawn":
                self._ring(ctx, d[0], d[1], 20 + 70 * ease_out(k), 5 * (1 - k) + 1, (1, 1, 1), 0.8 * (1 - k))
            elif kind == "collect":
                c = COLS[d[2]]
                self._sparks(ctx, d[0], d[1], d[4], age, (lighten(c, 0.25), (1, 1, 1)))
                big = BOMB_R if d[3] == "bomb" else 120
                k2 = ramp(age, 0, 0.42)
                self._ring(ctx, d[0], d[1], 24 + big * ease_out(k2), 12 * (1 - k2) + 1, lighten(c, 0.3), (1 - k2) ** 1.3)
                glow(ctx, d[0], d[1], big * 0.9, lighten(c, 0.3), 0.8 * (1 - ramp(age, 0, 0.22)))
                labels.append((age, d))
        ctx.set_operator(cairo.OPERATOR_OVER)
        for age, d in labels:
            sc = ease_out_back(ramp(age, 0, 0.22), 2.2)
            a = 1 - ramp(age, 0.8, 1.1)
            text = LABEL[d[3]]
            size = 66 * sc
            w = font.width(text, size)
            x = clamp(d[0], w / 2 + 14, A - w / 2 - 14)
            y = clamp(d[1] - 46 - 40 * ease_out(ramp(age, 0, 1.1)), 70, A - 20)
            font.draw(ctx, text, x, y, size, lighten(COLS[d[2]], 0.55), anchor="c", alpha=a,
                      stroke=INK, stroke_w=12)

    def _finale(self, ctx, t, font):
        """TIME! stamp at the buzzer, then the winner banner and the closing line."""
        if PLAY <= t < PLAY + 0.8:
            sc = lerp(2.3, 1.0, ease_out(ramp(t, PLAY, PLAY + 0.16), 4))
            a = ramp(t, PLAY, PLAY + 0.06) * (1 - ramp(t, PLAY + 0.55, PLAY + 0.8))
            size = 250 * sc
            font.draw(ctx, "TIME!", A / 2, A / 2 + size * font.cap_height / 2, size, WHITE, anchor="c",
                      alpha=a, stroke=INK, stroke_w=22, shadow=10)
        out = 1 - ease_in(ramp(t, T_RESET, T_RESET + 0.22))
        k = ease_out_back(ramp(t, T_REVEAL, T_REVEAL + 0.32), 1.3) * out
        if k <= 0.01:
            return
        win = self.winner
        col = COLS[win]
        cy = A / 2
        grow = ease_out(ramp(t, T_LINE - 0.06, T_LINE + 0.2))
        hh = (210 + 118 * grow) * k
        ctx.save()                                                    # rotating light rays behind the banner
        ctx.rectangle(-5, -5, A + 10, A + 10)
        ctx.clip()
        ctx.set_operator(cairo.OPERATOR_ADD)
        ra = 0.10 * out * ease_out(ramp(t, T_REVEAL, T_REVEAL + 0.4))
        for j in range(12):
            a0 = j * TAU / 12 + (t - T_REVEAL) * 0.35
            ctx.move_to(A / 2, cy)
            ctx.arc(A / 2, cy, 900, a0, a0 + TAU / 24)
            ctx.close_path()
            ctx.set_source_rgba(*lighten(col, 0.3), ra)
            ctx.fill()
        k3 = ramp(t, T_REVEAL, T_REVEAL + 0.6)
        self._ring(ctx, A / 2, cy, 60 + 620 * ease_out(k3), 26 * (1 - k3) + 1, lighten(col, 0.4), (1 - k3) ** 1.2)
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.restore()
        ctx.rectangle(-18, cy - hh / 2, A + 36, hh)
        ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.94)
        ctx.fill()
        for yy in (cy - hh / 2, cy + hh / 2 - 7):
            ctx.rectangle(-18, yy, A + 36, 7)
            ctx.set_source_rgb(*col)
            ctx.fill()
        # row 1: winner ball + "<COLOUR> WINS!"
        name = NAMES[win] + " "
        size = 158
        while font.width(name + "WINS!", size) + 170 > A - 80:
            size -= 4
        cap = size * font.cap_height
        tw = font.width(name + "WINS!", size)
        y1 = cy - 60 * grow
        x0 = A / 2 - (tw + 170) / 2
        pop = ease_out_back(ramp(t, T_REVEAL + 0.05, T_REVEAL + 0.4), 1.2) * out
        if pop > 0.01:
            ctx.save()
            ctx.translate(A / 2, y1)
            ctx.scale(pop, pop)
            ctx.translate(-A / 2, -y1)
            draw_ball(ctx, win, x0 + 66, y1, 62, rot=(t - T_REVEAL) * 0.8, glow_a=0.5, shadow=False)
            wn = font.draw(ctx, name, x0 + 170, y1 + cap / 2, size, lighten(col, 0.18), stroke=INK, stroke_w=14)
            font.draw(ctx, "WINS!", x0 + 170 + wn, y1 + cap / 2, size, WHITE, stroke=INK, stroke_w=14)
            ctx.restore()
        # row 2: closing line, word by word
        if grow > 0:
            size2 = 88
            words = ("DID", "YOU", "CALL", "IT?")
            gap = font.width(" ", size2)
            total = sum(font.width(w_, size2) for w_ in words) + gap * 3
            x = A / 2 - total / 2
            yb = cy + 62 + size2 * font.cap_height
            for j, w_ in enumerate(words):
                ww = font.width(w_, size2)
                p = ease_out_back(ramp(t, T_LINE + j * 0.07, T_LINE + j * 0.07 + 0.22), 2.4) * out
                if p > 0.01:
                    ctx.save()
                    ctx.translate(x + ww / 2, yb - size2 * font.cap_height / 2)
                    ctx.scale(p, p)
                    font.draw(ctx, w_, 0, size2 * font.cap_height / 2, size2, WHITE, anchor="c",
                              stroke=INK, stroke_w=10)
                    ctx.restore()
                x += ww + gap

    def _titles(self, ctx, t, font):
        tau = t if t < 14 else t - DUR            # time relative to the loop point: the question pops in over the cut
        # big opening question (present and readable on frame 0)
        a_big = ramp(tau, -0.26, -0.16) * (1 - ramp(tau, T_HOOK, T_HOOK + 0.3)) if tau < 5 else 0.0
        if a_big > 0.004:
            shrink = lerp(1.0, 0.78, ease_in(ramp(tau, T_HOOK, T_HOOK + 0.3)))
            size = 97
            cap = size * font.cap_height
            for j, (text, base) in enumerate((("WHICH COLOUR WINS", 262), ("THE MOST TERRITORY?", 368))):
                sc = (0.6 + 0.4 * ease_out_back(ramp(tau, -0.26 + j * 0.05, 0.12 + j * 0.05), 1.2)) * shrink
                ctx.save()
                ctx.translate(540, base - cap / 2 - 30 * (1 - shrink))
                ctx.scale(sc, sc)
                if j == 0:
                    fill = WHITE
                else:
                    w = font.width(text, size)
                    fill = cairo.LinearGradient(-w / 2, 0, w / 2, 0)
                    for s_, c in zip((0.0, 0.34, 0.67, 1.0), (COLS[0], COLS[3], COLS[2], COLS[1])):
                        c = lighten(c, 0.22)
                        fill.add_color_stop_rgba(s_, c[0], c[1], c[2], a_big)
                font.draw(ctx, text, 0, cap / 2, size, fill, anchor="c", alpha=a_big, stroke=INK, stroke_w=14,
                          shadow=6)
                ctx.restore()
        # compact status line above the arena
        size = 80
        cap = size * font.cap_height
        lines = (("WHICH COLOUR WINS?", T_HOOK + 0.12, PLAY), ("COUNTING TILES...", PLAY + 0.1, T_REVEAL),
                 ("FINAL TILE COUNT", T_REVEAL + 0.1, T_RESET))
        for text, t0, t1 in lines:                                    # old line leaves before the new one enters
            if not t0 <= t < t1 + 0.1:
                continue
            a = ramp(t, t0, t0 + 0.12) * (1 - ramp(t, t1, t1 + 0.1))
            dy = 26 * (1 - ease_out(ramp(t, t0, t0 + 0.2))) - 22 * ease_in(ramp(t, t1, t1 + 0.1))
            if text.endswith("..."):
                text = text[:-3] + "." * (1 + int((t - t0) / 0.25) % 3)
                x = 540 - font.width("COUNTING TILES...", size) / 2
                font.draw(ctx, text, x, 232 + dy, size, WHITE, alpha=a, stroke=INK, stroke_w=10)
            else:
                font.draw(ctx, text, 540, 232 + dy, size, WHITE, anchor="c", alpha=a, stroke=INK, stroke_w=10)

    def _scoreboard(self, ctx, f, t, font, shown, hot, rise, bscale):
        counts = self.sim["counts"][min(f, NPLAY)]
        NN = N * N
        won = ease_out(ramp(t, T_REVEAL, T_REVEAL + 0.3)) * (1 - ramp(t, T_RESET, T_RESET + 0.3))
        for i in range(4):
            y = BOARD_Y + i * (ROW_H + ROW_GAP)
            col = COLS[i]
            h = max(hot[i], won if i == self.winner else 0.0)
            pulse = 0.5 + 0.5 * math.sin(t * 9) if i == self.winner and won > 0 else 1.0
            rounded_rect(ctx, AX - 10, y, A + 20, ROW_H, 24)
            ctx.set_source_rgba(1, 1, 1, 0.055 + 0.075 * h)
            ctx.fill_preserve()
            if h > 0.02:
                ctx.set_line_width(3.5)
                ctx.set_source_rgba(*lighten(col, 0.2), h * (0.55 + 0.45 * pulse))
                ctx.stroke_preserve()
            if rise[i] > 0.004:                                       # flash when this colour takes the lead
                ctx.set_operator(cairo.OPERATOR_ADD)
                ctx.set_source_rgba(1, 1, 1, min(0.5, 2.4 * rise[i]))
                ctx.fill_preserve()
                ctx.set_operator(cairo.OPERATOR_OVER)
            ctx.new_path()
            draw_ball(ctx, i, AX + 36, y + ROW_H / 2, 28, glow_a=0.3 * h, shadow=False)
            if hot[i] > 0.5:                                          # crown on whoever leads
                ck = 22 * ease_out_back(hot[i] * 2 - 1, 2.5)
                ctx.save()
                ctx.translate(AX + 13, y + 14)
                ctx.rotate(-0.42)
                ctx.scale(ck, ck)
                polygon(ctx, [(-1, 0.55), (-1, -0.5), (-0.48, 0.05), (0, -0.75), (0.48, 0.05), (1, -0.5), (1, 0.55)])
                ctx.restore()
                ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                ctx.set_line_width(6)
                ctx.set_source_rgb(*INK)
                ctx.stroke_preserve()
                ctx.set_source_rgb(1, 1, 1)
                ctx.fill()
            tx0, tw, th = AX + 80, A - 80 - 152, 46.0
            ty = y + (ROW_H - th) / 2
            rounded_rect(ctx, tx0, ty, tw, th, 14)
            ctx.set_source_rgba(0.02, 0.03, 0.08, 0.85)
            ctx.fill()
            fw = clamp(shown[i] / NN / bscale) * tw
            if fw > 1.5:
                ctx.save()
                rounded_rect(ctx, tx0, ty, tw, th, 14)
                ctx.clip()
                g = cairo.LinearGradient(0, ty, 0, ty + th)
                g.add_color_stop_rgb(0, *lighten(col, 0.25))
                g.add_color_stop_rgb(0.5, *col)
                g.add_color_stop_rgb(1, *darken(col, 0.22))
                rounded_rect(ctx, tx0 - 30, ty, fw + 30, th, 14)
                ctx.set_source(g)
                ctx.fill()
                rounded_rect(ctx, tx0 - 30, ty + 4, fw + 26, th * 0.36, 9)
                ctx.set_source_rgba(1, 1, 1, 0.22)
                ctx.fill()
                ctx.restore()
            if t < PLAY:
                text, a = f"{round(100 * counts[i] / NN)}%", 1.0
            elif t < T_COUNT:
                text, a = f"{round(100 * counts[i] / NN)}%", 1 - ramp(t, PLAY, PLAY + 0.1)
            elif t < T_RESET + 0.2:
                text = str(int(shown[i] + 1e-6))
                a = ramp(t, T_COUNT, T_COUNT + 0.08) * (1 - ramp(t, T_RESET + 0.1, T_RESET + 0.2))
            else:
                text, a = "0%", ramp(t, T_RESET + 0.2, T_RESET + 0.35)
            size = 64 * (1 + 0.16 * won * pulse if i == self.winner else 1.0)
            font.draw(ctx, text, AX + A - 23, y + ROW_H / 2 + size * font.cap_height / 2, size, WHITE, anchor="r",
                      alpha=a, stroke=INK, stroke_w=8)

    def _confetti(self, ctx, t):
        age0 = t - T_REVEAL
        if not 0 <= age0 < 2.6:
            return
        col = COLS[self.winner]
        cols = (col, lighten(col, 0.45), WHITE)
        for side, ang, v, spin, ph, sz, ci, delay in self.confetti:
            age = age0 - delay
            if age <= 0:
                continue
            a = (1 - ramp(age, 1.5, 2.2)) * (1 - ramp(t, T_RESET - 0.1, T_RESET + 0.2))
            if a <= 0.01:
                continue
            d = v * (1 - math.exp(-age * 1.5)) / 1.5
            sgn = 1 if side == 0 else -1
            x = (AX - 20 if side == 0 else AX + A + 20) + sgn * math.sin(ang) * d
            y = AY + A + 10 - math.cos(ang) * d + 0.5 * 950 * age * age
            c = cols[ci]
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(ph + spin * age)
            ctx.scale(1.0, 0.35 + 0.65 * abs(math.cos(ph * 2 + spin * 1.3 * age)))
            ctx.rectangle(-sz / 2, -sz * 0.8, sz, sz * 1.6)
            ctx.set_source_rgba(c[0], c[1], c[2], a)
            ctx.fill()
            ctx.restore()

    # ------------------------------------------------------------ sound
    def audio(self):
        """Stereo float mix at au.SR: a note per bounce, a bed that builds, riser, tile count, win chime."""
        sim = self.sim
        rng = np.random.default_rng(self.seed + 4242)
        rnd = random.Random(self.seed + 99)
        sr = au.SR
        n, tail = int(DUR * sr), 3 * sr
        notes = np.zeros((n + tail, 2))          # bounces and effects (these get reverb)
        bed = np.zeros((n + tail, 2))            # rhythm and pads
        tt_ = lambda dur: np.arange(int(dur * sr)) / sr
        pan = lambda x: clamp((x / A - 0.5) * 1.3, -0.8, 0.8)

        # one note per bounce from A minor pentatonic; each ball has its own voice and register
        penta = (0, 3, 5, 7, 10)
        regs = [[m for m in range(lo, hi + 1) if (m - 57) % 12 in penta]
                for lo, hi in ((69, 86), (76, 93), (57, 74), (52, 69))]
        voices = (au.marimba, au.bell, au.pluck, au.chip)
        gains = (0.50, 0.26, 0.62, 0.50)
        ev = sorted([(w[0], w[1], w[2], 1.0) for w in sim["walls"]]
                    + [(h[0], h[1], h[3], 0.8) for h in sim["hits"]]
                    + [(h[0], h[2], h[3], 0.8) for h in sim["hits"]])
        deg = [len(r) // 2 for r in regs]
        last, last_any, cache = [-1.0] * 4, -1.0, {}
        for t, b, x, g in ev:
            if t - last[b] < 0.11 or t - last_any < 0.045:        # rate limit: never a wall of noise
                continue
            last[b] = last_any = t
            deg[b] = clamp(deg[b] + rnd.choice((-2, -1, -1, 1, 1, 2)), 0, len(regs[b]) - 1)
            m = regs[b][deg[b]]
            if (b, m) not in cache:
                cache[(b, m)] = voices[b](au.hz(m))
            au.place(notes, cache[(b, m)], t, gains[b] * g, pan(x))
        for h in sim["hits"]:
            au.place(notes, au.tick(520, 0.08), h[0], 0.22, pan(h[3]))
        for p in sim["picks"]:
            au.place(notes, au.glide(700, 1400, 0.10, 0.05), p["t0"], 0.16, pan(p["x"]))
            if p["by"] < 0:
                continue
            if p["kind"] == "grow":
                au.place(notes, au.glide(196, 784, 0.34, 0.25), p["t1"], 0.34, pan(p["x"]))
            elif p["kind"] == "fast":
                au.place(notes, au.glide(520, 2600, 0.20, 0.12), p["t1"], 0.22, pan(p["x"]))
                au.place(notes, au.swept_noise(rng, 0.3, 900, 6000) * np.exp(-tt_(0.3) * 7), p["t1"], 0.22, pan(p["x"]))
            else:
                au.place(notes, au.boom(rng, 0.9), p["t1"], 0.60, pan(p["x"]) * 0.5)
                au.place(notes, au.noise(rng, 0.4, 500, 7000) * np.exp(-tt_(0.4) * 11), p["t1"], 0.40, pan(p["x"]))
        au.place(notes, au.glide(300, 900, 0.14, 0.08), 0.0, 0.30)   # launch

        # bed: 120 BPM, two pickup beats then 12 bars ending on the buzzer; a new layer every two bars
        beat = 0.5
        kick = au.kick()
        bar0 = PLAY - 12 * 4 * beat
        for k in range(int(round(bar0 / beat))):
            au.place(bed, kick, k * beat, 0.30)
        roots = (45, 45, 41, 41, 48, 48, 43, 43, 45, 45, 41, 43)     # A A F F C C G G A A F G
        for bar, r in enumerate(roots):
            t0 = bar0 + bar * 4 * beat
            lvl = 0.45 + 0.85 * bar / 11                              # the bed builds bar by bar
            for m, g, pn in ((r + 12, 0.50, -0.3), (r + 19, 0.36, 0.3), (r + 24, 0.26, 0.0)):
                au.place(bed, au.pad(au.hz(m), 2.5, 0.5, 0.8), t0, 0.20 * lvl * g, pn)
            for b in ((0, 2) if bar < 2 else (0, 1, 2, 3)):
                au.place(bed, kick, t0 + b * beat, 0.48 * (0.6 + 0.4 * lvl))
            if bar >= 2:
                for k in range(4):
                    au.place(bed, au.hat(rng), t0 + (k + 0.5) * beat, 0.26 * lvl, 0.35)
            if bar >= 4:
                for k, on in enumerate((1, 0, 1, 1, 0, 1, 0, 1)):
                    if on:
                        au.place(bed, au.bass(au.hz(r + (12 if k == 7 else 0)), 0.2), t0 + k * beat / 2, 0.30 * lvl)
            if bar >= 6:
                for k in range(16):
                    au.place(bed, au.shaker(rng), t0 + k * beat / 4, (0.16 if k % 4 == 2 else 0.09) * lvl, -0.35)
            if bar >= 8:
                for b in (1, 3):
                    au.place(bed, au.clap(rng), t0 + b * beat, 0.24 * lvl)
        t = PLAY - 5.0                                               # clock ticks for the last five seconds
        while t < PLAY - 1e-6:
            left = PLAY - t
            au.place(bed, au.tick(1500 + 70 * (5 - left)), t, 0.15 + 0.03 * (5 - left))
            t += 0.25 if left <= 2.0 else 0.5
        rl = 4.0                                                     # riser into the buzzer
        env = (tt_(rl) / rl) ** 2
        au.place(bed, au.swept_noise(rng, rl, 300, 5000) * env, PLAY - rl, 0.30)
        au.place(bed, au.glide(220, 880, rl) * env, PLAY - rl, 0.10)
        au.place(notes, au.boom(rng, 0.9), PLAY, 0.60)               # buzzer
        au.place(notes, au.noise(rng, 0.5, 1500, 9000) * np.exp(-tt_(0.5) * 9), PLAY, 0.30)
        for j in range(16):                                          # tile count: ticks climbing in pitch
            au.place(notes, au.tick(900 * 2 ** (j / 16), 0.05), T_COUNT + j * COUNT_LEN / 16, 0.20 + 0.012 * j)

        # win chime: rising C major arpeggio on bells, low thump, shimmer, then a few sparkles
        for j, m in enumerate((72, 76, 79, 84)):
            au.place(notes, au.bell(au.hz(m), 2.2), T_REVEAL + j * 0.07, 0.42, (-0.5, -0.15, 0.15, 0.5)[j])
        au.place(notes, au.bell(au.hz(91), 2.0), T_REVEAL + 0.28, 0.24)
        au.place(notes, au.boom(rng, 1.2, 80, 40), T_REVEAL, 0.55)
        au.place(notes, au.noise(rng, 1.4, 3000, 12000) * np.exp(-tt_(1.4) * 4), T_REVEAL, 0.20)
        for m, pn in ((48, 0.0), (55, -0.4), (64, 0.4), (67, 0.0)):
            au.place(bed, au.pad(au.hz(m), T_RESET + 0.4 - T_REVEAL, 0.05, 1.2), T_REVEAL, 0.10, pn)
        t, k = T_REVEAL + 0.45, 0
        while t < T_RESET:
            m = rnd.choice((84, 86, 88, 91, 93, 96))
            au.place(notes, au.bell(au.hz(m), 0.8), t, 0.15 * 0.93 ** k, rnd.uniform(-0.7, 0.7))
            t += rnd.choice((0.125, 0.125, 0.25))
            k += 1
        au.place(notes, au.glide(520, 1040, 0.09, 0.05), T_LINE, 0.20)
        wl = DUR - T_RESET                                           # rewind whoosh into the loop point
        au.place(bed, au.swept_noise(rng, wl, 500, 6000) * (tt_(wl) / wl) ** 2, T_RESET, 0.60)

        mix_ = au.reverb(notes, rng, decay=1.0, wet=0.16) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]                    # tails past the end wrap to the start: seamless loop
        # small-speaker balance: less low-mid weight, more presence and air
        return au.eq(out, ((60, -2.0), (250, -2.5), (900, 0.0), (3000, 4.0), (8000, 6.0), (16000, 4.0)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        title = "Which colour wins the most territory? 4-ball paint battle"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "Four balls bounce around a tiled arena and paint every tile they roll over. "
                "Whoever holds the most tiles when the timer runs out wins.\n"
                "Three pickups drop in along the way and can swing it: GROW makes a ball bigger for a while, "
                "SPEED makes it faster, and BOMB paints a whole patch at once. The tiles are counted one by one "
                f"at the end. Physics, picture and sound are generated by code (seed {self.seed}), with no "
                "footage, samples or stock media.\n"
                "Did you call the winner?"),
            tags=["physics simulation", "bouncing balls", "territory battle", "colour battle", "color battle",
                  "paint battle", "satisfying", "made with code"],
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylised 2D animation of abstract shapes; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            drama=self.stats,
        )
